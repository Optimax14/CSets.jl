#
# A copy of CSets.color(D, Y) that times the forward and backward
# passes separately and records the shape of the decomposition.
#
include(joinpath(@__DIR__, "refsearch.jl"))
using .RefSearch: bagsearch!, normalize
using CSets: CSets, Decomposition, FVector, FScalar, invmap, invvtx, two, AbstractSemiring, Boolean, Lawvere, Probability, szero, sone, valtype
using Base: oneto

function depths(D::Decomposition{I}) where {I}
    nt = D.tree.nu
    dep = zeros(Int, nt)
    for t in nt:-1:1
        for j in D.tree.xadj[t]:D.tree.xadj[t + 1] - 1
            dep[D.tree.adjncy[j]] = dep[t] + 1
        end
    end
    return dep
end

function phases(D::Decomposition{I}, Y; R=Boolean(), W=nothing, T=valtype(R, W)) where {I}
    t0 = time_ns()
    Y, yord = normalize(Y)
    X = D.inst
    nu = X.dom.nu
    nt = D.tree.nu
    ny = Y.dom.nu
    nf = Y.dom.ne
    na = Y.cod.ne

    mbag = zero(I)
    mnar = zero(I)
    mchk = zero(I)

    for t in oneto(nt)
        rstrt = D.res.xadj[t]
        rstop = D.res.xadj[t + one(I)] - one(I)
        sstrt = D.sep.xadj[t]
        sstop = D.sep.xadj[t + one(I)] - one(I)
        cstrt = D.tree.xadj[t]
        cstop = D.tree.xadj[t + one(I)] - one(I)
        n = zero(I)
        d = zero(I)

        for j in cstrt:cstop
            c = D.tree.adjncy[j]
            n += D.sep.xadj[c + one(I)] - D.sep.xadj[c] + one(I)
        end
        #
        # every check is an edge with both ends in the bag, recorded
        # once at each end
        #
        for x in rstrt:rstop
            d += X.dom.xadj[x + one(I)] - X.dom.xadj[x]
        end

        for p in sstrt:sstop
            x = D.sep.adjncy[p]
            d += X.dom.xadj[x + one(I)] - X.dom.xadj[x]
        end

        mbag = max(mbag, rstop - rstrt + sstop - sstrt + two(I))
        mnar = max(mnar, n)
        mchk = max(mchk, two(I) * d)
    end

    ysrc = FVector{I}(undef, nf)
    base = FVector{I}(undef, na + one(I))
    ykey = FVector{I}(undef, nf)
    ord  = FVector{I}(undef, mbag)
    kind = FVector{I}(undef, mbag)
    arg  = FVector{I}(undef, mbag)
    cur  = FVector{I}(undef, mbag)
    stop = FVector{I}(undef, mbag)
    xnar = FVector{I}(undef, mbag + one(I))
    bknd = FVector{I}(undef, mbag)
    bpos = FVector{I}(undef, mbag)
    barc = FVector{I}(undef, mbag)
    xchk = FVector{I}(undef, mbag + one(I))
    chka = FVector{I}(undef, mchk)
    chkp = FVector{I}(undef, mchk)
    chkd = FVector{Bool}(undef, mchk)
    narc = FVector{I}(undef, mnar)
    nari = FVector{I}(undef, mnar)
    naro = FVector{I}(undef, mnar)
    rlo  = FVector{I}(undef, mnar)
    rhi  = FVector{I}(undef, mnar)
    pos  = FVector{I}(undef, nu)
    bag  = FVector{I}(undef, nu)
    h    = FVector{I}(undef, nu)
    xtab = FVector{I}(undef, nt)
    ncol = FVector{I}(undef, nt)
    tab  = Vector{I}(undef, 1024)
    ntab = FScalar{I}(undef)
    ntab[] = zero(I)
    vtab = FVector{I}(undef, nt)
    tval = Vector{T}(undef, 1024)
    nval = FScalar{I}(undef)
    nval[] = zero(I)
    coff = FVector{I}(undef, max(D.tree.ne, one(I)))
    csel = FVector{I}(undef, nt)

    for x in oneto(nu)
        bag[x] = zero(I)
    end

    for y in oneto(ny)
        pstrt = Y.dom.xadj[y]
        pstop = Y.dom.xadj[y + one(I)] - one(I)

        for p in pstrt:pstop
            ysrc[p] = y
        end
    end

    typ = invvtx(Y)
    #
    # the preimages of y under arc a are the column
    #
    #    base[a] + y - typ.xadj[tgt(a)]
    #
    base[one(I)] = one(I)

    for a in oneto(na)
        d = Y.cod.adjncy[a]
        base[a + one(I)] = base[a] + typ.xadj[d + one(I)] - typ.xadj[d]
    end

    for e in oneto(nf)
        a = Y.arc[e]
        ykey[e] = base[a] + Y.dom.adjncy[e] - typ.xadj[Y.cod.adjncy[a]]
    end

    pre = invmap(base[na + one(I)] - one(I), nf, ykey)

    for p in oneto(nf)
        pre.adjncy[p] = ysrc[pre.adjncy[p]]
    end
    #
    # Relabel the weights: element r of X gets the column
    #
    #    wval[woff[r]] ... wval[woff[r + 1] - 1]
    #
    # indexed by the elements of Y of its sort, or an empty column if
    # every image is allowed.
    #
    ns = X.cod.nu
    torg = FVector{I}(undef, nu)
    rank = FVector{I}(undef, nu)
    nsrt = FVector{I}(undef, ns)
    woff = FVector{I}(undef, nu + one(I))

    for r in oneto(nu)
        torg[D.res.adjncy[r]] = X.vtx[r]
    end

    for c in oneto(ns)
        nsrt[c] = zero(I)
    end

    for v in oneto(nu)
        c = torg[v]
        nsrt[c] += one(I)
        rank[v] = nsrt[c]
    end

    woff[one(I)] = one(I)

    for r in oneto(nu)
        c = X.vtx[r]

        if isnothing(W) || isnothing(W[c])
            woff[r + one(I)] = woff[r]
        else
            woff[r + one(I)] = woff[r] + typ.xadj[c + one(I)] - typ.xadj[c]
        end
    end

    wval = FVector{T}(undef, woff[nu + one(I)] - one(I))
    weighted = Val(woff[nu + one(I)] > one(I))

    for r in oneto(nu)
        c = X.vtx[r]
        l = woff[r]

        if l < woff[r + one(I)]
            M = W[c]
            j = rank[D.res.adjncy[r]]

            for i in oneto(woff[r + one(I)] - l)
                wval[l + i - one(I)] = M[i, j]
            end
        end
    end

    t1 = time_ns()
    ok = true

    for t in oneto(nt)
        if !bagsearch!(R, D, Y, pre, base, typ, woff, wval, weighted, ord, kind, arg, cur, stop, bknd, bpos, barc, xchk, chka, chkp, chkd, xnar, narc, nari, naro, rlo, rhi, pos, bag, h, xtab, ncol, tab, ntab, vtab, tval, nval, coff, csel, t, true)
            ok = false
            break
        end
    end

    t2 = time_ns()
    return (; setup=(t1 - t0) / 1e9, forward=(t2 - t1) / 1e9, backward=0.0, ok, ncol=collect(ncol), ntab=ntab[],
        xtab=collect(xtab), tab=tab[1:ntab[]], s=[D.sep.xadj[t + 1] - D.sep.xadj[t] for t in 1:nt])
end


function shape(D::Decomposition)
    nt = D.tree.nu
    res = [D.res.xadj[t + 1] - D.res.xadj[t] for t in 1:nt]
    sep = [D.sep.xadj[t + 1] - D.sep.xadj[t] for t in 1:nt]
    dep = depths(D)
    nch = [D.tree.xadj[t + 1] - D.tree.xadj[t] for t in 1:nt]
    lev = zeros(Int, maximum(dep) + 1)
    for d in dep
        lev[d + 1] += 1
    end
    return (; nt, maxbag=maximum(res .+ sep), meanbag=sum(res .+ sep) / nt, maxsep=maximum(sep),
        meansep=sum(sep) / nt, depth=maximum(dep) + 1, maxlevel=maximum(lev), leaves=count(iszero, nch))
end

function profile(name, X, S, Y; alg=MF(), reps=2)
    color(X, S, Y; alg)  # warm
    td = minimum(@elapsed(decomposition(X, S; alg)) for _ in 1:reps)
    D = decomposition(X, S; alg)
    ph = phases(D, Y)
    for _ in 2:reps
        p = phases(D, Y)
        ph = p.forward + p.backward < ph.forward + ph.backward ? p : ph
    end
    sh = shape(D)
    nc = ph.ncol[1:findlast(>(0), ph.ncol) === nothing ? 0 : end]
    println(rpad(name, 34), " nX=", X.dom.nu, " |Y|=", Y.dom.nu,
        " decomp=", round(td, digits=4), " fwd=", round(ph.forward, digits=4),
        " bwd=", round(ph.backward, digits=4), " sat=", ph.ok)
    println("   tree: nodes=", sh.nt, " depth=", sh.depth, " maxlevel=", sh.maxlevel, " leaves=", sh.leaves,
        " bag max/mean=", sh.maxbag, "/", round(sh.meanbag, digits=1),
        " sep max/mean=", sh.maxsep, "/", round(sh.meansep, digits=1),
        " tab cols max/mean=", maximum(ph.ncol), "/", round(sum(ph.ncol) / length(ph.ncol), digits=1),
        " tab words=", ph.ntab)
    return (; td, ph, sh)
end
