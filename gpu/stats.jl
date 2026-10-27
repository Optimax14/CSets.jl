#
# A copy of CSets.color(D, Y) that times the forward and backward
# passes separately and records the shape of the decomposition.
#
using CSets: CSets, Decomposition, FVector, FScalar, invmap, invvtx, normalize, bagsearch!
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

function phases(D::Decomposition{I}, Y) where {I}
    t0 = time_ns()
    Y, yord = normalize(Y)
    X = D.inst
    nu = X.dom.nu
    nt = D.tree.nu
    ny = Y.dom.nu
    nf = Y.dom.ne
    na = Y.cod.ne
    mbag = zero(I); mnar = zero(I); mchk = zero(I)
    for t in oneto(nt)
        rstrt = D.res.xadj[t]; rstop = D.res.xadj[t + one(I)] - one(I)
        sstrt = D.sep.xadj[t]; sstop = D.sep.xadj[t + one(I)] - one(I)
        n = zero(I); d = zero(I)
        for j in D.tree.xadj[t]:D.tree.xadj[t + one(I)] - one(I)
            c = D.tree.adjncy[j]
            n += D.sep.xadj[c + one(I)] - D.sep.xadj[c] + one(I)
        end
        for x in rstrt:rstop
            d += X.dom.xadj[x + one(I)] - X.dom.xadj[x]
        end
        for p in sstrt:sstop
            x = D.sep.adjncy[p]
            d += X.dom.xadj[x + one(I)] - X.dom.xadj[x]
        end
        mbag = max(mbag, rstop - rstrt + sstop - sstrt + 2)
        mnar = max(mnar, n)
        mchk = max(mchk, 2d)
    end
    ysrc = FVector{I}(undef, nf); base = FVector{I}(undef, na + 1); ykey = FVector{I}(undef, nf)
    ord = FVector{I}(undef, mbag); kind = FVector{I}(undef, mbag); arg = FVector{I}(undef, mbag)
    cur = FVector{I}(undef, mbag); stop = FVector{I}(undef, mbag); xnar = FVector{I}(undef, mbag + 1)
    bknd = FVector{I}(undef, mbag); bpos = FVector{I}(undef, mbag); barc = FVector{I}(undef, mbag)
    xchk = FVector{I}(undef, mbag + 1); chka = FVector{I}(undef, mchk); chkp = FVector{I}(undef, mchk)
    chkd = FVector{Bool}(undef, mchk); narc = FVector{I}(undef, mnar); nari = FVector{I}(undef, mnar)
    naro = FVector{I}(undef, mnar); rlo = FVector{I}(undef, mnar); rhi = FVector{I}(undef, mnar)
    pos = FVector{I}(undef, nu); bag = FVector{I}(undef, nu); h = FVector{I}(undef, nu)
    xtab = FVector{I}(undef, nt); ncol = FVector{I}(undef, nt); tab = Vector{I}(undef, 1024)
    ntab = FScalar{I}(undef); ntab[] = zero(I)
    fill!(bag, zero(I))
    for y in oneto(ny)
        for p in Y.dom.xadj[y]:Y.dom.xadj[y + one(I)] - one(I)
            ysrc[p] = y
        end
    end
    typ = invvtx(Y)
    base[1] = 1
    for a in oneto(na)
        d = Y.cod.adjncy[a]
        base[a + 1] = base[a] + typ.xadj[d + 1] - typ.xadj[d]
    end
    for e in oneto(nf)
        a = Y.arc[e]
        ykey[e] = base[a] + Y.dom.adjncy[e] - typ.xadj[Y.cod.adjncy[a]]
    end
    pre = invmap(base[na + 1] - 1, nf, ykey)
    for p in oneto(nf)
        pre.adjncy[p] = ysrc[pre.adjncy[p]]
    end
    t1 = time_ns()
    ok = true
    for t in oneto(nt)
        if !bagsearch!(D, Y, pre, base, typ, ord, kind, arg, cur, stop, bknd, bpos, barc, xchk, chka, chkp, chkd, xnar, narc, nari, naro, rlo, rhi, pos, bag, h, xtab, ncol, tab, ntab, t, true)
            ok = false
            break
        end
    end
    t2 = time_ns()
    if ok
        for t in nt:-1:1
            bagsearch!(D, Y, pre, base, typ, ord, kind, arg, cur, stop, bknd, bpos, barc, xchk, chka, chkp, chkd, xnar, narc, nari, naro, rlo, rhi, pos, bag, h, xtab, ncol, tab, ntab, t, false)
        end
    end
    t3 = time_ns()
    return (; setup=(t1 - t0) / 1e9, forward=(t2 - t1) / 1e9, backward=(t3 - t2) / 1e9, ok,
        ncol=collect(ncol), ntab=ntab[], xtab=collect(xtab), tab=tab[1:ntab[]], s=[D.sep.xadj[t + 1] - D.sep.xadj[t] for t in 1:nt])
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
