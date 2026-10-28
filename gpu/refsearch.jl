module RefSearch
# The search of CSets at d2927d3 (bagsearch!), kept as a reference that
# exposes its tables: phases() in stats.jl compares them with the GPU.
using Base: oneto
using CSets: CSets, Decomposition, Instance, Span, FVector, FScalar, invmap, invvtx, two, ispositive, AbstractSemiring, Boolean, Lawvere, Probability, szero, sone, splus, sprod, isintegral, isselective, valtype, permute
#
# The weights W are either nothing or a vector with one entry per sort
# c: either nothing (every image has weight 1) or a matrix of size
# |Y(c)| x |X(c)| whose entry (i, j) is the weight of mapping the j-th
# element of X of sort c to the i-th element of Y of sort c, both in
# the caller's numbering. A weight of 0 forbids the image.
#
# The weight of a homomorphism is the product of the weights of its
# images, and score computes the sum of the weights of all
# homomorphisms X -> Y.
#
function score(R::AbstractSemiring, W, X::Instance{I}, S::Span{I}, Y::Instance{I}; kw...) where {I}
    D = decomposition(X, S; kw...)
    return score(R, W, D, Y)
end

function score(R::AbstractSemiring, W, D::Decomposition{I}, Y::Instance{I}) where {I}
    value, h = search(R, W, D, Y, false, valtype(R, W))
    return value
end

#
# Find the score and a homomorphism attaining it, or nothing if the
# score is 0. The semiring must be selective.
#
function color(R::AbstractSemiring, W, X::Instance{I}, S::Span{I}, Y::Instance{I}; kw...) where {I}
    D = decomposition(X, S; kw...)
    return color(R, W, D, Y)
end

function color(R::AbstractSemiring, W, D::Decomposition{I}, Y::Instance{I}) where {I}
    @assert isselective(R)
    return search(R, W, D, Y, true, valtype(R, W))
end

#
# Find a homomorphism X -> Y, or nothing.
#
function color(W, X::Instance{I}, S::Span{I}, Y::Instance{I}; kw...) where {I}
    value, h = color(Boolean(), W, X, S, Y; kw...)
    return h
end

function color(X::Instance{I}, S::Span{I}, Y::Instance{I}; kw...) where {I}
    return color(nothing, X, S, Y; kw...)
end

function color(W, D::Decomposition{I}, Y::Instance{I}) where {I}
    value, h = color(Boolean(), W, D, Y)
    return h
end

function color(D::Decomposition{I}, Y::Instance{I}) where {I}
    return color(nothing, D, Y)
end

#
# The type of the weights: the element type of the first weight
# matrix, or the default for the semiring.
#

function search(R::AbstractSemiring, W, D::Decomposition{I}, Y::Instance{I}, witness::Bool, ::Type{T}) where {I, T}
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

    for t in oneto(nt)
        if !bagsearch!(R, D, Y, pre, base, typ, woff, wval, weighted, ord, kind, arg, cur, stop, bknd, bpos, barc, xchk, chka, chkp, chkd, xnar, narc, nari, naro, rlo, rhi, pos, bag, h, xtab, ncol, tab, ntab, vtab, tval, nval, coff, csel, t, true)
            return szero(R, T), nothing
        end
    end

    #
    # in the Boolean semiring every stored entry has value 1, so values
    # are not stored
    #
    if R isa Boolean
        value = sone(R, T)
    else
        value = tval[vtab[nt]]
    end

    if !witness
        return value, nothing
    end

    csel[nt] = one(I)

    for t in nt:-one(I):one(I)
        bagsearch!(R, D, Y, pre, base, typ, woff, wval, weighted, ord, kind, arg, cur, stop, bknd, bpos, barc, xchk, chka, chkp, chkd, xnar, narc, nari, naro, rlo, rhi, pos, bag, h, xtab, ncol, tab, ntab, vtab, tval, nval, coff, csel, t, false)
    end

    out = FVector{I}(undef, nu)

    for r in oneto(nu)
        out[D.res.adjncy[r]] = yord[h[r]]
    end

    return value, out
end

#
# Relabel Y so that its elements are grouped by type and each
# element's edges are listed in schema order. Returns the relabeled
# instance and the permutation from new labels to old.
#
function normalize(Y::Instance{I}) where {I}
    ny = Y.dom.nu
    nf = Y.dom.ne

    order = invvtx(Y).adjncy
    Z = permute(Y, order)

    arc    = FVector{I}(undef, nf)
    adjncy = FVector{I}(undef, nf)

    for y in oneto(ny)
        c = Z.vtx[y]
        pstrt = Z.dom.xadj[y]
        pstop = Z.dom.xadj[y + one(I)] - one(I)

        for p in pstrt:pstop
            a = Z.arc[p]
            q = pstrt + a - Z.cod.xadj[c]
            arc[q] = a
            adjncy[q] = Z.dom.adjncy[p]
        end
    end

    dom = Span{I}(ny, ny, nf, Z.dom.xadj, adjncy)
    return Instance{I}(dom, Z.cod, Z.vtx, arc), order
end

#
# The image of y under arc a, for Y normalized.
#
function image(Y::Instance{I}, y::I, a::I) where {I}
    return Y.dom.adjncy[Y.dom.xadj[y] + a - Y.cod.xadj[Y.vtx[y]]]
end

function bagsearch!(
        R::AbstractSemiring,
        D::Decomposition{I},
        Y::Instance{I},
        pre::Span{I},
        base::AbstractVector{I},
        typ::Span{I},
        woff::AbstractVector{I},
        wval::AbstractVector{T},
        ::Val{weighted},
        ord::AbstractVector{I},
        kind::AbstractVector{I},
        arg::AbstractVector{I},
        cur::AbstractVector{I},
        stop::AbstractVector{I},
        bknd::AbstractVector{I},
        bpos::AbstractVector{I},
        barc::AbstractVector{I},
        xchk::AbstractVector{I},
        chka::AbstractVector{I},
        chkp::AbstractVector{I},
        chkd::AbstractVector{Bool},
        xnar::AbstractVector{I},
        narc::AbstractVector{I},
        nari::AbstractVector{I},
        naro::AbstractVector{I},
        rlo::AbstractVector{I},
        rhi::AbstractVector{I},
        pos::AbstractVector{I},
        bag::AbstractVector{I},
        h::AbstractVector{I},
        xtab::AbstractVector{I},
        ncol::AbstractVector{I},
        tab::Vector{I},
        ntab::FScalar{I},
        vtab::AbstractVector{I},
        tval::Vector{T},
        nval::FScalar{I},
        coff::AbstractVector{I},
        csel::AbstractVector{I},
        t::I,
        forward::Bool,
    ) where {I, T, weighted}
    X = D.inst
    rstrt = D.res.xadj[t]
    rstop = D.res.xadj[t + one(I)] - one(I)
    sstrt = D.sep.xadj[t]
    sstop = D.sep.xadj[t + one(I)] - one(I)
    cstrt = D.tree.xadj[t]
    cstop = D.tree.xadj[t + one(I)] - one(I)
    s = sstop - sstrt + one(I)
    m = s + rstop - rstrt + one(I)
    #
    # order the bag: separator elements first, each part descending
    #
    k = zero(I)

    for p in sstop:-one(I):sstrt
        k += one(I)
        ord[k] = D.sep.adjncy[p]
    end

    for x in rstop:-one(I):rstrt
        k += one(I)
        ord[k] = x
    end

    for k in oneto(m)
        x = ord[k]
        bag[x] = t
        pos[x] = k
    end
    #
    # Plan the search. For each position k, record
    #
    #   - the checks on the element x = ord[k]: each edge between x and
    #     an element at an earlier position (or x itself), as its arc,
    #     the other position, and whether the edge leaves x; and
    #
    #   - where its candidates come from: the image of an earlier
    #     element pointing to x (3), the preimages of an earlier
    #     element x points to (2), or all elements of its type (1),
    #     with the position and arc that determine them.
    #
    # Every edge with both ends in the bag is an edge out of some bag
    # element, since bags are closed downward, so the checks are found
    # by scanning out-edges only. An edge w -> z is checked at the later
    # of the two positions.
    #
    for k in oneto(m)
        xchk[k + one(I)] = zero(I)
        bknd[k] = one(I)
        bpos[k] = zero(I)
        barc[k] = zero(I)
    end

    for k in oneto(m)
        w = ord[k]
        pstrt = X.dom.xadj[w]
        pstop = X.dom.xadj[w + one(I)] - one(I)

        for p in pstrt:pstop
            l = max(k, pos[X.dom.adjncy[p]])

            if l < m
                xchk[l + two(I)] += one(I)
            end
        end
    end

    xchk[one(I)] = q = one(I)

    for k in oneto(m)
        xchk[k + one(I)] = q += xchk[k + one(I)]
    end

    for k in oneto(m)
        w = ord[k]
        pstrt = X.dom.xadj[w]
        pstop = X.dom.xadj[w + one(I)] - one(I)

        for p in pstrt:pstop
            a = X.arc[p]
            j = pos[X.dom.adjncy[p]]

            if j <= k
                #
                # an edge out of w to an earlier element (or w itself)
                #
                r = xchk[k + one(I)]
                chka[r] = a
                chkp[r] = j
                chkd[r] = true
                xchk[k + one(I)] += one(I)

                if j < k && isone(bknd[k])
                    bknd[k] = two(I)
                    bpos[k] = j
                    barc[k] = a
                end
            else
                #
                # an edge into the later element at position j
                #
                r = xchk[j + one(I)]
                chka[r] = a
                chkp[r] = k
                chkd[r] = false
                xchk[j + one(I)] += one(I)

                if bknd[j] != two(I) + one(I)
                    bknd[j] = two(I) + one(I)
                    bpos[j] = k
                    barc[j] = a
                end
            end
        end
    end
    #
    # Each child's table is sorted by its separator in descending
    # order, which is the order in which this search assigns those
    # elements. For each child c and each component i of its key,
    # record the position at which component i is assigned. At that
    # position, the range of matching columns
    #
    #    rlo[o + i] ... rhi[o + i]
    #
    # is narrowed from the range for component i - 1.
    #
    for k in oneto(m)
        xnar[k + one(I)] = zero(I)
    end

    for j in cstrt:cstop
        c = D.tree.adjncy[j]
        pstrt = D.sep.xadj[c]
        pstop = D.sep.xadj[c + one(I)] - one(I)

        for p in pstrt:pstop
            k = pos[D.sep.adjncy[p]]

            if k < m
                xnar[k + two(I)] += one(I)
            end
        end
    end

    xnar[one(I)] = q = one(I)

    for k in oneto(m)
        xnar[k + one(I)] = q += xnar[k + one(I)]
    end

    o = one(I)

    for j in cstrt:cstop
        c = D.tree.adjncy[j]
        pstrt = D.sep.xadj[c]
        pstop = D.sep.xadj[c + one(I)] - one(I)
        rlo[o] = one(I)
        rhi[o] = ncol[c]
        coff[j] = o
        i = zero(I)

        for p in reverse(pstrt:pstop)
            i += one(I)
            k = pos[D.sep.adjncy[p]]
            l = xnar[k + one(I)]
            narc[l] = c
            nari[l] = i
            naro[l] = o
            xnar[k + one(I)] += one(I)
        end

        o += pstop - pstrt + two(I)
    end

    if forward
        xtab[t] = ntab[] + one(I)
        vtab[t] = nval[] + one(I)
        ncol[t] = zero(I)
    end

    if forward
        kmin = one(I)
    else
        kmin = s + one(I)
    end

    for k in oneto(kmin - one(I))
        narrow!(D, ord, xnar, narc, nari, naro, rlo, rhi, h, xtab, tab, k, zero(I))
    end

    if m < kmin
        if forward
            w = weight(R, D, typ, woff, wval, Val(weighted), ord, h, rlo, coff, vtab, tval, s, m, t)

            if w != szero(R, T)
                record!(R, tab, ntab, tval, nval, ncol, h, ord, s, t, w)
            end

            return ispositive(ncol[t])
        else
            select!(D, rlo, coff, csel, t)
            return true
        end
    end
    #
    # In the forward pass, acc sums the weights of the completions of
    # the current separator assignment. In the backward pass, target is
    # the weight of the entry chosen by the parent.
    #
    acc = szero(R, T)

    if forward || R isa Boolean
        target = sone(R, T)
    else
        target = tval[vtab[t] + csel[t] - one(I)]
    end

    k = kmin
    fresh = true

    while kmin <= k
        x = ord[k]

        if fresh
            #
            # candidates for x, in order of preference:
            #
            #   3. the image of an assigned element pointing to x
            #   4. the distinct values in the smallest child range
            #   2. the preimages of an assigned element x points to
            #   1. all elements of the type of x
            #
            fresh = false
            kind[k] = bknd[k]

            if kind[k] == two(I) + one(I)
                arg[k] = image(Y, h[ord[bpos[k]]], barc[k])
                cur[k] = one(I)
                stop[k] = one(I)
            else
                if kind[k] == two(I)
                    a = barc[k]
                    l = base[a] + h[ord[bpos[k]]] - typ.xadj[Y.cod.adjncy[a]]
                    cur[k] = pre.xadj[l]
                    stop[k] = pre.xadj[l + one(I)] - one(I)
                else
                    u = X.vtx[x]
                    cur[k] = typ.xadj[u]
                    stop[k] = typ.xadj[u + one(I)] - one(I)
                end

                jstrt = xnar[k]
                jstop = xnar[k + one(I)] - one(I)

                for j in jstrt:jstop
                    i = nari[j]
                    o = naro[j]
                    lo = rlo[o + i - one(I)]
                    hi = rhi[o + i - one(I)]

                    if kind[k] != two(I) + two(I) || hi - lo < stop[k] - cur[k]
                        kind[k] = two(I) + two(I)
                        arg[k] = j
                        cur[k] = lo
                        stop[k] = hi
                    end
                end
            end
        end

        found = false

        while cur[k] <= stop[k]
            c = cur[k]
            cur[k] += one(I)

            if isone(kind[k])
                y = c
            elseif kind[k] == two(I)
                y = pre.adjncy[c]
            elseif kind[k] == two(I) + one(I)
                y = arg[k]
            else
                #
                # walk the child range: y is the value at column c, and
                # the next candidate is the first column after the run
                # of columns with value y
                #
                j = arg[k]
                l = narc[j]
                i = nari[j]
                o = naro[j]
                w = D.sep.xadj[l + one(I)] - D.sep.xadj[l]
                b = xtab[l] + i - one(I)
                y = tab[b + (c - one(I)) * w]
                lo = c
                hi = c + one(I)
                step = one(I)

                while hi <= stop[k] && tab[b + (hi - one(I)) * w] <= y
                    lo = hi
                    hi = c + step
                    step += step
                end

                hi = min(hi, stop[k] + one(I))

                while lo + one(I) < hi
                    mid = (lo + hi) >>> one(I)

                    if tab[b + (mid - one(I)) * w] <= y
                        lo = mid
                    else
                        hi = mid
                    end
                end

                rlo[o + i] = c
                rhi[o + i] = lo
                cur[k] = lo + one(I)
            end

            if !ispositive(y)
                continue
            end

            if weighted
                l = woff[x]

                if l < woff[x + one(I)] && wval[l + y - typ.xadj[X.vtx[x]]] == szero(R, T)
                    continue
                end
            end

            h[x] = y
            valid = true
            #
            # edges between x and earlier elements
            #
            qstrt = xchk[k]
            qstop = xchk[k + one(I)] - one(I)

            for q in qstrt:qstop
                z = h[ord[chkp[q]]]

                if chkd[q]
                    valid = image(Y, y, chka[q]) == z
                else
                    valid = image(Y, z, chka[q]) == y
                end

                if !valid
                    break
                end
            end
            #
            # child tables
            #
            if valid && xnar[k] < xnar[k + one(I)]
                if kind[k] == two(I) + two(I)
                    skip = arg[k]
                else
                    skip = zero(I)
                end

                valid = narrow!(D, ord, xnar, narc, nari, naro, rlo, rhi, h, xtab, tab, k, skip)
            end

            if valid
                found = true
                break
            end
        end

        if found
            if k < m
                k += one(I)
                fresh = true
            else
                w = weight(R, D, typ, woff, wval, Val(weighted), ord, h, rlo, coff, vtab, tval, s, m, t)

                if forward
                    acc = splus(R, acc, w)
                    #
                    # the separator assignment is finished when there is
                    # no residual, or when the sum can no longer change
                    #
                    if k == s || (isintegral(R) && acc == sone(R, T))
                        if acc != szero(R, T)
                            record!(R, tab, ntab, tval, nval, ncol, h, ord, s, t, acc)
                        end

                        acc = szero(R, T)

                        if iszero(s)
                            return true
                        end

                        k = s
                    end
                elseif w == target
                    select!(D, rlo, coff, csel, t)
                    return true
                end
            end
        else
            k -= one(I)
            #
            # leaving the residual: the separator assignment is finished
            #
            if forward && k == s && s < m
                if acc != szero(R, T)
                    record!(R, tab, ntab, tval, nval, ncol, h, ord, s, t, acc)
                end

                acc = szero(R, T)
            end
        end
    end

    return forward && ispositive(ncol[t])
end

#
# The weight of a complete assignment of the bag at node t: the
# product of the weights of the residual elements and of the matched
# entries of the children's tables.
#
function weight(
        R::AbstractSemiring,
        D::Decomposition{I},
        typ::Span{I},
        woff::AbstractVector{I},
        wval::AbstractVector{T},
        ::Val{weighted},
        ord::AbstractVector{I},
        h::AbstractVector{I},
        rlo::AbstractVector{I},
        coff::AbstractVector{I},
        vtab::AbstractVector{I},
        tval::Vector{T},
        s::I,
        m::I,
        t::I,
    ) where {I, T, weighted}
    w = sone(R, T)
    #
    # in the Boolean semiring, zero weights are never chosen, so every
    # complete assignment has weight 1
    #
    if R isa Boolean
        return w
    end

    if weighted
        for k in s + one(I):m
            x = ord[k]
            l = woff[x]

            if l < woff[x + one(I)]
                w = sprod(R, w, wval[l + h[x] - typ.xadj[D.inst.vtx[x]]])
            end
        end
    end

    cstrt = D.tree.xadj[t]
    cstop = D.tree.xadj[t + one(I)] - one(I)

    for j in cstrt:cstop
        c = D.tree.adjncy[j]
        n = D.sep.xadj[c + one(I)] - D.sep.xadj[c]
        w = sprod(R, w, tval[vtab[c] + rlo[coff[j] + n] - one(I)])
    end

    return w
end

#
# Append a column to the table of node t: the separator assignment and
# its value. The tables grow geometrically.
#
function record!(
        R::AbstractSemiring,
        tab::Vector{I},
        ntab::FScalar{I},
        tval::Vector{T},
        nval::FScalar{I},
        ncol::AbstractVector{I},
        h::AbstractVector{I},
        ord::AbstractVector{I},
        s::I,
        t::I,
        value::T,
    ) where {I, T}
    n = ntab[]

    if length(tab) < n + s
        resize!(tab, max(2 * length(tab), n + s))
    end

    for i in oneto(s)
        tab[n + i] = h[ord[i]]
    end

    ntab[] = n + s
    ncol[t] += one(I)

    if !(R isa Boolean)
        n = nval[]

        if length(tval) < n + one(I)
            resize!(tval, 2 * length(tval))
        end

        tval[n + one(I)] = value
        nval[] = n + one(I)
    end

    return
end

#
# Record, for each child of node t, the column of its table matched by
# the current assignment.
#
function select!(D::Decomposition{I}, rlo::AbstractVector{I}, coff::AbstractVector{I}, csel::AbstractVector{I}, t::I) where {I}
    cstrt = D.tree.xadj[t]
    cstop = D.tree.xadj[t + one(I)] - one(I)

    for j in cstrt:cstop
        c = D.tree.adjncy[j]
        n = D.sep.xadj[c + one(I)] - D.sep.xadj[c]
        csel[c] = rlo[coff[j] + n]
    end

    return
end

#
# Narrow the child ranges for the element assigned at position k.
# Returns false if some range becomes empty.
#
function narrow!(
        D::Decomposition{I},
        ord::AbstractVector{I},
        xnar::AbstractVector{I},
        narc::AbstractVector{I},
        nari::AbstractVector{I},
        naro::AbstractVector{I},
        rlo::AbstractVector{I},
        rhi::AbstractVector{I},
        h::AbstractVector{I},
        xtab::AbstractVector{I},
        tab::Vector{I},
        k::I,
        skip::I,
    ) where {I}
    v = h[ord[k]]
    jstrt = xnar[k]
    jstop = xnar[k + one(I)] - one(I)

    for j in jstrt:jstop
        if j == skip
            continue
        end

        c = narc[j]
        i = nari[j]
        o = naro[j]
        s = D.sep.xadj[c + one(I)] - D.sep.xadj[c]
        b = xtab[c] + i - one(I)
        #
        # first column in range with component i at least v
        #
        lo = rlo[o + i - one(I)]
        hi = rhi[o + i - one(I)] + one(I)

        while lo < hi
            mid = (lo + hi) >>> one(I)

            if tab[b + (mid - one(I)) * s] < v
                lo = mid + one(I)
            else
                hi = mid
            end
        end

        strt = lo
        #
        # first column in range with component i greater than v
        #
        hi = rhi[o + i - one(I)] + one(I)

        while lo < hi
            mid = (lo + hi) >>> one(I)

            if tab[b + (mid - one(I)) * s] <= v
                lo = mid + one(I)
            else
                hi = mid
            end
        end

        if strt == lo
            return false
        end

        rlo[o + i] = strt
        rhi[o + i] = lo - one(I)
    end

    return true
end
end
