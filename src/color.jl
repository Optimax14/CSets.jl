function color(X::Instance{I}, S::Span{I}, Y::Instance{I}; kw...) where {I}
    D = decomposition(X, S; kw...)
    return color(D, Y)
end

function color(D::Decomposition{I}, Y::Instance{I}) where {I}
    Y, yord = normalize(Y)
    X = D.inst
    nu = X.dom.nu
    ne = X.dom.ne
    nt = D.tree.nu
    ny = Y.dom.nu
    nf = Y.dom.ne
    na = Y.cod.ne

    mbag = zero(I)
    mnar = zero(I)

    for t in oneto(nt)
        nres = D.res.xadj[t + one(I)] - D.res.xadj[t]
        nsep = D.sep.xadj[t + one(I)] - D.sep.xadj[t]
        cstrt = D.tree.xadj[t]
        cstop = D.tree.xadj[t + one(I)] - one(I)
        n = zero(I)

        for j in cstrt:cstop
            c = D.tree.adjncy[j]
            n += D.sep.xadj[c + one(I)] - D.sep.xadj[c] + one(I)
        end

        mbag = max(mbag, nres + nsep)
        mnar = max(mnar, n)
    end

    xsrc = FVector{I}(undef, ne)
    ysrc = FVector{I}(undef, nf)
    base = FVector{I}(undef, na + one(I))
    ykey = FVector{I}(undef, nf)
    ord  = FVector{I}(undef, mbag)
    kind = FVector{I}(undef, mbag)
    arg  = FVector{I}(undef, mbag)
    cur  = FVector{I}(undef, mbag)
    stop = FVector{I}(undef, mbag)
    xnar = FVector{I}(undef, mbag + one(I))
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

    for x in oneto(nu)
        pstrt = X.dom.xadj[x]
        pstop = X.dom.xadj[x + one(I)] - one(I)
        bag[x] = zero(I)

        for p in pstrt:pstop
            xsrc[p] = x
        end
    end

    for y in oneto(ny)
        pstrt = Y.dom.xadj[y]
        pstop = Y.dom.xadj[y + one(I)] - one(I)

        for p in pstrt:pstop
            ysrc[p] = y
        end
    end

    xin = invmap(nu, ne, X.dom.adjncy)
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

    for t in oneto(nt)
        if !bagsearch!(D, Y, xsrc, xin, pre, base, typ, ord, kind, arg, cur, stop, xnar, narc, nari, naro, rlo, rhi, pos, bag, h, xtab, ncol, tab, ntab, t, true)
            return
        end
    end

    for t in nt:-one(I):one(I)
        bagsearch!(D, Y, xsrc, xin, pre, base, typ, ord, kind, arg, cur, stop, xnar, narc, nari, naro, rlo, rhi, pos, bag, h, xtab, ncol, tab, ntab, t, false)
    end

    out = FVector{I}(undef, nu)

    for r in oneto(nu)
        out[D.res.adjncy[r]] = yord[h[r]]
    end

    return out
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
        D::Decomposition{I},
        Y::Instance{I},
        xsrc::AbstractVector{I},
        xin::Span{I},
        pre::Span{I},
        base::AbstractVector{I},
        typ::Span{I},
        ord::AbstractVector{I},
        kind::AbstractVector{I},
        arg::AbstractVector{I},
        cur::AbstractVector{I},
        stop::AbstractVector{I},
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
        t::I,
        forward::Bool,
    ) where {I}
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
        ncol[t] = zero(I)
    end

    if forward
        kmin = one(I)
    else
        kmin = s + one(I)
    end

    if m < kmin
        if forward
            ncol[t] = one(I)
        end

        return true
    end

    for k in oneto(kmin - one(I))
        narrow!(D, ord, xnar, narc, nari, naro, rlo, rhi, h, xtab, tab, k, zero(I))
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
            kind[k] = zero(I)
            qstrt = xin.xadj[x]
            qstop = xin.xadj[x + one(I)] - one(I)

            for q in qstrt:qstop
                e = xin.adjncy[q]
                w = xsrc[e]

                if bag[w] == t && pos[w] < k
                    kind[k] = two(I) + one(I)
                    arg[k] = image(Y, h[w], X.arc[e])
                    cur[k] = one(I)
                    stop[k] = one(I)
                    break
                end
            end

            if iszero(kind[k])
                u = X.vtx[x]
                kind[k] = one(I)
                cur[k] = typ.xadj[u]
                stop[k] = typ.xadj[u + one(I)] - one(I)
                pstrt = X.dom.xadj[x]
                pstop = X.dom.xadj[x + one(I)] - one(I)

                for p in pstrt:pstop
                    z = X.dom.adjncy[p]

                    if bag[z] == t && pos[z] < k
                        a = X.arc[p]
                        l = base[a] + h[z] - typ.xadj[Y.cod.adjncy[a]]
                        kind[k] = two(I)
                        cur[k] = pre.xadj[l]
                        stop[k] = pre.xadj[l + one(I)] - one(I)
                        break
                    end
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

            h[x] = y
            valid = true
            #
            # edges out of x
            #
            pstrt = X.dom.xadj[x]
            pstop = X.dom.xadj[x + one(I)] - one(I)

            for p in pstrt:pstop
                z = X.dom.adjncy[p]

                if bag[z] == t && pos[z] <= k && image(Y, y, X.arc[p]) != h[z]
                    valid = false
                    break
                end
            end
            #
            # edges into x
            #
            if valid
                qstrt = xin.xadj[x]
                qstop = xin.xadj[x + one(I)] - one(I)

                for q in qstrt:qstop
                    e = xin.adjncy[q]
                    w = xsrc[e]

                    if bag[w] == t && pos[w] < k && image(Y, h[w], X.arc[e]) != y
                        valid = false
                        break
                    end
                end
            end
            #
            # child tables
            #
            if valid
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
            elseif forward
                #
                # append the column, growing tab geometrically
                #
                n = ntab[]

                if length(tab) < n + s
                    resize!(tab, max(2 * length(tab), n + s))
                end

                for i in oneto(s)
                    tab[n + i] = h[ord[i]]
                end

                ntab[] = n + s

                ncol[t] += one(I)

                if iszero(s)
                    return true
                end

                k = s
            else
                return true
            end
        else
            k -= one(I)
        end
    end

    return forward && ispositive(ncol[t])
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
