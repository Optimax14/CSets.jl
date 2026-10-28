struct Contraction{I}
    #
    # We represent contractions as diagrams D: I → Set:
    #
    #   ┌───────────────────┐
    #   │          E        │
    #   │          ⇊        │
    #   │    C  ←  S  →  B  │
    #   └───────────────────┘
    #
    # where
    #
    #   - B is the bag of a node
    #   - E is a set of edges
    #   - C is the set of children of the node
    #   - S is a set of separator incidences
    #
    # This is the local part of a decomposition at a node: a tree with
    # two levels, whose children have empty residuals. The bag is
    #
    #    B = {1, ..., m}
    #
    # where the residual is {1, ..., nr} and the separator is
    # {nr + 1, ..., m}.
    #
    # # Attributes
    #
    # The size of the residual:
    #
    #    nr
    #
    nr::I
    #
    # The instance, restricted to the bag:
    #
    #    E ⇉ B
    #
    inst::Instance{I}
    #
    # The separators of the children:
    #
    #    C ← S → B
    #
    sep::Span{I}
end

#
# The local problem, forward. The arguments are the table of each child
# j, with ccol[j] columns, keys key[ckey[j]] ... and values
# val[cval[j]] ..., and the weights of each element x of the bag,
# wgt[xwgt[x]] .... Write the table of every assignment of the
# separator that extends to the bag, with its value: the sum over its
# extensions of the product of the residual weights and the matched
# entries of the children's tables. The keys are written after
# okey[ko] and the values after oval[vo]; nothing before is touched,
# and the arrays grow geometrically. Returns the number of columns.
#
# Unless fac is nothing, also write after fac[fo] a back-pointer for
# each column: the residual assignment of an extension attaining its
# value, and the matched column of each child's table. This requires a
# selective semiring.
#
# The search assigns the bag from the end: first the separator, then
# the residual. Each column is the separator assignment in bag order,
# and the columns are written in colexicographic order.
#
function contract!(
        R::AbstractSemiring,
        C::Contraction{I},
        Y::Instance{I},
        pre::Span{I},
        base::AbstractVector{I},
        typ::Span{I},
        xwgt,
        wgt,
        ckey::AbstractVector{I},
        cval::AbstractVector{I},
        ccol::AbstractVector{I},
        key::Vector{I},
        val::Vector{T},
        okey::Vector{I},
        oval::Vector{T},
        ko::I,
        vo::I,
        fac,
        fo::I,
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
        coff::AbstractVector{I},
        h::AbstractVector{I},
    ) where {I, T}
    nr = C.nr
    m = C.inst.dom.nu
    s = m - nr
    n = zero(I)
    plan!(C, xchk, chka, chkp, chkd, bknd, bpos, barc)
    plannarrow!(C, xnar, narc, nari, naro, rlo, rhi, coff, ccol)

    if iszero(m)
        w = weight(R, C, typ, xwgt, wgt, h, rlo, coff, cval, val)

        if w != szero(R, T)
            trace!(C, fac, fo, h, rlo, coff, n)
            record!(R, okey, oval, ko, vo, h, nr, s, n, w)
            n += one(I)
        end

        return n
    end
    #
    # acc sums the weights of the extensions of the current separator
    # assignment
    #
    acc = szero(R, T)
    k = m
    fresh = true

    while k <= m
        if fresh
            fresh = false
            candidates!(Y, pre, base, typ, C.inst.vtx, h, kind, arg, cur, stop, bknd, bpos, barc, xnar, nari, naro, rlo, rhi, k)
        end

        if advance!(R, C, Y, pre, typ, xwgt, wgt, kind, arg, cur, stop, xchk, chka, chkp, chkd, xnar, narc, nari, naro, rlo, rhi, h, ckey, key, k)
            if k > one(I)
                k -= one(I)
                fresh = true
            else
                w = weight(R, C, typ, xwgt, wgt, h, rlo, coff, cval, val)
                acc = splus(R, acc, w)
                #
                # in a selective semiring, the extension attains the sum
                # so far if the sum equals its weight
                #
                if !isnothing(fac) && acc == w
                    trace!(C, fac, fo, h, rlo, coff, n)
                end
                #
                # the separator assignment is finished when there is no
                # residual, or when the sum can no longer change
                #
                if iszero(nr) || (isintegral(R) && acc == sone(R, T))
                    if acc != szero(R, T)
                        record!(R, okey, oval, ko, vo, h, nr, s, n, acc)
                        n += one(I)
                    end

                    acc = szero(R, T)

                    if iszero(s)
                        return n
                    end

                    k = nr + one(I)
                end
            end
        else
            k += one(I)
            #
            # leaving the residual: the separator assignment is finished
            #
            if k == nr + one(I) && ispositive(nr)
                if acc != szero(R, T)
                    record!(R, okey, oval, ko, vo, h, nr, s, n, acc)
                    n += one(I)
                end

                acc = szero(R, T)
            end
        end
    end

    return n
end

#
# Assign the next valid candidate to the element k: one with nonzero
# weight, consistent with every edge to an element assigned earlier,
# and matching the children's tables. Returns false when the
# candidates are exhausted.
#
@inline function advance!(
        R::AbstractSemiring,
        C::Contraction{I},
        Y::Instance{I},
        pre::Span{I},
        typ::Span{I},
        xwgt,
        wgt,
        kind::AbstractVector{I},
        arg::AbstractVector{I},
        cur::AbstractVector{I},
        stop::AbstractVector{I},
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
        h::AbstractVector{I},
        ckey::AbstractVector{I},
        key::Vector{I},
        k::I,
    ) where {I}
    while cur[k] <= stop[k]
        y = candidate!(C, pre, key, ckey, narc, nari, naro, rlo, rhi, kind, arg, cur, stop, k)

        if !ispositive(y)
            continue
        end

        if !isnothing(wgt)
            l = xwgt[k]

            if l < xwgt[k + one(I)] && wgt[l + y - typ.xadj[C.inst.vtx[k]]] == szero(R, eltype(wgt))
                continue
            end
        end

        h[k] = y

        if !consistent(Y, h, xchk, chka, chkp, chkd, y, k)
            continue
        end
        #
        # child tables
        #
        if xnar[k] < xnar[k + one(I)]
            if kind[k] == two(I) + two(I)
                skip = arg[k]
            else
                skip = zero(I)
            end

            if !narrow!(C, xnar, narc, nari, naro, rlo, rhi, h, ckey, key, k, skip)
                continue
            end
        end

        return true
    end

    return false
end

#
# Plan the search, which assigns the elements m, ..., 1 in turn. For
# each element k, record
#
#   - the checks on k: each edge between k and an element assigned
#     earlier (or k itself), as its arc, the other element, and
#     whether the edge leaves k; and
#
#   - where its candidates come from: the image of an earlier element
#     pointing to k (3), the preimages of an earlier element k points
#     to (2), or all elements of its type (1), with the element and arc
#     that determine them.
#
# An edge w -> z is checked at the later of the two, min(w, z).
#
function plan!(
        C::Contraction{I},
        xchk::AbstractVector{I},
        chka::AbstractVector{I},
        chkp::AbstractVector{I},
        chkd::AbstractVector{Bool},
        bknd::AbstractVector{I},
        bpos::AbstractVector{I},
        barc::AbstractVector{I},
    ) where {I}
    X = C.inst
    m = X.dom.nu

    for k in oneto(m)
        xchk[k + one(I)] = zero(I)
        bknd[k] = one(I)
        bpos[k] = zero(I)
        barc[k] = zero(I)
    end

    for w in oneto(m)
        pstrt = X.dom.xadj[w]
        pstop = X.dom.xadj[w + one(I)] - one(I)

        for p in pstrt:pstop
            l = min(w, X.dom.adjncy[p])

            if l < m
                xchk[l + two(I)] += one(I)
            end
        end
    end

    xchk[one(I)] = q = one(I)

    for k in oneto(m)
        xchk[k + one(I)] = q += xchk[k + one(I)]
    end

    for w in m:-one(I):one(I)
        pstrt = X.dom.xadj[w]
        pstop = X.dom.xadj[w + one(I)] - one(I)

        for p in pstrt:pstop
            a = X.arc[p]
            z = X.dom.adjncy[p]

            if z >= w
                #
                # an edge out of w to an earlier element (or w itself)
                #
                r = xchk[w + one(I)]
                chka[r] = a
                chkp[r] = z
                chkd[r] = true
                xchk[w + one(I)] += one(I)

                if z > w && isone(bknd[w])
                    bknd[w] = two(I)
                    bpos[w] = z
                    barc[w] = a
                end
            else
                #
                # an edge into the later element z
                #
                r = xchk[z + one(I)]
                chka[r] = a
                chkp[r] = w
                chkd[r] = false
                xchk[z + one(I)] += one(I)

                if bknd[z] != two(I) + one(I)
                    bknd[z] = two(I) + one(I)
                    bpos[z] = w
                    barc[z] = a
                end
            end
        end
    end

    return
end

#
# Plan the narrowing. The table of each child j is sorted
# colexicographically by its separator, and the search assigns the
# separator from the end, so its ports are narrowed in the order
# w, ..., 1. The range of matching columns after narrowing ports
# i, ..., w is
#
#    rlo[o + i - 1] ... rhi[o + i - 1]
#
# where o = coff[j], so that the full range is at o + w and the final
# range at o. For each element k, record the ports on k.
#
function plannarrow!(
        C::Contraction{I},
        xnar::AbstractVector{I},
        narc::AbstractVector{I},
        nari::AbstractVector{I},
        naro::AbstractVector{I},
        rlo::AbstractVector{I},
        rhi::AbstractVector{I},
        coff::AbstractVector{I},
        ccol::AbstractVector{I},
    ) where {I}
    S = C.sep
    m = C.inst.dom.nu

    for k in oneto(m)
        xnar[k + one(I)] = zero(I)
    end

    for p in oneto(S.ne)
        k = S.adjncy[p]

        if k < m
            xnar[k + two(I)] += one(I)
        end
    end

    xnar[one(I)] = q = one(I)

    for k in oneto(m)
        xnar[k + one(I)] = q += xnar[k + one(I)]
    end

    o = one(I)

    for j in oneto(S.nu)
        pstrt = S.xadj[j]
        pstop = S.xadj[j + one(I)] - one(I)
        w = pstop - pstrt + one(I)
        rlo[o + w] = one(I)
        rhi[o + w] = ccol[j]
        coff[j] = o

        for p in pstrt:pstop
            k = S.adjncy[p]
            l = xnar[k + one(I)]
            narc[l] = j
            nari[l] = p - pstrt + one(I)
            naro[l] = o
            xnar[k + one(I)] += one(I)
        end

        o += w + one(I)
    end

    return
end

#
# Start the candidates for the element k, in order of preference:
#
#   3. the image of an assigned element pointing to it
#   4. the distinct values in the smallest child range
#   2. the preimages of an assigned element it points to
#   1. all elements of its type
#
@inline function candidates!(
        Y::Instance{I},
        pre::Span{I},
        base::AbstractVector{I},
        typ::Span{I},
        vtx::AbstractVector{I},
        h::AbstractVector{I},
        kind::AbstractVector{I},
        arg::AbstractVector{I},
        cur::AbstractVector{I},
        stop::AbstractVector{I},
        bknd::AbstractVector{I},
        bpos::AbstractVector{I},
        barc::AbstractVector{I},
        xnar::AbstractVector{I},
        nari::AbstractVector{I},
        naro::AbstractVector{I},
        rlo::AbstractVector{I},
        rhi::AbstractVector{I},
        k::I,
    ) where {I}
    kind[k] = bknd[k]

    if kind[k] == two(I) + one(I)
        arg[k] = image(Y, h[bpos[k]], barc[k])
        cur[k] = one(I)
        stop[k] = one(I)
        return
    end

    if kind[k] == two(I)
        a = barc[k]
        l = base[a] + h[bpos[k]] - typ.xadj[Y.cod.adjncy[a]]
        cur[k] = pre.xadj[l]
        stop[k] = pre.xadj[l + one(I)] - one(I)
    else
        u = vtx[k]
        cur[k] = typ.xadj[u]
        stop[k] = typ.xadj[u + one(I)] - one(I)
    end

    jstrt = xnar[k]
    jstop = xnar[k + one(I)] - one(I)

    for j in jstrt:jstop
        i = nari[j]
        o = naro[j]
        lo = rlo[o + i]
        hi = rhi[o + i]

        if kind[k] != two(I) + two(I) || hi - lo < stop[k] - cur[k]
            kind[k] = two(I) + two(I)
            arg[k] = j
            cur[k] = lo
            stop[k] = hi
        end
    end

    return
end

#
# The next candidate for the element k, or 0.
#
@inline function candidate!(
        C::Contraction{I},
        pre::Span{I},
        key::Vector{I},
        ckey::AbstractVector{I},
        narc::AbstractVector{I},
        nari::AbstractVector{I},
        naro::AbstractVector{I},
        rlo::AbstractVector{I},
        rhi::AbstractVector{I},
        kind::AbstractVector{I},
        arg::AbstractVector{I},
        cur::AbstractVector{I},
        stop::AbstractVector{I},
        k::I,
    ) where {I}
    c = cur[k]
    cur[k] += one(I)

    if isone(kind[k])
        return c
    elseif kind[k] == two(I)
        return pre.adjncy[c]
    elseif kind[k] == two(I) + one(I)
        return arg[k]
    end
    #
    # walk the child range: y is the value at column c, and the next
    # candidate is the first column after the run of columns with
    # value y
    #
    j = arg[k]
    l = narc[j]
    i = nari[j]
    o = naro[j]
    w = C.sep.xadj[l + one(I)] - C.sep.xadj[l]
    b = ckey[l] + i - one(I)
    y = key[b + (c - one(I)) * w]
    lo = c
    hi = c + one(I)
    step = one(I)

    while hi <= stop[k] && key[b + (hi - one(I)) * w] <= y
        lo = hi
        hi = c + step
        step += step
    end

    hi = min(hi, stop[k] + one(I))

    while lo + one(I) < hi
        mid = (lo + hi) >>> one(I)

        if key[b + (mid - one(I)) * w] <= y
            lo = mid
        else
            hi = mid
        end
    end

    rlo[o + i - one(I)] = c
    rhi[o + i - one(I)] = lo
    cur[k] = lo + one(I)
    return y
end

#
# Whether the element k, assigned to y, agrees with every edge between
# it and an element assigned earlier.
#
@inline function consistent(
        Y::Instance{I},
        h::AbstractVector{I},
        xchk::AbstractVector{I},
        chka::AbstractVector{I},
        chkp::AbstractVector{I},
        chkd::AbstractVector{Bool},
        y::I,
        k::I,
    ) where {I}
    qstrt = xchk[k]
    qstop = xchk[k + one(I)] - one(I)

    for q in qstrt:qstop
        z = h[chkp[q]]

        if chkd[q]
            image(Y, y, chka[q]) == z || return false
        else
            image(Y, z, chka[q]) == y || return false
        end
    end

    return true
end

#
# The weight of a complete assignment of the bag: the product of the
# weights of the residual elements and of the matched entries of the
# children's tables.
#
function weight(
        R::AbstractSemiring,
        C::Contraction{I},
        typ::Span{I},
        xwgt,
        wgt,
        h::AbstractVector{I},
        rlo::AbstractVector{I},
        coff::AbstractVector{I},
        cval::AbstractVector{I},
        val::Vector{T},
    ) where {I, T}
    w = sone(R, T)
    #
    # in the Boolean semiring, zero weights are never chosen, so every
    # complete assignment has weight 1
    #
    if R isa Boolean
        return w
    end

    if !isnothing(wgt)
        for x in oneto(C.nr)
            l = xwgt[x]

            if l < xwgt[x + one(I)]
                w = sprod(R, w, wgt[l + h[x] - typ.xadj[C.inst.vtx[x]]])
            end
        end
    end

    for j in oneto(C.sep.nu)
        w = sprod(R, w, val[cval[j] + rlo[coff[j]] - one(I)])
    end

    return w
end

#
# Write column n + 1 of the table: the separator assignment and its
# value. The arrays grow geometrically.
#
function record!(
        R::AbstractSemiring,
        okey::Vector{I},
        oval::Vector{T},
        ko::I,
        vo::I,
        h::AbstractVector{I},
        nr::I,
        s::I,
        n::I,
        value::T,
    ) where {I, T}
    l = ko + n * s

    if length(okey) < l + s
        resize!(okey, max(2 * length(okey), l + s))
    end

    for i in oneto(s)
        okey[l + i] = h[nr + i]
    end

    if !(R isa Boolean)
        l = vo + n

        if length(oval) < l + one(I)
            resize!(oval, max(2 * length(oval), l + one(I)))
        end

        oval[l + one(I)] = value
    end

    return
end

#
# Write the back-pointer of column n + 1 of the table: the residual
# assignment and the matched column of each child's table. The array
# grows geometrically.
#
function trace!(
        C::Contraction{I},
        fac,
        fo::I,
        h::AbstractVector{I},
        rlo::AbstractVector{I},
        coff::AbstractVector{I},
        n::I,
    ) where {I}
    if isnothing(fac)
        return
    end

    nr = C.nr
    nc = C.sep.nu
    l = fo + n * (nr + nc)

    if length(fac) < l + nr + nc
        resize!(fac, max(2 * length(fac), l + nr + nc))
    end

    for x in oneto(nr)
        fac[l + x] = h[x]
    end

    for j in oneto(nc)
        fac[l + nr + j] = rlo[coff[j]]
    end

    return
end

#
# Narrow the child ranges for the element k. Returns false if some
# range becomes empty.
#
function narrow!(
        C::Contraction{I},
        xnar::AbstractVector{I},
        narc::AbstractVector{I},
        nari::AbstractVector{I},
        naro::AbstractVector{I},
        rlo::AbstractVector{I},
        rhi::AbstractVector{I},
        h::AbstractVector{I},
        ckey::AbstractVector{I},
        key::Vector{I},
        k::I,
        skip::I,
    ) where {I}
    v = h[k]
    jstrt = xnar[k]
    jstop = xnar[k + one(I)] - one(I)

    for j in jstrt:jstop
        if j == skip
            continue
        end

        c = narc[j]
        i = nari[j]
        o = naro[j]
        s = C.sep.xadj[c + one(I)] - C.sep.xadj[c]
        b = ckey[c] + i - one(I)
        #
        # first column in range with port i at least v
        #
        lo = rlo[o + i]
        hi = rhi[o + i] + one(I)

        while lo < hi
            mid = (lo + hi) >>> one(I)

            if key[b + (mid - one(I)) * s] < v
                lo = mid + one(I)
            else
                hi = mid
            end
        end

        strt = lo
        #
        # first column in range with port i greater than v
        #
        hi = rhi[o + i] + one(I)

        while lo < hi
            mid = (lo + hi) >>> one(I)

            if key[b + (mid - one(I)) * s] <= v
                lo = mid + one(I)
            else
                hi = mid
            end
        end

        if strt == lo
            return false
        end

        rlo[o + i - one(I)] = strt
        rhi[o + i - one(I)] = lo - one(I)
    end

    return true
end
