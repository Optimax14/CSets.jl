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
function valtype(R::AbstractSemiring, W)
    if !isnothing(W)
        for M in W
            isnothing(M) || return eltype(M)
        end
    end

    return valtype(R)
end

function search(R::AbstractSemiring, W, D::Decomposition{I}, Y::Instance{I}, witness::Bool, ::Type{T}) where {I, T}
    Y, yord = normalize(Y)
    typ = invvtx(Y)
    base, pre = preimages(Y, typ)
    woff, wval = weights(W, D, typ, T)
    return solve(R, D, Y, yord, pre, base, typ, woff, wval, witness)
end

#
# The workspace sizes: the largest bag plus one, the largest number of
# narrowings at a node, and the largest number of edge checks at a node.
#
function bagsizes(D::Decomposition{I}) where {I}
    X = D.inst
    nt = D.tree.nu
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

    return mbag, mnar, mchk
end

#
# The preimages of the elements of Y under each arc. The preimages of y
# under arc a are the column
#
#    base[a] + y - typ.xadj[tgt(a)]
#
# of the returned span, listed in increasing order.
#
function preimages(Y::Instance{I}, typ::Span{I}) where {I}
    ny = Y.dom.nu
    nf = Y.dom.ne
    na = Y.cod.ne

    base = FVector{I}(undef, na + one(I))
    base[one(I)] = one(I)

    for a in oneto(na)
        d = Y.cod.adjncy[a]
        base[a + one(I)] = base[a] + typ.xadj[d + one(I)] - typ.xadj[d]
    end

    np = base[na + one(I)] - one(I)
    xadj   = FVector{I}(undef, np + one(I))
    adjncy = FVector{I}(undef, nf)

    for l in oneto(np)
        xadj[l + one(I)] = zero(I)
    end

    for y in oneto(ny)
        pstrt = Y.dom.xadj[y]
        pstop = Y.dom.xadj[y + one(I)] - one(I)

        for p in pstrt:pstop
            a = Y.arc[p]
            l = base[a] + Y.dom.adjncy[p] - typ.xadj[Y.cod.adjncy[a]]

            if l < np
                xadj[l + two(I)] += one(I)
            end
        end
    end

    xadj[one(I)] = q = one(I)

    for l in oneto(np)
        xadj[l + one(I)] = q += xadj[l + one(I)]
    end

    for y in oneto(ny)
        pstrt = Y.dom.xadj[y]
        pstop = Y.dom.xadj[y + one(I)] - one(I)

        for p in pstrt:pstop
            a = Y.arc[p]
            l = base[a] + Y.dom.adjncy[p] - typ.xadj[Y.cod.adjncy[a]]
            adjncy[xadj[l + one(I)]] = y
            xadj[l + one(I)] += one(I)
        end
    end

    return base, Span{I}(np, ny, nf, xadj, adjncy)
end

#
# Relabel the weights: element r of X gets the column
#
#    wval[woff[r]] ... wval[woff[r + 1] - 1]
#
# indexed by the elements of Y of its sort, or an empty column if every
# image has weight 1. Returns nothing, nothing if there are no weights.
#
function weights(W, D::Decomposition{I}, typ::Span{I}, ::Type{T}) where {I, T}
    X = D.inst
    nu = X.dom.nu
    ns = X.cod.nu
    nw = zero(I)

    if !isnothing(W)
        for r in oneto(nu)
            c = X.vtx[r]

            if !isnothing(W[c])
                nw += typ.xadj[c + one(I)] - typ.xadj[c]
            end
        end
    end

    if iszero(nw)
        return nothing, nothing
    end

    rank = FVector{I}(undef, nu)
    nsrt = FVector{I}(undef, ns)
    woff = FVector{I}(undef, nu + one(I))
    wval = FVector{T}(undef, nw)
    #
    # rank[v] is first the sort of element v in the caller's numbering,
    # then its rank among the elements of that sort
    #
    for r in oneto(nu)
        rank[D.res.adjncy[r]] = X.vtx[r]
    end

    for c in oneto(ns)
        nsrt[c] = zero(I)
    end

    for v in oneto(nu)
        c = rank[v]
        nsrt[c] += one(I)
        rank[v] = nsrt[c]
    end

    woff[one(I)] = one(I)

    for r in oneto(nu)
        c = X.vtx[r]

        if isnothing(W[c])
            woff[r + one(I)] = woff[r]
        else
            woff[r + one(I)] = woff[r] + typ.xadj[c + one(I)] - typ.xadj[c]
        end
    end

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

    return woff, wval
end

#
# Allocate the workspaces and run the forward pass (tables) and, for a
# witness, the backward pass. The weights woff, wval are either vectors
# or nothing, and this function is compiled separately for each.
#
function solve(
        R::AbstractSemiring,
        D::Decomposition{I},
        Y::Instance{I},
        yord::AbstractVector{I},
        pre::Span{I},
        base::AbstractVector{I},
        typ::Span{I},
        woff,
        wval,
        witness::Bool,
    ) where {I}
    if isnothing(wval)
        T = valtype(R)
    else
        T = eltype(wval)
    end

    nu = D.inst.dom.nu
    nt = D.tree.nu
    mbag, mnar, mchk = bagsizes(D)

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
    vtab = FVector{I}(undef, nt)
    tval = Vector{T}(undef, 1024)
    nval = FScalar{I}(undef)
    coff = FVector{I}(undef, max(D.tree.ne, one(I)))
    csel = FVector{I}(undef, nt)

    ntab[] = zero(I)
    nval[] = zero(I)

    for x in oneto(nu)
        bag[x] = zero(I)
    end

    for t in oneto(nt)
        if !contract!(R, D, Y, pre, base, typ, woff, wval, ord, kind, arg, cur, stop, bknd, bpos, barc, xchk, chka, chkp, chkd, xnar, narc, nari, naro, rlo, rhi, pos, bag, h, xtab, ncol, tab, ntab, vtab, tval, nval, coff, t)
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
        extend!(R, D, Y, pre, base, typ, woff, wval, ord, kind, arg, cur, stop, bknd, bpos, barc, xchk, chka, chkp, chkd, xnar, narc, nari, naro, rlo, rhi, pos, bag, h, xtab, ncol, tab, vtab, tval, coff, csel, t)
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

#
# The local problem at node t, forward: append to the table of t every
# assignment of the separator that extends to the bag, with its value:
# the sum over its extensions of the product of the residual weights
# and the matched entries of the children's tables. Returns whether
# the table is nonempty.
#
function contract!(
        R::AbstractSemiring,
        D::Decomposition{I},
        Y::Instance{I},
        pre::Span{I},
        base::AbstractVector{I},
        typ::Span{I},
        woff,
        wval,
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
        t::I,
    ) where {I, T}
    s = D.sep.xadj[t + one(I)] - D.sep.xadj[t]
    m = s + D.res.xadj[t + one(I)] - D.res.xadj[t]
    prepare!(D, ord, pos, bag, xchk, chka, chkp, chkd, bknd, bpos, barc, xnar, narc, nari, naro, rlo, rhi, coff, ncol, m, t)

    xtab[t] = ntab[] + one(I)
    vtab[t] = nval[] + one(I)
    ncol[t] = zero(I)

    if iszero(m)
        w = weight(R, D, typ, woff, wval, ord, h, rlo, coff, vtab, tval, s, m, t)

        if w != szero(R, T)
            record!(R, tab, ntab, tval, nval, ncol, h, ord, s, t, w)
        end

        return ispositive(ncol[t])
    end
    #
    # acc sums the weights of the extensions of the current separator
    # assignment
    #
    acc = szero(R, T)
    k = one(I)
    fresh = true

    while ispositive(k)
        if fresh
            fresh = false
            candidates!(Y, pre, base, typ, D.inst.vtx, ord, h, kind, arg, cur, stop, bknd, bpos, barc, xnar, nari, naro, rlo, rhi, k)
        end

        if advance!(R, D, Y, pre, typ, woff, wval, ord, kind, arg, cur, stop, xchk, chka, chkp, chkd, xnar, narc, nari, naro, rlo, rhi, h, xtab, tab, k)
            if k < m
                k += one(I)
                fresh = true
            else
                acc = splus(R, acc, weight(R, D, typ, woff, wval, ord, h, rlo, coff, vtab, tval, s, m, t))
                #
                # the separator assignment is finished when there is no
                # residual, or when the sum can no longer change
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
            end
        else
            k -= one(I)
            #
            # leaving the residual: the separator assignment is finished
            #
            if k == s && s < m
                if acc != szero(R, T)
                    record!(R, tab, ntab, tval, nval, ncol, h, ord, s, t, acc)
                end

                acc = szero(R, T)
            end
        end
    end

    return ispositive(ncol[t])
end

#
# The local problem at node t, backward: given the separator assignment
# chosen by the parent (entry csel[t] of the table of t), assign the
# residual so that the weight equals the value of that entry, and
# choose the matched entries of the children's tables.
#
function extend!(
        R::AbstractSemiring,
        D::Decomposition{I},
        Y::Instance{I},
        pre::Span{I},
        base::AbstractVector{I},
        typ::Span{I},
        woff,
        wval,
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
        vtab::AbstractVector{I},
        tval::Vector{T},
        coff::AbstractVector{I},
        csel::AbstractVector{I},
        t::I,
    ) where {I, T}
    s = D.sep.xadj[t + one(I)] - D.sep.xadj[t]
    m = s + D.res.xadj[t + one(I)] - D.res.xadj[t]
    prepare!(D, ord, pos, bag, xchk, chka, chkp, chkd, bknd, bpos, barc, xnar, narc, nari, naro, rlo, rhi, coff, ncol, m, t)
    #
    # the separator is already assigned
    #
    for k in oneto(s)
        narrow!(D, ord, xnar, narc, nari, naro, rlo, rhi, h, xtab, tab, k, zero(I))
    end

    if m == s
        select!(D, rlo, coff, csel, t)
        return
    end

    if R isa Boolean
        target = sone(R, T)
    else
        target = tval[vtab[t] + csel[t] - one(I)]
    end

    k = s + one(I)
    fresh = true

    while s < k
        if fresh
            fresh = false
            candidates!(Y, pre, base, typ, D.inst.vtx, ord, h, kind, arg, cur, stop, bknd, bpos, barc, xnar, nari, naro, rlo, rhi, k)
        end

        if advance!(R, D, Y, pre, typ, woff, wval, ord, kind, arg, cur, stop, xchk, chka, chkp, chkd, xnar, narc, nari, naro, rlo, rhi, h, xtab, tab, k)
            if k < m
                k += one(I)
                fresh = true
            elseif weight(R, D, typ, woff, wval, ord, h, rlo, coff, vtab, tval, s, m, t) == target
                select!(D, rlo, coff, csel, t)
                return
            end
        else
            k -= one(I)
        end
    end

    return
end

#
# Order and plan the bag at node t.
#
function prepare!(
        D::Decomposition{I},
        ord::AbstractVector{I},
        pos::AbstractVector{I},
        bag::AbstractVector{I},
        xchk::AbstractVector{I},
        chka::AbstractVector{I},
        chkp::AbstractVector{I},
        chkd::AbstractVector{Bool},
        bknd::AbstractVector{I},
        bpos::AbstractVector{I},
        barc::AbstractVector{I},
        xnar::AbstractVector{I},
        narc::AbstractVector{I},
        nari::AbstractVector{I},
        naro::AbstractVector{I},
        rlo::AbstractVector{I},
        rhi::AbstractVector{I},
        coff::AbstractVector{I},
        ncol::AbstractVector{I},
        m::I,
        t::I,
    ) where {I}
    order!(D, ord, pos, bag, t)
    plan!(D, ord, pos, xchk, chka, chkp, chkd, bknd, bpos, barc, m)
    plannarrow!(D, pos, xnar, narc, nari, naro, rlo, rhi, coff, ncol, m, t)
    return
end

#
# Assign the next valid candidate to the element at position k: one
# with nonzero weight, consistent with every edge to an earlier
# element, and matching the children's tables. Returns false when the
# candidates are exhausted.
#
@inline function advance!(
        R::AbstractSemiring,
        D::Decomposition{I},
        Y::Instance{I},
        pre::Span{I},
        typ::Span{I},
        woff,
        wval,
        ord::AbstractVector{I},
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
        xtab::AbstractVector{I},
        tab::Vector{I},
        k::I,
    ) where {I}
    x = ord[k]

    while cur[k] <= stop[k]
        y = candidate!(D, pre, tab, xtab, narc, nari, naro, rlo, rhi, kind, arg, cur, stop, k)

        if !ispositive(y)
            continue
        end

        if !isnothing(wval)
            l = woff[x]

            if l < woff[x + one(I)] && wval[l + y - typ.xadj[D.inst.vtx[x]]] == szero(R, eltype(wval))
                continue
            end
        end

        h[x] = y

        if !consistent(Y, ord, h, xchk, chka, chkp, chkd, y, k)
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

            if !narrow!(D, ord, xnar, narc, nari, naro, rlo, rhi, h, xtab, tab, k, skip)
                continue
            end
        end

        return true
    end

    return false
end

#
# Order the bag at node t: separator elements first, each part
# descending. Stamp each element with its node and position.
#
function order!(D::Decomposition{I}, ord::AbstractVector{I}, pos::AbstractVector{I}, bag::AbstractVector{I}, t::I) where {I}
    rstrt = D.res.xadj[t]
    rstop = D.res.xadj[t + one(I)] - one(I)
    sstrt = D.sep.xadj[t]
    sstop = D.sep.xadj[t + one(I)] - one(I)
    k = zero(I)

    for p in reverse(sstrt:sstop)
        k += one(I)
        ord[k] = D.sep.adjncy[p]
    end

    for x in reverse(rstrt:rstop)
        k += one(I)
        ord[k] = x
    end

    for j in oneto(k)
        x = ord[j]
        bag[x] = t
        pos[x] = j
    end

    return
end

#
# Plan the search. For each position k, record
#
#   - the checks on the element x = ord[k]: each edge between x and an
#     element at an earlier position (or x itself), as its arc, the
#     other position, and whether the edge leaves x; and
#
#   - where its candidates come from: the image of an earlier element
#     pointing to x (3), the preimages of an earlier element x points to
#     (2), or all elements of its type (1), with the position and arc
#     that determine them.
#
# Every edge with both ends in the bag is an edge out of some bag
# element, since bags are closed downward, so the checks are found by
# scanning out-edges only. An edge w -> z is checked at the later of
# the two positions.
#
function plan!(
        D::Decomposition{I},
        ord::AbstractVector{I},
        pos::AbstractVector{I},
        xchk::AbstractVector{I},
        chka::AbstractVector{I},
        chkp::AbstractVector{I},
        chkd::AbstractVector{Bool},
        bknd::AbstractVector{I},
        bpos::AbstractVector{I},
        barc::AbstractVector{I},
        m::I,
    ) where {I}
    X = D.inst

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

    return
end

#
# Plan the narrowing. Each child's table is sorted by its separator in
# descending order, which is the order in which this search assigns
# those elements. For each child c and each component i of its key,
# record the position at which component i is assigned. At that
# position, the range of matching columns
#
#    rlo[o + i] ... rhi[o + i]
#
# is narrowed from the range for component i - 1. The offset o of each
# child is coff[j].
#
function plannarrow!(
        D::Decomposition{I},
        pos::AbstractVector{I},
        xnar::AbstractVector{I},
        narc::AbstractVector{I},
        nari::AbstractVector{I},
        naro::AbstractVector{I},
        rlo::AbstractVector{I},
        rhi::AbstractVector{I},
        coff::AbstractVector{I},
        ncol::AbstractVector{I},
        m::I,
        t::I,
    ) where {I}
    cstrt = D.tree.xadj[t]
    cstop = D.tree.xadj[t + one(I)] - one(I)

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

    return
end

#
# Start the candidates for the element at position k, in order of
# preference:
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
        ord::AbstractVector{I},
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
        arg[k] = image(Y, h[ord[bpos[k]]], barc[k])
        cur[k] = one(I)
        stop[k] = one(I)
        return
    end

    if kind[k] == two(I)
        a = barc[k]
        l = base[a] + h[ord[bpos[k]]] - typ.xadj[Y.cod.adjncy[a]]
        cur[k] = pre.xadj[l]
        stop[k] = pre.xadj[l + one(I)] - one(I)
    else
        u = vtx[ord[k]]
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

    return
end

#
# The next candidate for the element at position k, or 0.
#
@inline function candidate!(
        D::Decomposition{I},
        pre::Span{I},
        tab::Vector{I},
        xtab::AbstractVector{I},
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
    return y
end

#
# Whether the element at position k, assigned to y, agrees with every
# edge between it and an earlier element.
#
@inline function consistent(
        Y::Instance{I},
        ord::AbstractVector{I},
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
        z = h[ord[chkp[q]]]

        if chkd[q]
            image(Y, y, chka[q]) == z || return false
        else
            image(Y, z, chka[q]) == y || return false
        end
    end

    return true
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
        woff,
        wval,
        ord::AbstractVector{I},
        h::AbstractVector{I},
        rlo::AbstractVector{I},
        coff::AbstractVector{I},
        vtab::AbstractVector{I},
        tval::Vector{T},
        s::I,
        m::I,
        t::I,
    ) where {I, T}
    w = sone(R, T)
    #
    # in the Boolean semiring, zero weights are never chosen, so every
    # complete assignment has weight 1
    #
    if R isa Boolean
        return w
    end

    if !isnothing(wval)
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
