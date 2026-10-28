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
    xwgt, wgt = weights(W, D, typ, T)
    #
    # the factor (back-pointers) is kept only for a witness
    #
    if witness
        fac = Vector{I}(undef, 1024)
    else
        fac = nothing
    end

    return solve(R, D, Y, yord, pre, base, typ, xwgt, wgt, fac)
end

#
# The workspace sizes: the largest bag, the largest number of children
# of a node, the largest number of separator incidences of the children
# of a node, and the largest number of edges in a bag.
#
function bagsizes(D::Decomposition{I}) where {I}
    X = D.inst
    nt = D.tree.nu
    mbag = zero(I)
    mchd = zero(I)
    msep = zero(I)
    medg = zero(I)

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
            n += D.sep.xadj[c + one(I)] - D.sep.xadj[c]
        end

        for x in rstrt:rstop
            d += X.dom.xadj[x + one(I)] - X.dom.xadj[x]
        end

        for p in sstrt:sstop
            x = D.sep.adjncy[p]
            d += X.dom.xadj[x + one(I)] - X.dom.xadj[x]
        end

        mbag = max(mbag, rstop - rstrt + sstop - sstrt + two(I))
        mchd = max(mchd, cstop - cstrt + one(I))
        msep = max(msep, n)
        medg = max(medg, d)
    end

    return mbag, mchd, msep, medg
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
#    wgt[xwgt[r]] ... wgt[xwgt[r + 1] - 1]
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
    xwgt = FVector{I}(undef, nu + one(I))
    wgt = FVector{T}(undef, nw)
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

    xwgt[one(I)] = one(I)

    for r in oneto(nu)
        c = X.vtx[r]

        if isnothing(W[c])
            xwgt[r + one(I)] = xwgt[r]
        else
            xwgt[r + one(I)] = xwgt[r] + typ.xadj[c + one(I)] - typ.xadj[c]
        end
    end

    for r in oneto(nu)
        c = X.vtx[r]
        l = xwgt[r]

        if l < xwgt[r + one(I)]
            M = W[c]
            j = rank[D.res.adjncy[r]]

            for i in oneto(xwgt[r + one(I)] - l)
                wgt[l + i - one(I)] = M[i, j]
            end
        end
    end

    return xwgt, wgt
end

#
# Allocate the workspaces and run the forward pass and, for a witness,
# the backward pass. The weights xwgt, wgt and the factor fac are
# either vectors or nothing, and this function is compiled separately
# for each.
#
# The forward pass visits the nodes in postorder. At each node, the
# decomposition is restricted to a contraction, the arguments (the
# children's tables, in place, and the weights of the bag, copied) are
# gathered, and the local problem is solved. The tables live on a
# stack: the children's tables are on top, the table of the node is
# written above them and then moved down over them. For a witness, each
# column of the table also gets a back-pointer in the factor: a residual
# assignment attaining its value, and the matched column of each
# child's table. The backward pass follows the back-pointers from the
# root.
#
function solve(
        R::AbstractSemiring,
        D::Decomposition{I},
        Y::Instance{I},
        yord::AbstractVector{I},
        pre::Span{I},
        base::AbstractVector{I},
        typ::Span{I},
        xwgt,
        wgt,
        fac,
    ) where {I}
    if isnothing(wgt)
        T = valtype(R)
    else
        T = eltype(wgt)
    end

    nu = D.inst.dom.nu
    nt = D.tree.nu
    mbag, mchd, msep, medg = bagsizes(D)
    mnar = msep + mchd
    #
    # the contraction
    #
    pos     = FVector{I}(undef, nu)
    xadj    = FVector{I}(undef, mbag + one(I))
    adjncy  = FVector{I}(undef, medg)
    vtx     = FVector{I}(undef, mbag)
    arc     = FVector{I}(undef, medg)
    sxadj   = FVector{I}(undef, mchd + one(I))
    sadjncy = FVector{I}(undef, msep)
    #
    # the arguments
    #
    ckey = FVector{I}(undef, mchd)
    cval = FVector{I}(undef, mchd)
    ccol = FVector{I}(undef, mchd)

    if isnothing(wgt)
        lxwgt = lwgt = nothing
    else
        lxwgt = FVector{I}(undef, mbag + one(I))
        lwgt  = Vector{T}(undef, 1024)
    end
    #
    # the local problem
    #
    kind = FVector{I}(undef, mbag)
    arg  = FVector{I}(undef, mbag)
    cur  = FVector{I}(undef, mbag)
    stop = FVector{I}(undef, mbag)
    bknd = FVector{I}(undef, mbag)
    bpos = FVector{I}(undef, mbag)
    barc = FVector{I}(undef, mbag)
    xchk = FVector{I}(undef, mbag + one(I))
    chka = FVector{I}(undef, medg)
    chkp = FVector{I}(undef, medg)
    chkd = FVector{Bool}(undef, medg)
    xnar = FVector{I}(undef, mbag + one(I))
    narc = FVector{I}(undef, msep)
    nari = FVector{I}(undef, msep)
    naro = FVector{I}(undef, msep)
    rlo  = FVector{I}(undef, mnar)
    rhi  = FVector{I}(undef, mnar)
    coff = FVector{I}(undef, mchd)
    hloc = FVector{I}(undef, mbag)
    #
    # the stack of tables: the table of node t has ncol[t] columns,
    # with keys tab[xtab[t]] ... and values tval[vtab[t]] ...
    #
    xtab = FVector{I}(undef, nt)
    vtab = FVector{I}(undef, nt)
    ncol = FVector{I}(undef, nt)
    tab  = Vector{I}(undef, 1024)
    tval = Vector{T}(undef, 1024)
    ntab = zero(I)
    nval = zero(I)
    #
    # the factor: the back-pointers of node t are fac[xfac[t]] ...
    #
    if isnothing(fac)
        xfac = nothing
    else
        xfac = FVector{I}(undef, nt)
    end

    nfac = zero(I)

    for t in oneto(nt)
        C = restrict!(D, pos, xadj, adjncy, vtx, arc, sxadj, sadjncy, t)
        lwgt = gather!(D, xtab, vtab, ncol, xwgt, wgt, ckey, cval, ccol, lxwgt, lwgt, t)
        n = contract!(R, C, Y, pre, base, typ, lxwgt, lwgt, ckey, cval, ccol, tab, tval, tab, tval, ntab, nval, fac, nfac, kind, arg, cur, stop, bknd, bpos, barc, xchk, chka, chkp, chkd, xnar, narc, nari, naro, rlo, rhi, coff, hloc)

        if iszero(n)
            return szero(R, T), nothing
        end
        #
        # pop the children's tables and move the table down over them
        #
        s = C.inst.dom.nu - C.nr
        kb = ntab
        vb = nval

        if ispositive(C.sep.nu)
            c = D.tree.adjncy[D.tree.xadj[t]]
            kb = xtab[c] - one(I)
            vb = vtab[c] - one(I)
        end

        for i in oneto(n * s)
            tab[kb + i] = tab[ntab + i]
        end

        if !(R isa Boolean)
            for i in oneto(n)
                tval[vb + i] = tval[nval + i]
            end
        end

        xtab[t] = kb + one(I)
        vtab[t] = vb + one(I)
        ncol[t] = n
        ntab = kb + n * s

        if !(R isa Boolean)
            nval = vb + n
        end

        if !isnothing(fac)
            xfac[t] = nfac + one(I)
            nfac += n * (C.inst.dom.nu - s + C.sep.nu)
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

    if isnothing(fac)
        return value, nothing
    end
    #
    # the backward pass: ncol[t] becomes the column of the table of t
    # chosen by its parent
    #
    h = FVector{I}(undef, nu)
    ncol[nt] = one(I)

    for t in nt:-one(I):one(I)
        rstrt = D.res.xadj[t]
        rstop = D.res.xadj[t + one(I)] - one(I)
        cstrt = D.tree.xadj[t]
        cstop = D.tree.xadj[t + one(I)] - one(I)
        nr = rstop - rstrt + one(I)
        nc = cstop - cstrt + one(I)
        l = xfac[t] + (ncol[t] - one(I)) * (nr + nc) - one(I)

        for x in oneto(nr)
            h[rstrt + x - one(I)] = fac[l + x]
        end

        for j in oneto(nc)
            ncol[D.tree.adjncy[cstrt + j - one(I)]] = fac[l + nr + j]
        end
    end

    out = FVector{I}(undef, nu)

    for r in oneto(nu)
        out[D.res.adjncy[r]] = yord[h[r]]
    end

    return value, out
end

#
# The arguments of the local problem at node t: the table of each
# child j, with ccol[j] columns, keys tab[ckey[j]] ... and values
# tval[cval[j]] ..., read in place; and the weights of each element x
# of the bag
#
#    lwgt[lxwgt[x]] ... lwgt[lxwgt[x + 1] - 1],
#
# copied. Returns the weights, which grow geometrically.
#
function gather!(
        D::Decomposition{I},
        xtab::AbstractVector{I},
        vtab::AbstractVector{I},
        ncol::AbstractVector{I},
        xwgt,
        wgt,
        ckey::AbstractVector{I},
        cval::AbstractVector{I},
        ccol::AbstractVector{I},
        lxwgt,
        lwgt,
        t::I,
    ) where {I}
    cstrt = D.tree.xadj[t]
    cstop = D.tree.xadj[t + one(I)] - one(I)
    #
    # the tables
    #
    for j in oneto(cstop - cstrt + one(I))
        c = D.tree.adjncy[cstrt + j - one(I)]
        ckey[j] = xtab[c]
        cval[j] = vtab[c]
        ccol[j] = ncol[c]
    end
    #
    # the weights
    #
    if !isnothing(wgt)
        rstrt = D.res.xadj[t]
        rstop = D.res.xadj[t + one(I)] - one(I)
        sstrt = D.sep.xadj[t]
        nr = rstop - rstrt + one(I)
        m = nr + D.sep.xadj[t + one(I)] - sstrt
        lxwgt[one(I)] = one(I)

        for x in oneto(m)
            if x <= nr
                g = rstrt + x - one(I)
            else
                g = D.sep.adjncy[sstrt + x - nr - one(I)]
            end

            lxwgt[x + one(I)] = lxwgt[x] + xwgt[g + one(I)] - xwgt[g]
        end

        if length(lwgt) < lxwgt[m + one(I)] - one(I)
            lwgt = Vector{eltype(wgt)}(undef, max(2 * length(lwgt), lxwgt[m + one(I)] - one(I)))
        end

        for x in oneto(m)
            if x <= nr
                g = rstrt + x - one(I)
            else
                g = D.sep.adjncy[sstrt + x - nr - one(I)]
            end

            l = lxwgt[x] - xwgt[g]

            for i in xwgt[g]:xwgt[g + one(I)] - one(I)
                lwgt[l + i] = wgt[i]
            end
        end
    end

    return lwgt
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
