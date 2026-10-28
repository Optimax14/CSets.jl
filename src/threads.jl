#
# Multithreaded search. The nodes of a level of the tree (nodes of the
# same height) do not depend on each other, so they are solved in
# parallel, each into a buffer of its task; the tables are then copied
# to the end of one array, in the order of the nodes. A node with many
# columns is split by bounds on its first separator elements into
# pieces that are solved apart and concatenated. The tables, values,
# and witnesses are the same as those of solve, whatever the number of
# tasks.
#
struct Threaded
    #
    # the number of tasks
    #
    ntasks::Int
    #
    # a level runs on the calling task unless the sum of the column
    # counts of the children of its nodes is at least this
    #
    minwork::Int
    #
    # a Boolean search on a tree with at least this many levels uses
    # tree contraction (treecontraction.jl)
    #
    deep::Int
end

function Threaded(ntasks::Integer=Threads.nthreads(); minwork::Integer=256, deep::Integer=256)
    return Threaded(ntasks, minwork, deep)
end

function score(P::Threaded, R::AbstractSemiring, W, X::Instance{I}, S::Span{I}, Y::Instance{I}; kw...) where {I}
    D = decomposition(X, S; kw...)
    return score(P, R, W, D, Y)
end

function score(P::Threaded, R::AbstractSemiring, W, D::Decomposition{I}, Y::Instance{I}) where {I}
    value, h = psolve(P, R, W, D, Y, false, valtype(R, W))
    return value
end

function color(P::Threaded, R::AbstractSemiring, W, X::Instance{I}, S::Span{I}, Y::Instance{I}; kw...) where {I}
    D = decomposition(X, S; kw...)
    return color(P, R, W, D, Y)
end

function color(P::Threaded, R::AbstractSemiring, W, D::Decomposition{I}, Y::Instance{I}) where {I}
    @assert isselective(R)
    return psolve(P, R, W, D, Y, true, valtype(R, W))
end

function color(P::Threaded, W, D::Decomposition{I}, Y::Instance{I}) where {I}
    value, h = color(P, Boolean(), W, D, Y)
    return h
end

function color(P::Threaded, D::Decomposition{I}, Y::Instance{I}) where {I}
    return color(P, nothing, D, Y)
end

#
# Many targets: each is solved serially, the targets in parallel.
#
function score(P::Threaded, R::AbstractSemiring, Ws::AbstractVector, D::Decomposition{I}, Ys::AbstractVector{<:Instance{I}}) where {I}
    return tmap(P.ntasks, eachindex(Ys)) do q
        score(R, Ws[q], D, Ys[q])
    end
end

function color(P::Threaded, R::AbstractSemiring, Ws::AbstractVector, D::Decomposition{I}, Ys::AbstractVector{<:Instance{I}}) where {I}
    return tmap(P.ntasks, eachindex(Ys)) do q
        color(R, Ws[q], D, Ys[q])
    end
end

#
# Apply f to each index, with at most n tasks, each taking the next
# index in turn. The results are in the order of the indices.
#
function tmap(f, n::Integer, idx)
    idx = collect(idx)
    out = Vector{Any}(undef, length(idx))
    next = Threads.Atomic{Int}(1)

    @sync for _ in 1:min(n, length(idx))
        Threads.@spawn begin
            while true
                i = Threads.atomic_add!(next, 1)
                i > length(idx) && break
                out[i] = f(idx[i])
            end
        end
    end

    return [x for x in out]
end

function treeheight(D::Decomposition{I}) where {I}
    nt = D.tree.nu
    hgt = zeros(I, nt)

    for t in oneto(nt), j in D.tree.xadj[t]:D.tree.xadj[t + one(I)] - one(I)
        hgt[t] = max(hgt[t], hgt[D.tree.adjncy[j]] + one(I))
    end

    return iszero(nt) ? zero(I) : hgt[nt] + one(I)
end

#
# The levels of the tree: the nodes of height l (forward) and of depth
# l (backward), as spans from levels to nodes, each level in increasing
# order.
#
function levels(D::Decomposition{I}) where {I}
    nt = D.tree.nu
    hgt = FVector{I}(undef, nt)
    dep = FVector{I}(undef, nt)

    for t in oneto(nt)
        hgt[t] = one(I)
    end
    #
    # children come before their parents
    #
    for t in oneto(nt)
        for j in D.tree.xadj[t]:D.tree.xadj[t + one(I)] - one(I)
            hgt[t] = max(hgt[t], hgt[D.tree.adjncy[j]] + one(I))
        end
    end

    for t in oneto(nt)
        dep[t] = one(I)
    end

    for t in nt:-one(I):one(I)
        for j in D.tree.xadj[t]:D.tree.xadj[t + one(I)] - one(I)
            dep[D.tree.adjncy[j]] = dep[t] + one(I)
        end
    end

    nh = isempty(hgt) ? zero(I) : maximum(hgt)
    nd = isempty(dep) ? zero(I) : maximum(dep)
    return invmap(nh, nt, hgt), invmap(nd, nt, dep)
end

#
# The workspace of a task: the contraction, the arguments, and buffers
# for the tables (keys, values, back-pointers) it writes.
#
struct TaskWorkspace{I, T, W, F}
    pos::FVector{I}
    xadj::FVector{I}
    adjncy::FVector{I}
    vtx::FVector{I}
    arc::FVector{I}
    sxadj::FVector{I}
    sadjncy::FVector{I}
    ckey::FVector{I}
    cval::FVector{I}
    ccol::FVector{I}
    lxwgt::W
    lwgt::Base.RefValue{Vector{T}}
    hloc::FVector{I}
    sel::FVector{I}
    lo::FVector{I}
    hi::FVector{I}
    okey::Vector{I}
    oval::Vector{T}
    ofac::F
    work::ContractionWorkspace{I}
end

function TaskWorkspace(D::Decomposition{I}, ::Type{T}, weighted::Bool, witness::Bool) where {I, T}
    nu = D.inst.dom.nu
    mbag, mchd, msep, medg = bagsizes(D)

    if weighted
        lxwgt = FVector{I}(undef, mbag + one(I))
    else
        lxwgt = nothing
    end

    if witness
        ofac = Vector{I}(undef, 1024)
    else
        ofac = nothing
    end

    return TaskWorkspace(
        FVector{I}(undef, nu),
        FVector{I}(undef, mbag + one(I)),
        FVector{I}(undef, medg),
        FVector{I}(undef, mbag),
        FVector{I}(undef, medg),
        FVector{I}(undef, mchd + one(I)),
        FVector{I}(undef, msep),
        FVector{I}(undef, mchd),
        FVector{I}(undef, mchd),
        FVector{I}(undef, mchd),
        lxwgt,
        Ref(Vector{T}(undef, 1024)),
        FVector{I}(undef, mbag),
        FVector{I}(undef, mchd),
        FVector{I}(undef, two(I)),
        FVector{I}(undef, two(I)),
        Vector{I}(undef, 1024),
        Vector{T}(undef, 1024),
        ofac,
        ContractionWorkspace{I}(mbag, mchd, msep, medg),
    )
end

#
# A piece of the work of a level: node t, with bounds (lo1, hi1) on
# its last element and (lo2, hi2) on the one before (d of them used).
# The piece writes n columns, starting at its task's buffers at ko, vo,
# fo, with nf back-pointer entries.
#
mutable struct Piece{I}
    t::I
    d::I
    lo1::I
    hi1::I
    lo2::I
    hi2::I
    task::Int
    ko::I
    vo::I
    fo::I
    n::I
    nf::I
end

#
# The pieces of a level: one per node, or, when the level has few
# nodes, several per node with a separator, split by the values of its
# last one or two elements.
#
function pieces(P::Threaded, D::Decomposition{I}, typ::Span{I}, nodes, ncol) where {I}
    out = Piece{I}[]
    X = D.inst
    want = 8 * P.ntasks
    est = estimates(D, typ, nodes, ncol)
    total = max(sum(est; init=0), 1)

    for (i, t) in enumerate(nodes)
        sstrt = D.sep.xadj[t]
        sstop = D.sep.xadj[t + one(I)] - one(I)
        s = sstop - sstrt + one(I)
        #
        # a node gets pieces in proportion to its share of the work
        #
        np = clamp(cld(want * est[i], total), 1, want)

        if np == 1 || s < one(I)
            push!(out, Piece{I}(t, zero(I), zero(I), zero(I), zero(I), zero(I), 0, zero(I), zero(I), zero(I), zero(I), zero(I)))
            continue
        end
        #
        # the last element of the bag is the last separator element;
        # split its values, and those of the one before when there are
        # too few
        #
        g1 = D.sep.adjncy[sstop]
        a1 = typ.xadj[X.vtx[g1]]
        b1 = typ.xadj[X.vtx[g1] + one(I)] - one(I)
        n1 = min(np, b1 - a1 + one(I))

        if n1 < np && s >= two(I)
            g2 = D.sep.adjncy[sstop - one(I)]
            a2 = typ.xadj[X.vtx[g2]]
            b2 = typ.xadj[X.vtx[g2] + one(I)] - one(I)
            n2 = min(cld(np, n1), b2 - a2 + one(I))
            d = two(I)
        else
            a2 = b2 = zero(I)
            n2 = 1
            d = one(I)
        end

        for i in 1:n1, j in 1:n2
            lo1 = a1 + I(fld((i - 1) * (b1 - a1 + 1), n1))
            hi1 = a1 + I(fld(i * (b1 - a1 + 1), n1)) - one(I)

            if d == two(I)
                lo2 = a2 + I(fld((j - 1) * (b2 - a2 + 1), n2))
                hi2 = a2 + I(fld(j * (b2 - a2 + 1), n2)) - one(I)
            else
                lo2 = hi2 = zero(I)
            end

            lo1 <= hi1 && (d == one(I) || lo2 <= hi2) || continue
            push!(out, Piece{I}(t, d, lo1, hi1, lo2, hi2, 0, zero(I), zero(I), zero(I), zero(I), zero(I)))
        end
    end

    return out
end

#
# The estimated work of each node: the column counts of its children,
# or for a leaf the number of assignments of its separator.
#
function estimates(D::Decomposition{I}, typ::Span{I}, nodes, ncol) where {I}
    X = D.inst
    est = zeros(Int, length(nodes))

    for (i, t) in enumerate(nodes)
        cstrt = D.tree.xadj[t]
        cstop = D.tree.xadj[t + one(I)] - one(I)

        if cstrt <= cstop
            for j in cstrt:cstop
                est[i] += ncol[D.tree.adjncy[j]]
            end
        else
            w = 1

            for p in D.sep.xadj[t]:D.sep.xadj[t + one(I)] - one(I)
                u = X.vtx[D.sep.adjncy[p]]
                w = min(w * (typ.xadj[u + one(I)] - typ.xadj[u]), 1 << 40)
            end

            est[i] = w
        end
    end

    return est
end

#
# Solve a piece with the workspace of its task, writing to its buffers.
#
function solve!(
        R::AbstractSemiring,
        D::Decomposition{I},
        Y::Instance{I},
        pre::Span{I},
        base::AbstractVector{I},
        typ::Span{I},
        xwgt,
        wgt,
        xtab::AbstractVector{I},
        vtab::AbstractVector{I},
        ncol::AbstractVector{I},
        tab::Vector{I},
        tval::Vector{T},
        ws::TaskWorkspace{I, T},
        pc::Piece{I},
        ko::I,
        vo::I,
        fo::I,
    ) where {I, T}
    t = pc.t
    C = restrict!(D, ws.pos, ws.xadj, ws.adjncy, ws.vtx, ws.arc, ws.sxadj, ws.sadjncy, t)
    if isnothing(wgt)
        lwgt = nothing
        gather!(D, xtab, vtab, ncol, xwgt, wgt, ws.ckey, ws.cval, ws.ccol, ws.lxwgt, lwgt, t)
    else
        lwgt = ws.lwgt[] = gather!(D, xtab, vtab, ncol, xwgt, wgt, ws.ckey, ws.cval, ws.ccol, ws.lxwgt, ws.lwgt[], t)
    end

    if iszero(pc.d)
        bounds = nothing
    else
        ws.lo[one(I)] = pc.lo1; ws.hi[one(I)] = pc.hi1
        ws.lo[two(I)] = pc.lo2; ws.hi[two(I)] = pc.hi2
        bounds = Bounds{I}(pc.d, ws.lo, ws.hi)
    end

    n, nf = contract!(R, C, Y, pre, base, typ, ws.lxwgt, lwgt, ws.ckey, ws.cval, ws.ccol, tab, tval, ws.okey, ws.oval, ko, vo, ws.ofac, fo, ws.work, bounds)
    pc.ko = ko; pc.vo = vo; pc.fo = fo; pc.n = n; pc.nf = nf
    return C
end

function psolve(P::Threaded, R::AbstractSemiring, W, D::Decomposition{I}, Y::Instance{I}, witness::Bool, ::Type{T}) where {I, T}
    if R isa Boolean && treeheight(D) >= P.deep
        value, h = contractsolve(P, R, W, D, Y, witness)
        return convert(T, value), h
    end

    Y, yord = normalize(Y)
    typ = invvtx(Y)
    base, pre = preimages(Y, typ)
    xwgt, wgt = weights(W, D, typ, T)

    if isnothing(wgt)
        V = valtype(R)
    else
        V = eltype(wgt)
    end

    nu = D.inst.dom.nu
    nt = D.tree.nu
    up, down = levels(D)
    nw = P.ntasks
    wss = [TaskWorkspace(D, V, !isnothing(wgt), witness) for _ in 1:nw]
    boolean = R isa Boolean
    #
    # the table of node t: keys tkey[t], values tval[t], back-pointers
    # tfac[t]. Before a level is solved, the tables of the children of
    # its nodes are moved into one array (the arena), where xtab and
    # vtab point, and freed.
    #
    tkey = Vector{Vector{I}}(undef, nt)
    tval = Vector{Vector{V}}(undef, nt)
    tfac = Vector{Vector{I}}(undef, witness ? nt : 0)
    xtab = FVector{I}(undef, nt)
    vtab = FVector{I}(undef, nt)
    ncol = FVector{I}(undef, nt)
    akey = Vector{I}(undef, 1024)
    aval = Vector{V}(undef, 1024)

    for l in oneto(up.nu)
        nodes = view(up.adjncy, up.xadj[l]:up.xadj[l + one(I)] - one(I))
        #
        # the arena
        #
        nk = zero(I)
        nv = zero(I)

        for t in nodes, j in D.tree.xadj[t]:D.tree.xadj[t + one(I)] - one(I)
            c = D.tree.adjncy[j]
            nk += length(tkey[c])
            nv += length(tval[c])
        end

        length(akey) < nk && resize!(akey, max(nk, 2 * length(akey)))
        length(aval) < nv && resize!(aval, max(nv, 2 * length(aval)))
        nk = nv = zero(I)

        for t in nodes, j in D.tree.xadj[t]:D.tree.xadj[t + one(I)] - one(I)
            c = D.tree.adjncy[j]
            copyto!(akey, nk + one(I), tkey[c], 1, length(tkey[c]))
            copyto!(aval, nv + one(I), tval[c], 1, length(tval[c]))
            xtab[c] = nk + one(I)
            vtab[c] = nv + one(I)
            nk += length(tkey[c])
            nv += length(tval[c])
            tkey[c] = I[]
            tval[c] = V[]
        end
        #
        # the work of the level: the column counts of the children (for
        # a leaf, the number of assignments of its separator)
        #
        est = estimates(D, typ, nodes, ncol)
        work = sum(est; init=0)

        if nw == 1 || work < P.minwork
            pcs = [Piece{I}(t, zero(I), zero(I), zero(I), zero(I), zero(I), 1, zero(I), zero(I), zero(I), zero(I), zero(I)) for t in nodes]
            solvechunk!(R, D, Y, pre, base, typ, xwgt, wgt, xtab, vtab, ncol, akey, aval, wss[1], pcs, 1, Threads.Atomic{Int}(1))
        else
            pcs = pieces(P, D, typ, nodes, ncol)
            next = Threads.Atomic{Int}(1)

            @sync for k in 1:min(nw, length(pcs))
                Threads.@spawn solvechunk!(R, D, Y, pre, base, typ, xwgt, wgt, xtab, vtab, ncol, akey, aval, wss[k], pcs, k, next)
            end
        end
        #
        # the table of each node: its pieces, in order
        #
        i = 1

        while i <= length(pcs)
            t = pcs[i].t
            s = D.sep.xadj[t + one(I)] - D.sep.xadj[t]
            j = i
            n = nf = zero(I)

            while j <= length(pcs) && pcs[j].t == t
                n += pcs[j].n
                nf += pcs[j].nf
                j += 1
            end

            iszero(n) && return szero(R, V), nothing
            ncol[t] = n
            key = tkey[t] = Vector{I}(undef, n * s)
            val = tval[t] = Vector{V}(undef, boolean ? 0 : n)
            witness && (tfac[t] = Vector{I}(undef, nf))
            ko = vo = fo = 0

            for q in i:j - 1
                pc = pcs[q]
                ws = wss[pc.task]
                copyto!(key, ko + 1, ws.okey, pc.ko + one(I), pc.n * s)
                ko += pc.n * s

                if !boolean
                    copyto!(val, vo + 1, ws.oval, pc.vo + one(I), pc.n)
                    vo += pc.n
                end

                if witness
                    copyto!(tfac[t], fo + 1, ws.ofac, pc.fo + one(I), pc.nf)
                    fo += pc.nf
                end
            end

            i = j
        end
    end

    if boolean
        value = sone(R, V)
    else
        value = tval[nt][1]
    end

    if !witness
        return value, nothing
    end
    #
    # backward, level by level from the root: ncol[t] becomes the choice
    # for node t made by its parent
    #
    h = FVector{I}(undef, nu)
    ncol[nt] = one(I)

    for l in oneto(down.nu)
        nodes = view(down.adjncy, down.xadj[l]:down.xadj[l + one(I)] - one(I))
        nk = length(nodes) < 64 ? 1 : min(nw, length(nodes))

        @sync for k in 1:nk
            Threads.@spawn liftchunk!(D, h, ncol, tfac, wss[k], nodes, fld((k - 1) * length(nodes), nk) + 1:fld(k * length(nodes), nk))
        end
    end

    out = FVector{I}(undef, nu)

    for r in oneto(nu)
        out[D.res.adjncy[r]] = yord[h[r]]
    end

    return value, out
end

#
# Solve pieces, taking the next one in turn, with the workspace of task
# k, writing one after the other to its buffers.
#
function solvechunk!(R, D::Decomposition{I}, Y, pre, base, typ, xwgt, wgt, xtab, vtab, ncol, tab, tval, ws::TaskWorkspace{I}, pcs, k::Int, next) where {I}
    ko = vo = fo = zero(I)

    while true
        i = Threads.atomic_add!(next, 1)
        i > length(pcs) && break
        pc = pcs[i]
        pc.task = k
        solve!(R, D, Y, pre, base, typ, xwgt, wgt, xtab, vtab, ncol, tab, tval, ws, pc, ko, vo, fo)
        ko += pc.n * (D.sep.xadj[pc.t + one(I)] - D.sep.xadj[pc.t])

        if !(R isa Boolean)
            vo += pc.n
        end

        fo += pc.nf
    end

    return
end

function liftchunk!(D::Decomposition{I}, h, ncol, tfac, ws::TaskWorkspace{I}, nodes, range) where {I}
    for i in range
        lift(D, h, ncol, tfac, ws, nodes[i])
    end

    return
end

function lift(D::Decomposition{I}, h, ncol, tfac, ws::TaskWorkspace{I}, t::I) where {I}
    C = restrict!(D, ws.pos, ws.xadj, ws.adjncy, ws.vtx, ws.arc, ws.sxadj, ws.sadjncy, t)
    nr = C.nr
    rstrt = D.res.xadj[t]
    sstrt = D.sep.xadj[t]
    cstrt = D.tree.xadj[t]

    for x in nr + one(I):C.inst.dom.nu
        ws.hloc[x] = h[D.sep.adjncy[sstrt + x - nr - one(I)]]
    end

    lift!(C, tfac[t], zero(I), ws.hloc, ncol[t], ws.sel)

    for x in oneto(nr)
        h[rstrt + x - one(I)] = ws.hloc[x]
    end

    for j in oneto(C.sep.nu)
        ncol[D.tree.adjncy[cstrt + j - one(I)]] = ws.sel[j]
    end

    return
end
