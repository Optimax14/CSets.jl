#
# Tree contraction for deep trees, in the Boolean semiring.
#
# A path of bags is a composition of relations: with all children but
# one (c) known, the table of node t is the image of the table of c
# under a Boolean matrix
#
#    M_t : Loc_c × Loc_t,
#
# where Loc_t is the set of separator assignments of t that extend to
# its bag (ignoring its children). So a chain of bags collapses by
# Boolean matrix products in O(log n) rounds instead of n levels. The
# rounds are rake (solve the nodes whose children are known) and
# compress (random mate: a chain node absorbs its chain child), each
# over all nodes in parallel. Messages and matrix rows are bitsets.
#
# The tables (Loc_t filtered by the messages) and the witness are those
# of solve.
#
struct LocalTables{I}
    #
    # Full_t, the assignments of bag t consistent with its edges and
    # weights: column x has the keys
    #
    #    full[kt[t]][xf[t] + (x - 1) * m + 1] ...
    #
    # (the buffer of the task that solved t)
    #
    nf::Vector{I}
    kt::Vector{Int}
    xf::Vector{Int}
    full::Vector{Vector{I}}
    #
    # Loc_t: the runs of columns with the same separator; proj[xp[t] + x]
    # is the run of column x, first[xp[t] + b] the first column of run b
    #
    nl::Vector{I}
    xp::Vector{Int}
    proj::Vector{I}
    first::Vector{I}
    #
    # the run of child j matched by column x: cproj[xc[t] + (x - 1) * nc + j]
    # (0 if none)
    #
    xc::Vector{Int}
    cproj::Vector{I}
end

#
# The local tables of every bag, in parallel.
#
function localtables(P::Threaded, R::Boolean, D::Decomposition{I}, Y::Instance{I}, pre, base, typ, xwgt, wgt) where {I}
    nt = D.tree.nu
    nw = P.ntasks
    wss = [TaskWorkspace(D, Bool, !isnothing(wgt), false) for _ in 1:nw]
    #
    # the columns of node t are in the buffer of task ktask[t] at kofs[t]
    #
    ktask = Vector{Int}(undef, nt)
    kofs = Vector{Int}(undef, nt)
    nf = Vector{I}(undef, nt)
    xtab = FVector{I}(undef, nt)
    vtab = FVector{I}(undef, nt)
    ncol = FVector{I}(undef, nt)
    tab = Vector{I}(undef, 1)
    tval = Vector{Bool}(undef, 1)
    next = Threads.Atomic{Int}(1)

    @sync for k in 1:nw
        Threads.@spawn localchunk!(R, D, Y, pre, base, typ, xwgt, wgt, xtab, vtab, ncol, tab, tval, wss[k], k, ktask, kofs, nf, next)
    end

    full = [ws.okey for ws in wss]
    #
    # the runs, and the matched runs of the children
    #
    nl = Vector{I}(undef, nt)
    xp = Vector{Int}(undef, nt + 1)
    xc = Vector{Int}(undef, nt + 1)
    xp[1] = xc[1] = 0

    for t in oneto(nt)
        xp[t + 1] = xp[t] + nf[t]
        xc[t + 1] = xc[t] + nf[t] * (D.tree.xadj[t + one(I)] - D.tree.xadj[t])
    end

    proj = Vector{I}(undef, xp[nt + 1])
    first = Vector{I}(undef, xp[nt + 1])
    cproj = Vector{I}(undef, xc[nt + 1])
    L = LocalTables{I}(nf, ktask, kofs, full, nl, xp, proj, first, xc, cproj)

    Threads.@threads for t in oneto(nt)
        runs!(D, L, t)
    end

    Threads.@threads for t in oneto(nt)
        match!(D, L, wss[1].pos, t)
    end

    return L
end

function localchunk!(R, D::Decomposition{I}, Y, pre, base, typ, xwgt, wgt, xtab, vtab, ncol, tab, tval, ws::TaskWorkspace{I}, k::Int, ktask, kofs, nf, next) where {I}
    nt = D.tree.nu
    ko = zero(I)

    while true
        t = I(Threads.atomic_add!(next, 1))
        t > nt && break
        C = restrict!(D, ws.pos, ws.xadj, ws.adjncy, ws.vtx, ws.arc, ws.sxadj, ws.sadjncy, t)
        #
        # every element as separator, no children
        #
        m = C.inst.dom.nu
        F = Contraction{I}(zero(I), C.inst, Span{I}(zero(I), m, zero(I), ws.sxadj, ws.sadjncy))
        ws.sxadj[one(I)] = one(I)

        if isnothing(wgt)
            lwgt = nothing
        else
            lwgt = ws.lwgt[] = gather!(D, xtab, vtab, ncol, xwgt, wgt, ws.ckey, ws.cval, ws.ccol, ws.lxwgt, ws.lwgt[], t)
        end

        n, _ = contract!(R, F, Y, pre, base, typ, ws.lxwgt, lwgt, ws.ckey, ws.cval, ws.ccol, tab, tval, ws.okey, ws.oval, ko, zero(I), nothing, zero(I), ws.work)
        ktask[t] = k
        kofs[t] = ko
        nf[t] = n
        ko += n * m
    end

    return
end

#
# The runs of the columns of Full_t with the same separator (the last
# s elements of the bag; columns are colexicographic, so equal
# separators are consecutive).
#
function runs!(D::Decomposition{I}, L::LocalTables{I}, t::I) where {I}
    nr = D.res.xadj[t + one(I)] - D.res.xadj[t]
    m = nr + D.sep.xadj[t + one(I)] - D.sep.xadj[t]
    o = L.xf[t]
    F = L.full[L.kt[t]]
    n = zero(I)

    for x in oneto(L.nf[t])
        new = isone(x)

        if !new
            for k in nr + one(I):m
                if F[o + (x - 1) * m + k] != F[o + (x - 2) * m + k]
                    new = true
                    break
                end
            end
        end

        if new
            n += one(I)
            L.first[L.xp[t] + n] = x
        end

        L.proj[L.xp[t] + x] = n
    end

    L.nl[t] = n
    return
end

#
# For each column of Full_t and each child c, the run of Loc_c with the
# same values on the separator of c, by binary search (Loc_c is
# colexicographic in the separator of c).
#
function match!(D::Decomposition{I}, L::LocalTables{I}, pos, t::I) where {I}
    rstrt = D.res.xadj[t]
    nr = D.res.xadj[t + one(I)] - rstrt
    sstrt = D.sep.xadj[t]
    m = nr + D.sep.xadj[t + one(I)] - sstrt
    cstrt = D.tree.xadj[t]
    nc = D.tree.xadj[t + one(I)] - cstrt
    o = L.xf[t]
    F = L.full[L.kt[t]]
    #
    # the label in the bag of element g (pos is not shared: search)
    #
    function label(g)
        if rstrt <= g < rstrt + nr
            return g - rstrt + one(I)
        end

        for x in nr + one(I):m
            D.sep.adjncy[sstrt + x - nr - one(I)] == g && return x
        end

        return zero(I)
    end

    for j in oneto(nc)
        c = D.tree.adjncy[cstrt + j - one(I)]
        pstrt = D.sep.xadj[c]
        sc = D.sep.xadj[c + one(I)] - pstrt
        nrc = D.res.xadj[c + one(I)] - D.res.xadj[c]
        mc = nrc + sc
        ports = [label(D.sep.adjncy[pstrt + i - one(I)]) for i in oneto(sc)]
        oc = L.xf[c]
        Fc = L.full[L.kt[c]]

        for x in oneto(L.nf[t])
            lo = one(I)
            hi = L.nl[c]
            r = zero(I)

            while lo <= hi
                mid = (lo + hi) >>> one(I)
                y = L.first[L.xp[c] + mid]
                cmp = 0
                #
                # colexicographic: the last component is the most
                # significant
                #
                for i in sc:-one(I):one(I)
                    a = F[o + (x - 1) * m + ports[i]]
                    b = Fc[oc + (y - 1) * mc + nrc + i]

                    if a != b
                        cmp = b < a ? -1 : 1
                        break
                    end
                end

                if iszero(cmp)
                    r = mid
                    break
                elseif cmp < 0
                    lo = mid + one(I)
                else
                    hi = mid - one(I)
                end
            end

            L.cproj[L.xc[t] + (x - 1) * nc + j] = r
        end
    end

    return
end

@inline function testbit(v::Vector{UInt64}, o::Int, b)
    return isone((v[o + ((b - 1) >> 6) + 1] >> ((b - 1) & 63)) & one(UInt64))
end

@inline function setbit!(v::Vector{UInt64}, o::Int, b)
    v[o + ((b - 1) >> 6) + 1] |= one(UInt64) << ((b - 1) & 63)
    return
end

#
# Whether column x of Full_t matches a nonzero entry of the message of
# each done child (except skip).
#
function feasible(D::Decomposition{I}, L::LocalTables{I}, done, msg, t::I, x, skip) where {I}
    cstrt = D.tree.xadj[t]
    nc = D.tree.xadj[t + one(I)] - cstrt

    for j in oneto(nc)
        c = D.tree.adjncy[cstrt + j - one(I)]

        if c != skip && done[c]
            r = L.cproj[L.xc[t] + (x - 1) * nc + j]

            if iszero(r) || !testbit(msg[c], 0, r)
                return false
            end
        end
    end

    return true
end

#
# The coin of node t in round r of random mate.
#
function coin(t, r)
    x = (UInt64(t) * 0x9e3779b97f4a7c15) ⊻ (UInt64(r) * 0xbf58476d1ce4e5b9)
    x ⊻= x >> 31
    x *= 0x94d049bb133111eb
    x ⊻= x >> 29
    return isone(x & one(UInt64))
end

#
# The messages of every node (bitsets over Loc_t) by tree contraction.
# Returns nothing if some message is empty (no homomorphism).
#
function contracttree(P::Threaded, D::Decomposition{I}, L::LocalTables{I}) where {I}
    nt = D.tree.nu
    parent = zeros(I, nt)

    for t in oneto(nt), j in D.tree.xadj[t]:D.tree.xadj[t + one(I)] - one(I)
        parent[D.tree.adjncy[j]] = t
    end

    nwd = [cld(Int(L.nl[t]), 64) for t in oneto(nt)]
    msg = [zeros(UInt64, nwd[t]) for t in oneto(nt)]
    #
    # flags are Vector{Bool}, not BitVector: tasks write neighbors
    #
    done = zeros(Bool, nt)
    nund = [D.tree.xadj[t + one(I)] - D.tree.xadj[t] for t in oneto(nt)]
    chain = zeros(Bool, nt)
    ptr = zeros(I, nt)
    slot = zeros(I, nt)
    M = Vector{Vector{UInt64}}(undef, nt)
    rem = zeros(Int, nt)
    rptr = zeros(I, nt)
    absorb = zeros(Bool, nt)
    newd = zeros(Bool, nt)
    empty = Threads.Atomic{Bool}(false)
    phase = 0

    while !done[nt]
        phase += 1
        #
        # chain nodes (one undone child) and their matrices
        #
        Threads.@threads for t in oneto(nt)
            chain[t] = !done[t] && isone(nund[t])
            absorb[t] = false
            rem[t] = 0

            if chain[t]
                cstrt = D.tree.xadj[t]
                nc = D.tree.xadj[t + one(I)] - cstrt

                for j in oneto(nc)
                    c = D.tree.adjncy[cstrt + j - one(I)]

                    if !done[c]
                        ptr[t] = c
                        slot[t] = j
                    end
                end

                c = ptr[t]
                A = zeros(UInt64, Int(L.nl[c]) * nwd[t])

                for x in oneto(L.nf[t])
                    a = L.cproj[L.xc[t] + (x - 1) * nc + slot[t]]

                    if ispositive(a) && feasible(D, L, done, msg, t, x, c)
                        setbit!(A, (a - 1) * nwd[t], L.proj[L.xp[t] + x])
                    end
                end

                M[t] = A
            end
        end
        #
        # random mate: a chain node with heads absorbs its chain child
        # with tails
        #
        r = 0

        while true
            r += 1
            more = Threads.Atomic{Bool}(false)

            Threads.@threads for t in oneto(nt)
                absorb[t] = false

                if chain[t] && iszero(rem[t]) && chain[ptr[t]]
                    more[] = true
                    absorb[t] = coin(t, r + 1000 * phase) && !coin(ptr[t], r + 1000 * phase)
                end
            end

            more[] || break
            Mnew = Vector{Vector{UInt64}}(undef, nt)

            Threads.@threads for t in oneto(nt)
                if absorb[t]
                    Mnew[t] = compose(M[ptr[t]], nwd[ptr[t]], M[t], nwd[t], Int(L.nl[ptr[ptr[t]]]))
                end
            end

            for t in oneto(nt)
                if absorb[t]
                    u = ptr[t]
                    rem[u] = r
                    rptr[u] = ptr[u]
                    ptr[t] = ptr[u]
                    M[t] = Mnew[t]
                end
            end
        end

        nmate = r - 1
        #
        # rake: nodes whose children are done
        #
        Threads.@threads for t in oneto(nt)
            newd[t] = !done[t] && iszero(nund[t])

            if newd[t]
                v = msg[t]
                fill!(v, zero(UInt64))

                for x in oneto(L.nf[t])
                    if feasible(D, L, done, msg, t, x, zero(I))
                        setbit!(v, 0, L.proj[L.xp[t] + x])
                    end
                end
            end
        end

        finish!(newd, done, nund, parent, msg, empty) || return nothing
        #
        # chains whose bottom is done, then the absorbed nodes in reverse
        #
        Threads.@threads for t in oneto(nt)
            newd[t] = chain[t] && iszero(rem[t]) && done[ptr[t]]
            newd[t] && apply!(msg[t], msg[ptr[t]], M[t], nwd[t])
        end

        finish!(newd, done, nund, parent, msg, empty) || return nothing

        for q in nmate:-1:1
            Threads.@threads for t in oneto(nt)
                newd[t] = rem[t] == q && done[rptr[t]]
                newd[t] && apply!(msg[t], msg[rptr[t]], M[t], nwd[t])
            end

            finish!(newd, done, nund, parent, msg, empty) || return nothing
        end
    end

    return msg
end

#
# Mark the new nodes done; false if one has an empty message.
#
function finish!(newd, done, nund, parent, msg, empty)
    for t in eachindex(newd)
        if newd[t]
            done[t] = true
            p = parent[t]
            ispositive(p) && (nund[p] -= 1)
            all(iszero, msg[t]) && return false
        end
    end

    return true
end

#
# v = u ⋅ A, where A has a row of nw words for each entry of u
#
function apply!(v::Vector{UInt64}, u::Vector{UInt64}, A::Vector{UInt64}, nw::Int)
    fill!(v, zero(UInt64))

    for k in eachindex(u)
        w = u[k]

        while !iszero(w)
            a = (k - 1) * 64 + trailing_zeros(w)

            for i in 1:nw
                v[i] |= A[a * nw + i]
            end

            w &= w - one(UInt64)
        end
    end

    return
end

#
# The product A ⋅ B of Boolean matrices: A has na rows of nwa words, B
# has a row of nwb words for each column of A.
#
function compose(A::Vector{UInt64}, nwa::Int, B::Vector{UInt64}, nwb::Int, na::Int)
    C = zeros(UInt64, na * nwb)

    for a in 0:na - 1
        for k in 1:nwa
            w = A[a * nwa + k]

            while !iszero(w)
                b = (k - 1) * 64 + trailing_zeros(w)

                for i in 1:nwb
                    C[a * nwb + i] |= B[b * nwb + i]
                end

                w &= w - one(UInt64)
            end
        end
    end

    return C
end

#
# The Boolean search by tree contraction: the value and, for a
# witness, the assignment of solve.
#
function contractsolve(P::Threaded, R::Boolean, W, D::Decomposition{I}, Y::Instance{I}, witness::Bool) where {I}
    Y, yord = normalize(Y)
    typ = invvtx(Y)
    base, pre = preimages(Y, typ)
    xwgt, wgt = weights(W, D, typ, Bool)
    L = localtables(P, R, D, Y, pre, base, typ, xwgt, wgt)
    nt = D.tree.nu

    for t in oneto(nt)
        iszero(L.nf[t]) && return false, nothing
    end

    msg = contracttree(P, D, L)

    if isnothing(msg)
        return false, nothing
    elseif !witness
        return true, nothing
    end
    #
    # the witness, from the root: the first column of the chosen run
    # whose children match their messages (the back-pointer of solve)
    #
    nu = D.inst.dom.nu
    h = FVector{I}(undef, nu)
    choice = zeros(I, nt)
    choice[nt] = one(I)
    done = ones(Bool, nt)

    for t in nt:-one(I):one(I)
        rstrt = D.res.xadj[t]
        nr = D.res.xadj[t + one(I)] - rstrt
        m = nr + D.sep.xadj[t + one(I)] - D.sep.xadj[t]
        cstrt = D.tree.xadj[t]
        nc = D.tree.xadj[t + one(I)] - cstrt
        b = choice[t]
        x = L.first[L.xp[t] + b]

        while !feasible(D, L, done, msg, t, x, zero(I))
            x += one(I)
        end

        for k in oneto(nr)
            h[rstrt + k - one(I)] = L.full[L.kt[t]][L.xf[t] + (x - 1) * m + k]
        end

        for j in oneto(nc)
            c = D.tree.adjncy[cstrt + j - one(I)]
            choice[c] = L.cproj[L.xc[t] + (x - 1) * nc + j]
        end
    end

    out = FVector{I}(undef, nu)

    for r in oneto(nu)
        out[D.res.adjncy[r]] = yord[h[r]]
    end

    return true, out
end
