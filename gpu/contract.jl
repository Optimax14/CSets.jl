#
# Design (d): tree contraction. A sequential reference of a
# round-synchronous algorithm: every loop marked "parallel" has
# independent iterations and would be one GPU kernel. Counts the
# rounds (dependent steps) and the work.
#
# Each bag t has its local table Full_t: the assignments of the bag
# consistent with the edges inside it, ignoring the children. Its
# separator domain Loc_t is the set of their projections onto the
# separator. A child c of t sees x ∈ Full_t through proj[t, c][x] ∈ Loc_c.
#
# The message of a bag is a Boolean vector v_t over Loc_t: the
# separator assignments that extend to its subtree (the CPU table of t
# is Loc_t filtered by v_t). With all children but one (c) known, the
# map v_c ↦ v_t is a Boolean matrix M_t : Loc_c × Loc_t, so chains of
# such bags compose by Boolean matrix products: pointer jumping
# instead of a serial walk.
#
using CSets: Decomposition, Span, FVector, normalize

struct Local
    m::Vector{Int}
    s::Vector{Int}
    ord::Vector{Vector{Int}}
    full::Vector{Matrix{Int}}     # m × |Full_t|
    loc::Vector{Matrix{Int}}      # s × |Loc_t|, sorted
    proj::Vector{Vector{Int}}     # Full_t → Loc_t
    cproj::Vector{Vector{Vector{Int}}}  # per child: Full_t → Loc_c (0 if absent)
    children::Vector{Vector{Int}}
    parent::Vector{Int}
end

function lexsearch(L::Matrix{Int}, key)
    lo, hi = 1, size(L, 2)
    while lo <= hi
        mid = (lo + hi) >>> 1
        c = 0
        for i in eachindex(key)
            if L[i, mid] != key[i]
                c = L[i, mid] < key[i] ? -1 : 1
                break
            end
        end
        c == 0 && return mid
        c < 0 ? (lo = mid + 1) : (hi = mid - 1)
    end
    return 0
end

function locals(D::Decomposition, Y)
    X = D.inst
    nt = D.tree.nu
    nu = X.dom.nu
    #
    # the bag as its own separator: bagsearch! then lists every local
    # assignment, in the order of the original bag
    #
    xadj = Int[1]; adjncy = Int[]
    for t in 1:nt
        append!(adjncy, D.res.xadj[t]:D.res.xadj[t + 1] - 1)
        append!(adjncy, D.sep.adjncy[D.sep.xadj[t]:D.sep.xadj[t + 1] - 1])
        push!(xadj, length(adjncy) + 1)
    end
    sep = Span{Int}(nt, nu, length(adjncy), FVector{Int}(xadj), FVector{Int}(adjncy))
    res = Span{Int}(nt, nu, nu, FVector{Int}(ones(Int, nt + 1)), D.res.adjncy)
    T0 = Span{Int}(nt, nt, 0, FVector{Int}(ones(Int, nt + 1)), FVector{Int}(Int[]))
    ph = phases(Decomposition{Int}(X, res, sep, T0), Y)
    ph.ok || return nothing
    m = [xadj[t + 1] - xadj[t] for t in 1:nt]
    s = [D.sep.xadj[t + 1] - D.sep.xadj[t] for t in 1:nt]
    ord = [reverse(adjncy[xadj[t]:xadj[t + 1] - 1]) for t in 1:nt]
    full = [reshape(ph.tab[ph.xtab[t]:ph.xtab[t] + m[t] * ph.ncol[t] - 1], m[t], ph.ncol[t]) for t in 1:nt]
    loc = Matrix{Int}[]; proj = Vector{Int}[]
    for t in 1:nt
        F = full[t]; n = size(F, 2)
        p = zeros(Int, n); cols = Int[]
        for x in 1:n
            if x == 1 || F[1:s[t], x] != F[1:s[t], x - 1]
                push!(cols, x)
            end
            p[x] = length(cols)
        end
        push!(loc, F[1:s[t], cols]); push!(proj, p)
    end
    children = [collect(D.tree.adjncy[D.tree.xadj[t]:D.tree.xadj[t + 1] - 1]) for t in 1:nt]
    parent = zeros(Int, nt)
    for t in 1:nt, c in children[t]
        parent[c] = t
    end
    pos = zeros(Int, nu)
    cproj = Vector{Vector{Int}}[]
    for t in 1:nt
        for k in 1:m[t]
            pos[ord[t][k]] = k
        end
        F = full[t]
        push!(cproj, [begin
            pk = [pos[ord[c][i]] for i in 1:s[c]]
            [lexsearch(loc[c], F[pk, x]) for x in 1:size(F, 2)]
        end for c in children[t]])
    end
    return Local(m, s, ord, full, loc, proj, cproj, children, parent)
end

# Boolean matrices as Vector{BitVector} rows
function compose(A, B, nb)   # (La × Lb) ∘ (Lb × Lc)
    C = [falses(nb) for _ in eachindex(A)]
    w = 0
    for a in eachindex(A)
        for b in findall(A[a])
            C[a] .|= B[b]
            w += cld(nb, 64)
        end
    end
    return C, w
end

function contract(D::Decomposition, Y)
    Ln = locals(D, Y)
    isnothing(Ln) && return nothing, (; rounds=0, work=0)
    nt = length(Ln.m)
    L = [size(Ln.loc[t], 2) for t in 1:nt]
    done = falses(nt)
    v = [falses(L[t]) for t in 1:nt]
    rounds = 0; work = 0
    nundone(t) = count(c -> !done[c], Ln.children[t])
    # mask of Full_t consistent with the done children
    function okx(t)
        F = size(Ln.full[t], 2)
        ok = trues(F)
        for (j, c) in enumerate(Ln.children[t])
            done[c] || continue
            for x in 1:F
                p = Ln.cproj[t][j][x]
                ok[x] &= p > 0 && v[c][p]
            end
        end
        return ok
    end
    phases_ = 0
    while !done[nt]
        phases_ += 1
        #
        # chains: undone bags with exactly one undone child. Build M_t
        # (parallel), then pointer-jump (log rounds of parallel products)
        #
        chain = falses(nt)
        for t in 1:nt
            chain[t] = !done[t] && nundone(t) == 1
        end
        M = Dict{Int, Vector{BitVector}}()
        ptr = zeros(Int, nt)
        for t in findall(chain)
            j = findfirst(c -> !done[c], Ln.children[t]); c = Ln.children[t][j]
            ok = okx(t)
            R = [falses(L[t]) for _ in 1:L[c]]
            for x in eachindex(ok)
                p = Ln.cproj[t][j][x]
                ok[x] && p > 0 && (R[p][Ln.proj[t][x]] = true)
            end
            work += length(ok) * (1 + length(Ln.children[t]))
            M[t] = R; ptr[t] = c
        end
        rounds += 1
        while true
            # parallel: every chain bag whose pointer is a chain bag jumps
            jumpers = [t for t in findall(chain) if chain[ptr[t]]]
            isempty(jumpers) && break
            newM = Dict{Int, Vector{BitVector}}(); newp = Dict{Int, Int}()
            for t in jumpers
                u = ptr[t]
                C, w = compose(M[u], M[t], L[t])
                work += w
                newM[t] = C; newp[t] = ptr[u]
            end
            for t in jumpers
                M[t] = newM[t]; ptr[t] = newp[t]
            end
            rounds += 1
        end
        #
        # rake: bags whose children are all done (parallel)
        #
        ready = [t for t in 1:nt if !done[t] && nundone(t) == 0]
        for t in ready
            ok = okx(t)
            for x in eachindex(ok)
                ok[x] && (v[t][Ln.proj[t][x]] = true)
            end
            work += length(ok) * (1 + length(Ln.children[t]))
        end
        done[ready] .= true
        rounds += 1
        any(t -> isempty(findall(v[t])), ready) && return nothing, (; rounds, work, phases=phases_)
        #
        # chain bags whose bottom is done: v_t = v_bottom ⋅ M_t (parallel)
        #
        fin = [t for t in findall(chain) if done[ptr[t]]]
        for t in fin
            b = ptr[t]
            for a in findall(v[b])
                v[t] .|= M[t][a]
                work += cld(L[t], 64)
            end
        end
        done[fin] .= true
        isempty(fin) || (rounds += 1)
        any(t -> !any(v[t]), fin) && return nothing, (; rounds, work, phases=phases_)
    end
    #
    # witness: x_t(b) for every b, then b_t for every bag by pointer
    # jumping on the parent pointers
    #
    xsel = Vector{Vector{Int}}(undef, nt)
    for t in 1:nt                       # parallel
        ok = okx(t)
        xs = zeros(Int, L[t])
        for x in eachindex(ok)
            b = Ln.proj[t][x]
            ok[x] && v[t][b] && xs[b] == 0 && (xs[b] = x)
        end
        xsel[t] = xs
    end
    rounds += 1
    # f_t : Loc_parent → Loc_t
    f = Vector{Vector{Int}}(undef, nt)
    anc = copy(Ln.parent)
    for t in 1:nt                       # parallel
        p = Ln.parent[t]
        p == 0 && continue
        j = findfirst(==(t), Ln.children[p])
        f[t] = [xsel[p][b] == 0 ? 0 : Ln.cproj[p][j][xsel[p][b]] for b in 1:L[p]]
    end
    rounds += 1
    broot = findfirst(v[nt])
    while any(t -> anc[t] != 0 && anc[t] != nt, 1:nt)
        g = copy(f); a2 = copy(anc)
        for t in 1:nt                   # parallel
            a = anc[t]
            (a == 0 || a == nt) && continue
            g[t] = [b == 0 ? 0 : f[t][b] for b in f[a]]   # f_t ∘ f_a
            a2[t] = anc[a]
            work += length(f[a])
        end
        f = g; anc = a2
        rounds += 1
    end
    bsel = zeros(Int, nt)
    bsel[nt] = broot
    for t in 1:nt - 1                   # parallel
        bsel[t] = f[t][broot]
    end
    _, yord = normalize(Y)
    h = zeros(Int, D.inst.dom.nu)
    for t in 1:nt                       # parallel
        x = xsel[t][bsel[t]]
        for k in Ln.s[t] + 1:Ln.m[t]
            h[Ln.ord[t][k]] = Ln.full[t][k, x]
        end
    end
    rounds += 1
    out = zeros(Int, D.inst.dom.nu)
    for r in 1:D.inst.dom.nu
        out[D.res.adjncy[r]] = yord[h[r]]
    end
    return out, (; rounds, work, phases=phases_, maxL=maximum(L), maxfull=maximum(size.(Ln.full, 2)))
end
