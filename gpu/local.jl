# Sizes for design (d): local tables (bag consistency only, children ignored)
include("workloads.jl")
include("stats.jl")
using CSets: Decomposition, Span, FVector
function localsizes(D, Y)
    nt = D.tree.nu
    T0 = Span{Int}(nt, nt, 0, FVector{Int}(ones(Int, nt + 1)), FVector{Int}(Int[]))
    D0 = Decomposition{Int}(D.inst, D.res, D.sep, T0)
    ph = phases(D0, Y)
    ph0 = phases(D, Y)
    L = ph.ncol
    # chain-composition cost estimate: compose relations along tree edges c -> t -> parent(t),
    # bitset product cost L[c] * L[t] * L[p] / 64 word-ops
    prnt = zeros(Int, nt)
    for t in 1:nt, j in D.tree.xadj[t]:D.tree.xadj[t + 1] - 1
        prnt[D.tree.adjncy[j]] = t
    end
    work = 0.0
    for c in 1:nt
        t = prnt[c]; t == 0 && continue
        p = prnt[t]; p == 0 && continue
        work += L[c] * L[t] * cld(L[p], 64)
    end
    sep = [D.sep.xadj[t + 1] - D.sep.xadj[t] for t in 1:nt]
    println("   local table cols max/mean=", maximum(L), "/", round(sum(L) / nt, digits=1),
        "  actual max/mean=", maximum(ph0.ncol), "/", round(sum(ph0.ncol) / nt, digits=1),
        "  bitset compose work ≈ ", round(work / 1e9, digits=3), " G word-ops, max L^3/64 = ", maximum(L)^3 ÷ 64,
        "  max sep=", maximum(sep))
end
B = instance(boolean())
for n in (1000, 10000)
    c, = miter(n, adder!, adder2!); X = instance(c); println("adder≡adder2 n=$n"); localsizes(decomposition(X, support(X); alg=MF()), B)
    c, = miter(n, adder!, (c, a, b) -> buggy!(c, a, b, n ÷ 2)); X = instance(c); println("adder vs buggy n=$n"); localsizes(decomposition(X, support(X); alg=MF()), B)
end
for (w, l) in ((4, 2000), (8, 500))
    nv, E = grid(w, l); X = graphinstance(nv, E); println("grid $w×$l →K3"); localsizes(decomposition(X, graphsupport(X); alg=MF()), complete(3))
end
rng = MersenneTwister(1)
nv, E = partialktree(rng, 20000, 4, 0.7; bushy=false); X = graphinstance(nv, E); println("pk4 deep →K5"); localsizes(decomposition(X, graphsupport(X); alg=MF()), complete(5))
