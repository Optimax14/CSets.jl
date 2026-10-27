include("workloads.jl")
include("stats.jl")
B = instance(boolean())
for n in (64, 1000, 10000)
    c, = miter(n, adder!, adder2!); X = instance(c); profile("adder≡adder2 n=$n", X, support(X), B)
    c, = miter(n, adder!, (c, a, b) -> buggy!(c, a, b, n ÷ 2)); X = instance(c); profile("adder vs buggy n=$n", X, support(X), B)
end
rng = MersenneTwister(1)
for (n, k, q, p, bushy) in ((20000, 3, 4, 0.8, true), (20000, 4, 5, 0.7, true), (20000, 5, 4, 0.6, true), (2000, 6, 7, 0.6, true), (20000, 4, 5, 0.7, false))
    nv, E = partialktree(rng, n, k, p; bushy); X = graphinstance(nv, E)
    profile("pk$k-tree n=$n →K$q bushy=$bushy", X, graphsupport(X), complete(q))
end
for (w, l, q) in ((4, 2000, 3), (6, 1000, 3), (8, 500, 3))
    nv, E = grid(w, l); X = graphinstance(nv, E)
    profile("grid $w×$l →K$q", X, graphsupport(X), complete(q))
end
for (n, k, p, ny, py) in ((2000, 3, 0.8, 30, 0.3), (1000, 4, 0.7, 40, 0.3))
    nv, E = partialktree(rng, n, k, p); X = graphinstance(nv, E)
    Y = symmetric(ny, randomgraph(rng, ny, py))
    profile("pk$k-tree n=$n →G($ny,$py)", X, graphsupport(X), Y)
end
