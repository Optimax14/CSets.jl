include("workloads.jl"); include("stats.jl"); include("CSetsGPU.jl"); using .CSetsGPU
rng = MersenneTwister(1)
for (n, k, q, p, bushy) in ((20000, 3, 4, 0.8, true), (20000, 4, 5, 0.7, true), (2000, 6, 7, 0.6, true), (20000, 4, 5, 0.7, false)); partialktree(rng, n, k, p; bushy); end
for (w, l, q) in ((4, 2000, 3), (8, 500, 3)); end
nv, E = partialktree(rng, 2000, 3, 0.8); X = graphinstance(nv, E); Y = symmetric(30, randomgraph(rng, 30, 0.3))
D = decomposition(X, graphsupport(X); alg=MF())
for kw in ((;), (; minwork=0), (; minwork=1024))
    CSetsGPU.color(D, Y; kw...)
    st = Dict{Symbol,Any}()
    t = @elapsed CSetsGPU.color(D, Y; stats=st, kw...)
    println(kw, " ", round(t, digits=4), " rounds=", st[:rounds])
end
