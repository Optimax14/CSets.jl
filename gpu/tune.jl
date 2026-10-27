include("workloads.jl"); include("stats.jl"); include("CSetsGPU.jl"); using .CSetsGPU
rng = MersenneTwister(2)
nv, E = partialktree(rng, 2000, 3, 0.8); X = graphinstance(nv, E); Y = symmetric(30, randomgraph(rng, 30, 0.3))
D = decomposition(X, graphsupport(X); alg=MF())
for kw in ((nitems=1<<15,), (nitems=1<<17,), (nitems=1<<19,), (nitems=1<<17, maxround=12), (nitems=1<<19, maxround=12), (nitems=1<<17, threads=64), (nitems=1<<17, threads=256))
    CSetsGPU.color(D, Y; kw...)
    st = Dict{Symbol,Any}(); t = minimum(@elapsed(CSetsGPU.color(D, Y; stats=st, kw...)) for _ in 1:2)
    println(kw, " ", round(t, digits=4), " rounds=", st[:rounds])
end
