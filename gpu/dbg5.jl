include("workloads.jl"); include("stats.jl"); include("CSetsGPU.jl"); using .CSetsGPU
rng = MersenneTwister(2)
partialktree(rng, 2000, 3, 0.8); randomgraph(rng, 30, 0.3); partialktree(rng, 2000, 6, 0.6)
nv, E = partialktree(rng, 20000, 4, 0.7); X = graphinstance(nv, E); D = decomposition(X, graphsupport(X); alg=MF()); Y = complete(5)
P = CSetsGPU.prepare(D)
for _ in 1:3
st = Dict{Symbol,Any}(); t = @elapsed CSetsGPU.color(P, Y; stats=st); println("t=$t ", filter(p -> p[1] in (:fwd, :bwd, :kernel, :rounds, :setup, :plan), st))
end
