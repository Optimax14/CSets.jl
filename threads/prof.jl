include("workloads.jl")
using CSets: CSets, score, Boolean, Threaded
rng = MersenneTwister(1)
partialktree(rng, 2000, 3, 0.8); randomgraph(rng, 30, 0.3)
nv, E = partialktree(rng, 2000, 6, 0.6); X = graphinstance(nv, E); D = decomposition(X, graphsupport(X); alg=MF())
P = Threaded(16); score(P, Boolean(), nothing, D, complete(7))
CSets.DEBUG[] = true
t = @elapsed score(P, Boolean(), nothing, D, complete(7)); println("total $t")
