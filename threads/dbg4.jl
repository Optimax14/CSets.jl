include("workloads.jl")
using CSets: CSets, score, Boolean, Threaded
rng = MersenneTwister(3)
nv, E = partialktree(rng, 40, 2, 0.9; bushy=true)
X = graphinstance(nv, E); D = decomposition(X, graphsupport(X); alg=MF()); Y = complete(3)
for P in (Threaded(1; minwork=0), Threaded(2; minwork=10^9), Threaded(2; minwork=0))
    try
        println(P, " ", score(P, Boolean(), nothing, D, Y))
    catch e
        println(P, " FAIL ", typeof(e))
    end
end
