include("workloads.jl")
using CSets: CSets, score, Boolean, Threaded
X = graphinstance(4, [(1, 2), (2, 3), (3, 4)]); D = decomposition(X, graphsupport(X); alg=MF()); Y = complete(3)
println(score(Boolean(), nothing, D, Y))
println(score(Threaded(1), Boolean(), nothing, D, Y))
