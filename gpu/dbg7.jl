include("workloads.jl"); include("stats.jl"); include("CSetsGPU.jl"); using .CSetsGPU
using CSets: score, Boolean, Lawvere, Probability, sprod, sone
include("rwutil.jl")
rng = MersenneTwister(5)
nv, E = partialktree(rng, 2000, 3, 0.8); X = graphinstance(nv, E); D = decomposition(X, graphsupport(X); alg=MF())
Y = symmetric(30, randomgraph(rng, 30, 0.3))
randweights(rng, Boolean(), Y, nv); randweights(rng, Lawvere(), Y, nv); R = Probability(); W = randweights(rng, R, Y, nv)
kw = eval(Meta.parse(ARGS[1]))
println(kw, " cpu=", score(R, W, D, Y))
println(kw, " gpu=", CSetsGPU.score(R, W, D, Y; method=:levels, kw...))
