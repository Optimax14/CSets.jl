include("workloads.jl"); include("stats.jl"); include("CSetsGPU.jl"); using .CSetsGPU
using CSets: score, Boolean, Lawvere, Probability
X = graphinstance(3, [(1, 2), (2, 3)]); D = decomposition(X, graphsupport(X); alg=MF()); Y = complete(3)
for R in (Probability(), Lawvere())
    W = Any[[Float64(i + j) for i in 1:3, j in 1:3], nothing]
    alg = CSetsGPU.algebra(R, Float64)
    println(R, " cpu=", score(R, W, D, Y), " gpu=", CSetsGPU.color(CSetsGPU.prepare(D), Instance{Int}[Y]; S=alg, Ws=Any[W], witness=false, minwork=10^9)[1],
       " noW cpu=", score(R, nothing, D, Y), " gpu=", CSetsGPU.color(CSetsGPU.prepare(D), Instance{Int}[Y]; S=alg, witness=false, minwork=10^9)[1])
end
R = Lawvere(); alg = CSetsGPU.algebra(R, Float64)
W = Any[[Float64(i + j) for i in 1:3, j in 1:3], nothing]; W[1][2, 2] = Inf
println("inf: cpu=", score(R, W, D, Y), " gpu=", CSetsGPU.color(CSetsGPU.prepare(D), Instance{Int}[Y]; S=alg, Ws=Any[W], witness=false, minwork=10^9)[1])
W2 = Any[[Float64(i * j) for i in 1:3, j in 1:3], nothing]
println("two: cpu=", score(R, W, D, Y), " ", score(R, W2, D, Y), " gpu=", CSetsGPU.color(CSetsGPU.prepare(D), Instance{Int}[Y, Y]; S=alg, Ws=Any[W, W2], witness=false, minwork=10^9))
println("budget: gpu=", CSetsGPU.color(CSetsGPU.prepare(D), Instance{Int}[Y]; S=alg, Ws=Any[W2], witness=false, minwork=0, budget=4, nitems=16))
for kw in ((minwork=0, nitems=16), (budget=4,), (minwork=0, nitems=16, budget=0))
println(kw, " gpu=", CSetsGPU.color(CSetsGPU.prepare(D), Instance{Int}[Y]; S=alg, Ws=Any[W2], witness=false, kw...))
end
