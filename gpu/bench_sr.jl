include("workloads.jl"); include("stats.jl"); include("CSetsGPU.jl"); using .CSetsGPU
using CSets: score, Boolean, Lawvere, Probability, sprod, sone
include("rwutil.jl")
rng = MersenneTwister(5)
for (n, k, p, ny, py) in ((2000, 3, 0.8, 30, 0.3), (1000, 4, 0.7, 25, 0.3))
    nv, E = partialktree(rng, n, k, p); X = graphinstance(nv, E); D = decomposition(X, graphsupport(X); alg=MF())
    Y = symmetric(ny, randomgraph(rng, ny, py))
    for R in (Boolean(), Lawvere(), Probability())
        W = randweights(rng, R, Y, nv)
        sc = score(R, W, D, Y); tc = @elapsed score(R, W, D, Y)
        sg = CSetsGPU.score(R, W, D, Y); tg = minimum(@elapsed(CSetsGPU.score(R, W, D, Y)) for _ in 1:2)
        ok = isapprox(sc, sg; rtol=1e-9) || sc == sg
        println("pk$k-n$n-G$ny $R score cpu=", round(tc, digits=3), " gpu=", round(tg, digits=3), " x", round(tc / tg, digits=1), " ok=", ok); flush(stdout)
    end
end
