include("workloads.jl"); include("stats.jl"); include("CSetsGPU.jl"); using .CSetsGPU
rng = MersenneTwister(13)
nok = 0; N = 0; nsat = 0
for trial in 1:60
    n = rand(rng, 5:150); k = rand(rng, 1:4)
    nv, E = partialktree(rng, n, k, rand(rng, 0.5:0.1:1.0); bushy=rand(rng, Bool))
    isempty(E) && continue
    X = graphinstance(nv, E); D = decomposition(X, graphsupport(X); alg=MF())
    Ys = [begin ny = rand(rng, 2:9); rand(rng, Bool) ? complete(rand(rng, 2:5)) : symmetric(ny, randomgraph(rng, ny, rand(rng, 0.3:0.1:0.8))) end for _ in 1:rand(rng, 1:12)]
    Ys = [Y for Y in Ys if Y.dom.nu > 0]
    P = CSetsGPU.prepare(D)
    hs = CSetsGPU.color(P, Ys; minwork=rand(rng, (0, 256)), budget=rand(rng, (4, 1 << 14)), nitems=rand(rng, (0, 16)))
    for (i, Y) in enumerate(Ys)
        hc = color(D, Y)
        ok = isnothing(hc) == isnothing(hs[i]) && (isnothing(hs[i]) || ishom(X, Y, hs[i]))
        ok || println("MISMATCH trial $trial target $i cpu=", isnothing(hc), " gpu=", isnothing(hs[i]))
        global nok += ok; global N += 1; global nsat += !isnothing(hc)
    end
end
println("many ok $nok / $N (sat $nsat)")
