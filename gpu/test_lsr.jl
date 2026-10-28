# level-synchronous semiring search (values + weights + witness) vs upstream CPU
include("workloads.jl"); include("stats.jl"); include("CSetsGPU.jl"); using .CSetsGPU
using CSets: score, Boolean, Lawvere, Probability, sprod, sone
include("rwutil.jl")
rng = MersenneTwister(parse(Int, get(ENV, "SEED", "31")))
nok = 0; N = 0
for trial in 1:80
    n = rand(rng, 4:60); k = rand(rng, 1:3)
    nv, E = partialktree(rng, n, k, rand(rng, 0.5:0.1:1.0); bushy=rand(rng, Bool))
    isempty(E) && continue
    X = graphinstance(nv, E); S = graphsupport(X)
    D = decomposition(X, S; alg=MF())
    R = rand(rng, (Boolean(), Lawvere(), Probability()))
    T = R isa Boolean ? Bool : Float64
    Ys = Any[]; Wsv = Any[]
    for _ in 1:rand(rng, 1:5)
        ny = rand(rng, 2:6)
        Y = rand(rng, Bool) ? complete(rand(rng, 2:5)) : symmetric(ny, randomgraph(rng, ny, rand(rng, 0.4:0.1:0.9)))
        push!(Ys, Y); push!(Wsv, randweights(rng, R, Y, nv))
    end
    alg = R isa Boolean ? nothing : CSetsGPU.algebra(R, T)
    P = CSetsGPU.prepare(D)
    kw = (; minwork=rand(rng, (0, 256)), budget=rand(rng, (4, 1 << 14)), nitems=rand(rng, (0, 16)))
    if R isa Boolean
        hs = CSetsGPU.color(P, Instance{Int}[Ys...]; Ws=Wsv, kw...)
        for (i, Y) in enumerate(Ys)
            vc, hc = color(R, Wsv[i], D, Y)
            ok = isnothing(hc) == isnothing(hs[i]) && (isnothing(hs[i]) || (ishom(X, Y, hs[i]) && hweight(R, Wsv[i], X, Y, hs[i]) == true))
            ok || println("MISMATCH bool trial $trial target $i")
            global nok += ok; global N += 1
        end
    else
        rs = CSetsGPU.color(P, Instance{Int}[Ys...]; S=alg, Ws=Wsv, witness=CSets.isselective(R), kw...)
        for (i, Y) in enumerate(Ys)
            sc = score(R, Wsv[i], D, Y)
            vg, hg = rs[i]
            ok = isapprox(sc, vg; rtol=1e-9) || sc == vg
            if CSets.isselective(R) && sc != CSets.szero(R, T)
                ok &= !isnothing(hg) && ishom(X, Y, hg) && isapprox(hweight(R, Wsv[i], X, Y, hg), sc; rtol=1e-9)
            end
            ok || println("MISMATCH $R trial $trial target $i cpu=$sc gpu=$vg hnothing=", isnothing(hg))
            global nok += ok; global N += 1
        end
    end
end
println("lsr ok $nok / $N")
