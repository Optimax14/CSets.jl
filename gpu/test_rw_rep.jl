# GPU vs CPU on Richard's (R, W) API: score and color for Boolean, Lawvere, Probability
include("workloads.jl"); include("stats.jl"); include("CSetsGPU.jl"); using .CSetsGPU
using CSets: score, Boolean, Lawvere, Probability, sprod, sone
for rep in 1:4
global rng = MersenneTwister(200 + rep)
function hweight(R, W, X, Y, h)
    # weight of h: product over elements x of sort c of W[c][rank in Y sort, rank in X sort]
    T = CSets.valtype(R, W)
    w = sone(R, T)
    for c in 1:X.cod.nu
        (isnothing(W) || isnothing(W[c])) && continue
        xs = findall(==(c), collect(X.vtx)); ys = findall(==(c), collect(Y.vtx))
        for (j, x) in enumerate(xs)
            w = sprod(R, w, W[c][findfirst(==(h[x]), ys), j])
        end
    end
    return w
end
nok = 0; N = 0
for trial in 1:150
    n = rand(rng, 4:40); k = rand(rng, 1:3)
    nv, E = partialktree(rng, n, k, rand(rng, 0.5:0.1:1.0); bushy=rand(rng, Bool))
    isempty(E) && continue
    X = graphinstance(nv, E); S = graphsupport(X)
    ny = rand(rng, 2:6)
    Y = rand(rng, Bool) ? complete(rand(rng, 2:5)) : symmetric(ny, randomgraph(rng, ny, rand(rng, 0.4:0.1:0.9)))
    nyv = count(==(1), collect(Y.vtx))
    D = decomposition(X, S; alg=MF())
    R = rand(rng, (Boolean(), Lawvere(), Probability()))
    W = if R isa Boolean
        rand(rng, Bool) ? nothing : Any[rand(rng, nyv, nv) .< 0.85, nothing]
    elseif R isa Lawvere
        Any[[rand(rng) < 0.1 ? Inf : round(10rand(rng), digits=2) for i in 1:nyv, j in 1:nv], nothing]
    else
        Any[rand(rng, nyv, nv), nothing]
    end
    sc = score(R, W, D, Y); meth = rand(rng, (:auto, :levels, :contract)); sg = CSetsGPU.score(R, W, D, Y; method=meth)
    ok = isapprox(sc, sg; rtol=1e-9) || (sc == sg)
    if CSets.isselective(R)
        vc, hc = color(R, W, D, Y); vg, hg = CSetsGPU.color(R, W, D, Y; method=meth)
        ok &= (isapprox(vc, vg; rtol=1e-9) || vc == vg) && isnothing(hc) == isnothing(hg)
        if !isnothing(hg)
            ok &= ishom(X, Y, hg) && isapprox(hweight(R, W, X, Y, hg), vc; rtol=1e-9)
        end
    end
    if !ok
        println("MISMATCH trial $trial $R score cpu=$sc gpu=$sg")
        if CSets.isselective(R)
            vc, hc = color(R, W, D, Y); vg, hg = CSetsGPU.color(R, W, D, Y; method=meth)
            println("   color cpu=$vc gpu=$vg hom=", isnothing(hg) ? "none" : ishom(X, Y, hg), " w=", isnothing(hg) ? 0 : hweight(R, W, X, Y, hg), " cpunothing=", isnothing(hc))
        end
    end
    global nok += ok; global N += 1
end
println("rw ok $nok / $N"); global nok = 0; global N = 0
end
