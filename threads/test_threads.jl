# Threaded vs serial: identical values (===) and identical witnesses, any thread count
include("workloads.jl")
using CSets: CSets, score, Boolean, Lawvere, Probability, sprod, sone, Threaded
include("rwutil.jl")
rng = MersenneTwister(parse(Int, get(ENV, "SEED", "1")))
nok = 0; N = 0
same(a, b) = a === b || (a isa AbstractFloat && b isa AbstractFloat && isnan(a) && isnan(b))
B = instance(boolean())
for trial in 1:parse(Int, get(ENV, "NTRIAL", "120"))
    if trial % 10 == 0
        n = rand(rng, 4:40); c, = miter(n, adder!, rand(rng, Bool) ? adder2! : (c, a, b) -> buggy!(c, a, b, max(2, n ÷ 2)))
        X = instance(c); S = support(X); Y = B; R = Boolean(); W = nothing
    else
        n = rand(rng, 4:80); k = rand(rng, 1:4)
        nv, E = partialktree(rng, n, k, rand(rng, 0.5:0.1:1.0); bushy=rand(rng, Bool))
        isempty(E) && continue
        X = graphinstance(nv, E); S = graphsupport(X)
        ny = rand(rng, 2:7)
        Y = rand(rng, Bool) ? complete(rand(rng, 2:5)) : symmetric(ny, randomgraph(rng, ny, rand(rng, 0.4:0.1:0.9)))
        R = rand(rng, (Boolean(), Lawvere(), Probability()))
        W = randweights(rng, R, Y, nv)
    end
    D = decomposition(X, S; alg=MF())
    P = Threaded(rand(rng, (1, 2, 3, 8, 16)); minwork=rand(rng, (0, 256)), deep=rand(rng, (1, 256)))
    ok = try
        same(score(R, W, D, Y), score(P, R, W, D, Y))
    catch e
        println("ERR trial $trial $R ntasks=$(P.ntasks) minwork=$(P.minwork) ", typeof(e)); false
    end
    ok || continue
    if CSets.isselective(R)
        vs, hs = color(R, W, D, Y); vp, hp = color(P, R, W, D, Y)
        ok &= same(vs, vp) && isnothing(hs) == isnothing(hp) && (isnothing(hs) || hs == hp)
    end
    ok || println("MISMATCH trial $trial $R ntasks=$(P.ntasks) minwork=$(P.minwork) deep=$(P.deep)")
    global nok += ok; global N += 1
end
println("threads ok $nok / $N (nthreads=$(Threads.nthreads()))")
# many targets
nm = 0; Nm = 0
for trial in 1:20
    nv, E = partialktree(rng, rand(rng, 5:60), rand(rng, 1:3), 0.8)
    isempty(E) && continue
    X = graphinstance(nv, E); D = decomposition(X, graphsupport(X); alg=MF())
    R = rand(rng, (Boolean(), Lawvere(), Probability()))
    Ys = [complete(rand(rng, 2:5)) for _ in 1:rand(rng, 1:6)]
    Ws = [randweights(rng, R, Y, nv) for Y in Ys]
    P = Threaded(rand(rng, (1, 4, 16)))
    ok = all(same.(score(P, R, Ws, D, Ys), [score(R, Ws[i], D, Ys[i]) for i in eachindex(Ys)]))
    if CSets.isselective(R)
        ok &= color(P, R, Ws, D, Ys) == [color(R, Ws[i], D, Ys[i]) for i in eachindex(Ys)]
    end
    ok || println("MISMATCH many trial $trial")
    global nm += ok; global Nm += 1
end
println("many ok $nm / $Nm")
