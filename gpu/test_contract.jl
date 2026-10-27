include("workloads.jl"); include("stats.jl"); include("contract.jl")
rng = MersenneTwister(11)
nok = 0; N = 0; nsat = 0
function chk(X, S, Y)
    D = decomposition(X, S; alg=MF())
    hc = color(D, Y)
    hd, st = contract(D, Y)
    ok = isnothing(hc) == isnothing(hd) && (isnothing(hd) || ishom(X, Y, hd))
    ok || println("MISMATCH cpu=", isnothing(hc), " contract=", isnothing(hd))
    return ok, !isnothing(hc), st
end
B = instance(boolean())
for n in (4, 8, 16, 33, 100)
    for f in (adder2!, (c, a, b) -> buggy!(c, a, b, max(2, n ÷ 2)))
        c, = miter(n, adder!, f); X = instance(c)
        o, sat, st = chk(X, support(X), B); global nok += o; global N += 1; global nsat += sat
        println("miter n=$n sat=$sat ", st)
    end
end
for trial in 1:200
    n = rand(rng, 5:120); k = rand(rng, 1:4); p = rand(rng, 0.5:0.1:1.0)
    nv, E = partialktree(rng, n, k, p; bushy=rand(rng, Bool))
    isempty(E) && continue
    X = graphinstance(nv, E)
    ny = rand(rng, 3:10)
    Y = rand(rng, Bool) ? complete(rand(rng, 2:5)) : symmetric(ny, randomgraph(rng, ny, rand(rng, 0.3:0.1:0.7)))
    o, sat, st = chk(X, graphsupport(X), Y); global nok += o; global N += 1; global nsat += sat
end
println("ok $nok / $N (sat $nsat)")
