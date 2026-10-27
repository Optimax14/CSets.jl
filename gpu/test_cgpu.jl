include("workloads.jl"); include("stats.jl"); include("CSetsGPU.jl"); using .CSetsGPU
rng = MersenneTwister(5)
nok = 0; N = 0; nsat = 0; nwide = 0
function chk(X, S, Y; kw...)
    D = decomposition(X, S; alg=MF())
    hc = color(D, Y)
    hd = rand(Bool) ? CSetsGPU.contract(D, Y; kw...) : CSetsGPU.auto(D, Y; kw...)
    hd === :wide && return true, false, true
    ok = isnothing(hc) == isnothing(hd) && (isnothing(hd) || ishom(X, Y, hd))
    ok || println("MISMATCH cpu=", isnothing(hc), " contract=", isnothing(hd))
    return ok, !isnothing(hc), false
end
B = instance(boolean())
for n in (4, 8, 16, 33, 100)
    for f in (adder2!, (c, a, b) -> buggy!(c, a, b, max(2, n ÷ 2)))
        c, = miter(n, adder!, f); X = instance(c)
        o, sat, w = chk(X, support(X), B; minwork=rand(rng, (0, 256))); global nok += o; global N += 1; global nsat += sat; global nwide += w
    end
end
for trial in 1:300
    n = rand(rng, 5:150); k = rand(rng, 1:4); p = rand(rng, 0.5:0.1:1.0)
    nv, E = partialktree(rng, n, k, p; bushy=rand(rng, Bool))
    isempty(E) && continue
    X = graphinstance(nv, E)
    ny = rand(rng, 3:8)
    Y = rand(rng, Bool) ? complete(rand(rng, 2:4)) : symmetric(ny, randomgraph(rng, ny, rand(rng, 0.3:0.1:0.7)))
    o, sat, w = chk(X, graphsupport(X), Y); global nok += o; global N += 1; global nsat += sat; global nwide += w
end
println("ok $nok / $N (sat $nsat, wide $nwide)")
