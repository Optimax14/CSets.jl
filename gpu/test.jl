include("workloads.jl")
include("stats.jl")
include("CSetsGPU.jl")
using .CSetsGPU
using CSets: Decomposition
# CPU tables for comparison
function cputables(D, Y)
    ph = phases(D, Y)
    return ph
end
function check(name, X, S, Y; kw...)
    D = decomposition(X, S; alg=MF())
    hc = color(D, Y)
    st = Dict{Symbol,Any}(:wanttab => true)
    hg = CSetsGPU.color(D, Y; stats=st, kw...)
    ok = isnothing(hc) == isnothing(hg) && (isnothing(hg) || ishom(X, Y, hg))
    if !isnothing(hc)
        ph = phases(D, Y)
        ok &= ph.ncol == st[:ncol]
        for t in eachindex(ph.ncol)
            w = ph.s[t] * ph.ncol[t]
            ok &= ph.tab[ph.xtab[t]:ph.xtab[t] + w - 1] == st[:tab][st[:xtab][t]:st[:xtab][t] + w - 1]
        end
    end
    ok || println("MISMATCH ", name, " cpu=", isnothing(hc) ? "none" : "hom", " gpu=", isnothing(hg) ? "none" : "hom")
    return ok, isnothing(hc)
end
rng = MersenneTwister(7)
nok = 0; nsat = 0; N = 0
B = instance(boolean())
for n in (4, 8, 16, 33)
    for f in (adder2!, (c, a, b) -> buggy!(c, a, b, max(2, n ÷ 2)))
        c, = miter(n, adder!, f); X = instance(c)
        for nit in (1, 64, 1 << 15)
            o, u = check("miter $n", X, support(X), B; nitems=nit, minwork=0); global nok += o; global N += 1; global nsat += !u
        end
    end
end
for trial in 1:150
    n = rand(rng, 5:120); k = rand(rng, 1:4); p = rand(rng, 0.5:0.1:1.0)
    nv, E = partialktree(rng, n, k, p; bushy=rand(rng, Bool))
    isempty(E) && continue
    X = graphinstance(nv, E)
    ny = rand(rng, 3:12)
    Y = rand(rng, Bool) ? complete(rand(rng, 2:5)) : symmetric(ny, randomgraph(rng, ny, rand(rng, 0.3:0.1:0.7)))
    Y.dom.nu == 0 && continue
    o, u = check("pk trial $trial", X, graphsupport(X), Y; nitems=rand(rng, (1, 16, 1 << 15)), minwork=rand(rng, (0, 1 << 14)))
    global nok += o; global N += 1; global nsat += !u
end
println("ok $nok / $N  (sat $nsat)")
