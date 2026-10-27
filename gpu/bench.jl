include("workloads.jl")
include("stats.jl")
include("CSetsGPU.jl")
using .CSetsGPU
function bench(name, X, S, Y; reps=2, kw...)
    D = decomposition(X, S; alg=MF())
    hc = color(D, Y)
    tc = minimum(@elapsed(color(D, Y)) for _ in 1:reps)
    st = Dict{Symbol,Any}()
    hg = CSetsGPU.color(D, Y; kw...)
    tg = minimum(@elapsed(CSetsGPU.color(D, Y; stats=st, kw...)) for _ in 1:reps)
    ok = isnothing(hc) == isnothing(hg) && (isnothing(hg) || ishom(X, Y, hg))
    println(rpad(name, 32), " cpu=", round(tc, digits=4), " gpu=", round(tg, digits=4), " (fwd ", round(st[:fwd], digits=4), " [search kernels ", round(get(st, :kernel, 0.0), digits=4), "]",
        " bwd ", round(get(st, :bwd, 0.0), digits=4), ") x", round(tc / tg, digits=2), " levels=", st[:levels], " rounds=", st[:rounds],
        " tabwords=", st[:ntab], " ok=", ok)
    flush(stdout)
end
B = instance(boolean())
rng = MersenneTwister(1)
which = isempty(ARGS) ? "all" : ARGS[1]
if which in ("all", "circ")
for n in (1000, 10000)
    c, = miter(n, adder!, adder2!); X = instance(c); bench("adder≡adder2 n=$n", X, support(X), B)
    c, = miter(n, adder!, (c, a, b) -> buggy!(c, a, b, n ÷ 2)); X = instance(c); bench("adder vs buggy n=$n", X, support(X), B)
end
end
if which in ("all", "graph")
for (n, k, q, p, bushy) in ((20000, 3, 4, 0.8, true), (20000, 4, 5, 0.7, true), (2000, 6, 7, 0.6, true), (20000, 4, 5, 0.7, false))
    nv, E = partialktree(rng, n, k, p; bushy); X = graphinstance(nv, E)
    bench("pk$k n=$n →K$q bushy=$bushy", X, graphsupport(X), complete(q))
end
for (w, l, q) in ((4, 2000, 3), (8, 500, 3))
    nv, E = grid(w, l); X = graphinstance(nv, E)
    bench("grid $w×$l →K$q", X, graphsupport(X), complete(q))
end
for (n, k, p, ny, py) in ((2000, 3, 0.8, 30, 0.3), (1000, 4, 0.7, 40, 0.3))
    nv, E = partialktree(rng, n, k, p); X = graphinstance(nv, E)
    Y = symmetric(ny, randomgraph(rng, ny, py))
    bench("pk$k n=$n →G($ny,$py)", X, graphsupport(X), Y)
end
end
