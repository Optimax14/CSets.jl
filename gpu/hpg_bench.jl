# usage: julia hpg_bench.jl cpu|gpu big|contract out.tsv
include("workloads.jl"); include("stats.jl")
const MODE = ARGS[1]; const SUITE = ARGS[2]; const OUT = ARGS[3]
MODE == "gpu" && (include("CSetsGPU.jl"); using .CSetsGPU)
B = instance(boolean())
function pk(seed, n, k, p, bushy, Yf)
    rng = MersenneTwister(seed); nv, E = partialktree(rng, n, k, p; bushy); X = graphinstance(nv, E)
    return X, graphsupport(X), Yf(rng)
end
randY(ny, py) = rng -> symmetric(ny, randomgraph(rng, ny, py))
function work(name, X, S, Y; reps=2)
    td = minimum(@elapsed(decomposition(X, S; alg=MF())) for _ in 1:reps)
    D = decomposition(X, S; alg=MF())
    sh = shape(D)
    if MODE == "cpu"
        h = color(D, Y); t = h === nothing && false ? 0.0 : Inf
        t = minimum(@elapsed(color(D, Y)) for _ in 1:reps)
        extra = ""
    elseif SUITE == "contract"
        h = CSetsGPU.contract(D, Y)
        h === :wide && (println(name, " wide"); return)
        st = Dict{Symbol,Any}(); t = Inf
        for _ in 1:reps
            s1 = Dict{Symbol,Any}(); t1 = @elapsed CSetsGPU.contract(D, Y; stats=s1); t1 < t && (t = t1; st = s1)
        end
        extra = "local=$(round(get(st, :local, 0.0), digits=4)) contract=$(round(get(st, :contract, 0.0), digits=4)) witness=$(round(get(st, :witness, 0.0), digits=4)) phases=$(get(st, :phases, 0)) jumps=$(get(st, :jumps, 0))"
    else
        h = CSetsGPU.color(D, Y)
        st = Dict{Symbol,Any}(); t = Inf
        for _ in 1:reps
            s1 = Dict{Symbol,Any}(); t1 = @elapsed CSetsGPU.color(D, Y; stats=s1); t1 < t && (t = t1; st = s1)
        end
        extra = "plan=$(round(get(st, :plan, 0.0), digits=4)) fwd=$(round(get(st, :fwd, 0.0), digits=4)) kern=$(round(get(st, :kernel, 0.0), digits=4)) bwd=$(round(get(st, :bwd, 0.0), digits=4)) rounds=$(get(st, :rounds, 0)) tab=$(get(st, :ntab, 0))"
    end
    sat = !isnothing(h)
    ok = !sat || ishom(X, Y, h)
    line = join([name, MODE, round(td, digits=4), round(t, digits=4), sat, ok, sh.nt, sh.depth, sh.maxbag, sh.maxsep, extra], '\t')
    println(line); flush(stdout)
    open(OUT, "a") do io; println(io, line); end
end
if SUITE == "big"
    for (n, k, q, p) in ((20000, 4, 5, 0.7), (5000, 6, 7, 0.6), (20000, 5, 6, 0.6))
        X, S, Y = pk(1, n, k, p, true, _ -> complete(q)); work("pk$k-n$n-K$q", X, S, Y)
    end
    for (n, k, p, ny, py) in ((5000, 3, 0.8, 30, 0.3), (1000, 4, 0.7, 40, 0.3), (3000, 4, 0.7, 40, 0.3), (1000, 5, 0.7, 60, 0.25))
        X, S, Y = pk(2, n, k, p, true, randY(ny, py)); work("pk$k-n$n-G$ny", X, S, Y)
    end
    for (w, l, q) in ((8, 500, 3), (10, 200, 3), (6, 1000, 4), (7, 500, 4))
        nv, E = grid(w, l); X = graphinstance(nv, E); work("grid$(w)x$l-K$q", X, graphsupport(X), complete(q))
    end
else
    for n in (10000, 30000, 100000)
        c, = miter(n, adder!, adder2!); X = instance(c); work("miter-eq-$n", X, support(X), B)
        c, = miter(n, adder!, (c, a, b) -> buggy!(c, a, b, n ÷ 2)); X = instance(c); work("miter-bug-$n", X, support(X), B)
    end
    for (w, l) in ((4, 20000), (4, 100000))
        nv, E = grid(w, l); X = graphinstance(nv, E); work("grid$(w)x$l-K3", X, graphsupport(X), complete(3))
    end
end
