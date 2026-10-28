# usage: julia [-t N] hpgcmp.jl serial|threads|gpu out.tsv [threadcounts]
# One workload list for the serial CPU code (upstream), the threaded CPU
# code (Threaded), and the GPU (CSetsGPU from the itay/gpu worktree).
const MODE = ARGS[1]; const OUT = ARGS[2]
include(joinpath(@__DIR__, "workloads.jl"))
include(joinpath(@__DIR__, "rwutil.jl"))
using CSets: CSets, score, Boolean, Lawvere, Probability, sprod, sone
const HAVE_THREADED = isdefined(CSets, :Threaded)
if MODE == "gpu"
    include(ENV["CSETSGPU"]); using .CSetsGPU
end
const ONLY = get(ENV, "ONLY", "")
r4(x) = round(x, digits=4)
emit(xs...) = (line = join(xs, '\t'); println(line); flush(stdout); open(io -> println(io, line), OUT, "a"))
function timeit(f; reps=2, cap=600.0)
    t1 = @elapsed v = f()
    t1 > cap && return t1, v
    t = t1
    for _ in 2:reps
        t = min(t, @elapsed f())
    end
    return t, v
end
function workloads()
    L = Any[]
    B = instance(boolean())
    function pk(seed, n, k, p)
        rng = MersenneTwister(seed)
        nv, E = partialktree(rng, n, k, p; bushy=true)
        return graphinstance(nv, E), nv, rng
    end
    # big targets (Boolean)
    for (name, seed, n, k, p, Yf) in (("pk4-n20000-K5", 1, 20000, 4, 0.7, _ -> complete(5)), ("pk5-n20000-K6", 1, 20000, 5, 0.6, _ -> complete(6)),
            ("pk6-n5000-K7", 1, 5000, 6, 0.6, _ -> complete(7)), ("pk3-n5000-G30", 2, 5000, 3, 0.8, rng -> symmetric(30, randomgraph(rng, 30, 0.3))),
            ("pk4-n1000-G40", 2, 1000, 4, 0.7, rng -> symmetric(40, randomgraph(rng, 40, 0.3))), ("pk4-n3000-G40", 2, 3000, 4, 0.7, rng -> symmetric(40, randomgraph(rng, 40, 0.3))))
        X, nv, rng = pk(seed, n, k, p); push!(L, (name, "big", X, graphsupport(X), Yf(rng), Boolean(), nothing))
    end
    # weighted semirings on a big target
    for R in (Lawvere(), Probability())
        X, nv, rng = pk(3, 3000, 3, 0.8); Y = symmetric(30, randomgraph(rng, 30, 0.3))
        nyv = 30
        W = R isa Lawvere ? Any[[round(10rand(rng), digits=2) for i in 1:nyv, j in 1:nv], nothing] : Any[[(0.5 + rand(rng)) / nyv for i in 1:nyv, j in 1:nv], nothing]
        push!(L, ("pk3-n3000-G30-$(nameof(typeof(R)))", "big", X, graphsupport(X), Y, R, W))
    end
    # deep
    for (w, l, q) in ((4, 100000, 3), (6, 1000, 3), (5, 1000, 4), (6, 4000, 3))
        nv, E = grid(w, l); X = graphinstance(nv, E); push!(L, ("grid$(w)x$l-K$q", "deep", X, graphsupport(X), complete(q), Boolean(), nothing))
    end
    for n in (10000, 30000, 100000)
        c, = miter(n, adder!, (c, a, b) -> buggy!(c, a, b, n ÷ 2)); X = instance(c); push!(L, ("miter-bug-$n", "deep", X, support(X), B, Boolean(), nothing))
    end
    c, = miter(30000, adder!, adder2!); X = instance(c); push!(L, ("miter-eq-30000", "deep", X, support(X), B, Boolean(), nothing))
    # small
    X, nv, rng = pk(4, 500, 3, 0.8); push!(L, ("pk3-n500-K4", "small", X, graphsupport(X), complete(4), Boolean(), nothing))
    nv, E = grid(4, 200); X = graphinstance(nv, E); push!(L, ("grid4x200-K3", "small", X, graphsupport(X), complete(3), Boolean(), nothing))
    c, = miter(1000, adder!, (c, a, b) -> buggy!(c, a, b, 500)); X = instance(c); push!(L, ("miter-bug-1000", "small", X, support(X), B, Boolean(), nothing))
    return L
end
tcounts = length(ARGS) >= 3 ? parse.(Int, split(ARGS[3], ",")) : [Threads.nthreads()]
for (name, kind, X, S, Y, R, W) in workloads()
    !isempty(ONLY) && !occursin(ONLY, name * " " * kind) && continue
    D = decomposition(X, S; alg=MF())
    td = minimum(@elapsed(decomposition(X, S; alg=MF())) for _ in 1:2)
    if MODE == "serial"
        t, v = timeit(() -> score(R, W, D, Y))
        emit(name, kind, "serial", 1, r4(td), r4(t), v)
    elseif MODE == "threads"
        for nt in tcounts
            P = CSets.Threaded(nt)
            t, v = timeit(() -> score(P, R, W, D, Y))
            emit(name, kind, "threads", nt, r4(td), r4(t), v)
        end
    else
        t, v = timeit(() -> CSetsGPU.score(R, W, D, Y))
        emit(name, kind, "gpu", 0, r4(td), r4(t), v)
    end
end
# many targets: one X, 24 targets
rng = MersenneTwister(4)
nv, E = partialktree(rng, 20000, 3, 0.8); X = graphinstance(nv, E); D = decomposition(X, graphsupport(X); alg=MF())
Ys = [begin ny = rand(rng, 4:9); symmetric(ny, randomgraph(rng, ny, rand(rng, 0.4:0.1:0.8))) end for _ in 1:24]
Ws = Any[nothing for _ in Ys]
if isempty(ONLY) || occursin(ONLY, "many")
    if MODE == "serial"
        t, v = timeit(() -> [score(Boolean(), nothing, D, Y) for Y in Ys])
        emit("many24-pk3-n20000", "many", "serial", 1, 0, r4(t), count(v))
    elseif MODE == "threads"
        for nt in tcounts
            P = CSets.Threaded(nt)
            t, v = timeit(() -> score(P, Boolean(), Ws, D, Ys))
            emit("many24-pk3-n20000", "many", "threads", nt, 0, r4(t), count(v))
        end
    else
        t, v = timeit(() -> CSetsGPU.score(Boolean(), Ws, D, Ys))
        emit("many24-pk3-n20000", "many", "gpu", 0, 0, r4(t), count(v))
    end
end
