# usage: julia [-t N] hpg_cmp.jl gpu|serial|threads out.tsv [threadcounts]
#
# The threads/hpgcmp.jl workloads (same names and instances, so the
# itay/threads CPU numbers line up) plus a size sweep for the :auto
# thresholds: tiny instances, Lawvere / Probability, deep Lawvere.
# gpu mode times every method (:auto, :levels, :contract) and records
# the decomposition features and the level-synchronous kernel stats.
# CPU modes run with the itay/threads project (CSets.Threaded).
#
const MODE = ARGS[1]; const OUT = ARGS[2]
include(joinpath(@__DIR__, "workloads.jl"))
include(joinpath(@__DIR__, "rwutil.jl"))
using CSets: CSets, score, Boolean, Lawvere, Probability, sprod, sone
if MODE == "gpu"
    include(joinpath(@__DIR__, "CSetsGPU.jl")); using .CSetsGPU
end
const ONLY = get(ENV, "ONLY", "")
r4(x) = round(x, digits=4)
emit(xs...) = (line = join(xs, '\t'); println(line); flush(stdout); open(io -> println(io, line), OUT, "a"))
function timeit(f; reps=2, cap=60.0)
    t1 = @elapsed v = f()
    t1 > cap && return t1, v
    t = t1
    for _ in 2:(t1 < 1 ? reps + 1 : reps)
        t = min(t, @elapsed f())
    end
    return t, v
end
function features(D, Y)
    nt = D.tree.nu
    hgt = zeros(Int, nt)
    for t in 1:nt, j in D.tree.xadj[t]:D.tree.xadj[t + 1] - 1
        hgt[t] = max(hgt[t], hgt[D.tree.adjncy[j]] + 1)
    end
    vx = collect(D.inst.vtx); ny = [count(==(u), collect(Y.vtx)) for u in 1:Y.cod.nu]
    lg(x) = log2(max(ny[vx[x]], 1))
    lsep = 0.0; lbag = 0.0; maxbag = 0; maxsep = 0
    for t in 1:nt
        s = sum((lg(D.sep.adjncy[p]) for p in D.sep.xadj[t]:D.sep.xadj[t + 1] - 1); init=0.0)
        r = sum((lg(D.res.adjncy[p]) for p in D.res.xadj[t]:D.res.xadj[t + 1] - 1); init=0.0)
        lsep = max(lsep, s); lbag = max(lbag, s + r)
        ns = D.sep.xadj[t + 1] - D.sep.xadj[t]; nr = D.res.xadj[t + 1] - D.res.xadj[t]
        maxsep = max(maxsep, ns); maxbag = max(maxbag, ns + nr)
    end
    return (; height=nt > 0 ? hgt[nt] : 0, nt, nx=D.inst.dom.nu, ny=Y.dom.nu, maxbag, maxsep, lsep=round(lsep, digits=1), lbag=round(lbag, digits=1))
end
fstr(f) = join(("$k=$(getfield(f, k))" for k in keys(f)), " ")
function pk(seed, n, k, p)
    rng = MersenneTwister(seed)
    nv, E = partialktree(rng, n, k, p; bushy=true)
    return graphinstance(nv, E), nv, rng
end
function wts(R, rng, nyv, nv)
    R isa Boolean && return nothing
    return R isa Lawvere ? Any[[round(10rand(rng), digits=2) for i in 1:nyv, j in 1:nv], nothing] : Any[[(0.5 + rand(rng)) / nyv for i in 1:nyv, j in 1:nv], nothing]
end
function workloads()
    L = Any[]
    B = instance(boolean())
    # --- threads/hpgcmp.jl, unchanged ---
    for (name, seed, n, k, p, Yf) in (("pk4-n20000-K5", 1, 20000, 4, 0.7, _ -> complete(5)), ("pk5-n20000-K6", 1, 20000, 5, 0.6, _ -> complete(6)),
            ("pk6-n5000-K7", 1, 5000, 6, 0.6, _ -> complete(7)), ("pk3-n5000-G30", 2, 5000, 3, 0.8, rng -> symmetric(30, randomgraph(rng, 30, 0.3))),
            ("pk4-n1000-G40", 2, 1000, 4, 0.7, rng -> symmetric(40, randomgraph(rng, 40, 0.3))), ("pk4-n3000-G40", 2, 3000, 4, 0.7, rng -> symmetric(40, randomgraph(rng, 40, 0.3))))
        X, nv, rng = pk(seed, n, k, p); push!(L, (name, "big", X, graphsupport(X), Yf(rng), Boolean(), nothing))
    end
    for R in (Lawvere(), Probability())
        X, nv, rng = pk(3, 3000, 3, 0.8); Y = symmetric(30, randomgraph(rng, 30, 0.3))
        push!(L, ("pk3-n3000-G30-$(nameof(typeof(R)))", "big", X, graphsupport(X), Y, R, wts(R, rng, 30, nv)))
    end
    for (w, l, q) in ((4, 100000, 3), (6, 1000, 3), (5, 1000, 4), (6, 4000, 3))
        nv, E = grid(w, l); X = graphinstance(nv, E); push!(L, ("grid$(w)x$l-K$q", "deep", X, graphsupport(X), complete(q), Boolean(), nothing))
    end
    for n in (10000, 30000, 100000)
        c, = miter(n, adder!, (c, a, b) -> buggy!(c, a, b, n ÷ 2)); X = instance(c); push!(L, ("miter-bug-$n", "deep", X, support(X), B, Boolean(), nothing))
    end
    c, = miter(30000, adder!, adder2!); X = instance(c); push!(L, ("miter-eq-30000", "deep", X, support(X), B, Boolean(), nothing))
    X, nv, rng = pk(4, 500, 3, 0.8); push!(L, ("pk3-n500-K4", "small", X, graphsupport(X), complete(4), Boolean(), nothing))
    nv, E = grid(4, 200); X = graphinstance(nv, E); push!(L, ("grid4x200-K3", "small", X, graphsupport(X), complete(3), Boolean(), nothing))
    c, = miter(1000, adder!, (c, a, b) -> buggy!(c, a, b, 500)); X = instance(c); push!(L, ("miter-bug-1000", "small", X, support(X), B, Boolean(), nothing))
    # --- sweep for the :auto thresholds ---
    for n in (200, 2000, 5000)
        X, nv, rng = pk(4, n, 3, 0.8); push!(L, ("pk3-n$n-K4", "sweep", X, graphsupport(X), complete(4), Boolean(), nothing))
    end
    for n in (300, 1000, 3000, 10000), R in (Boolean(), Lawvere(), Probability())
        n == 3000 && !(R isa Boolean) && continue
        X, nv, rng = pk(3, n, 3, 0.8); Y = symmetric(30, randomgraph(rng, 30, 0.3))
        push!(L, ("pk3-n$n-G30-$(nameof(typeof(R)))", "sweep", X, graphsupport(X), Y, R, wts(R, rng, 30, nv)))
    end
    for (n, R) in ((1000, Lawvere()), (1000, Probability()))
        X, nv, rng = pk(2, n, 4, 0.7); Y = symmetric(40, randomgraph(rng, 40, 0.3))
        push!(L, ("pk4-n$n-G40-$(nameof(typeof(R)))", "sweep", X, graphsupport(X), Y, R, wts(R, rng, 40, nv)))
    end
    for l in (50, 1000, 5000, 20000)
        nv, E = grid(4, l); X = graphinstance(nv, E); push!(L, ("grid4x$l-K3", "sweep", X, graphsupport(X), complete(3), Boolean(), nothing))
    end
    for (w, l) in ((4, 20000), (6, 1000))
        nv, E = grid(w, l); X = graphinstance(nv, E); rng = MersenneTwister(5)
        for R in (Lawvere(), Probability())
            push!(L, ("grid$(w)x$l-K3-$(nameof(typeof(R)))", "sweep", X, graphsupport(X), complete(3), R, wts(R, rng, 3, nv)))
        end
    end
    for n in (100, 300, 3000)
        c, = miter(n, adder!, (c, a, b) -> buggy!(c, a, b, n ÷ 2)); X = instance(c); push!(L, ("miter-bug-$n", "sweep", X, support(X), B, Boolean(), nothing))
    end
    return L
end
tcounts = length(ARGS) >= 3 ? parse.(Int, split(ARGS[3], ",")) : [Threads.nthreads()]
if MODE == "gpu"
    # compile every path once on a tiny instance
    let (X0, nv0, rng0) = pk(9, 60, 3, 0.8), Y0 = symmetric(8, randomgraph(rng0, 8, 0.5)), D0 = decomposition(X0, graphsupport(X0); alg=MF())
        for R in (Boolean(), Lawvere(), Probability()), m in (:levels, :contract)
            CSetsGPU.score(R, wts(R, rng0, 8, nv0), D0, Y0; method=m)
        end
    end
end
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
        f = features(D, Y)
        GC.gc()
        for m in (:auto, :levels, :contract)
            if m == :levels && f.height > 50000
                emit(name, kind, "gpu", m, r4(td), "skip", "", fstr(f)); continue
            end
            if m == :contract && f.lsep > 24
                emit(name, kind, "gpu", m, r4(td), "skip", "", fstr(f)); continue
            end
            t, v = timeit(() -> CSetsGPU.score(R, W, D, Y; method=m))
            extra = ""
            if m == :levels
                st = Dict{Symbol,Any}()
                P = CSetsGPU.prepare(D)
                CSetsGPU.color(P, typeof(Y)[Y]; S=R isa Boolean ? nothing : CSetsGPU.algebra(R, CSets.valtype(R, W)), Ws=Any[W], witness=false, stats=st)
                g(k) = get(st, k, -1); extra = "kernel=$(r4(g(:kernel))) kmax=$(r4(g(:kmax))) kmaxlev=$(g(:kmaxlev))/$(g(:levels)) kmaxitems=$(g(:kmaxitems)) rounds=$(g(:rounds)) nkern=$(g(:nkern))"
            end
            emit(name, kind, "gpu", m, r4(td), r4(t), v, fstr(f), extra)
        end
    end
end
# many targets: one X, 24 targets (pk3-n20000 as in hpgcmp.jl, plus the deep grid4x2000)
for (mname, gen) in (("many24-pk3-n20000", rng -> partialktree(rng, 20000, 3, 0.8)), ("many24-grid4x2000", rng -> grid(4, 2000)))
    !isempty(ONLY) && !occursin(ONLY, mname * " many") && continue
    local rng = MersenneTwister(4)
    local nv, E = gen(rng); local X = graphinstance(nv, E); local D = decomposition(X, graphsupport(X); alg=MF())
    Ys = [begin ny = rand(rng, 4:9); symmetric(ny, randomgraph(rng, ny, rand(rng, 0.4:0.1:0.8))) end for _ in 1:24]
    Ws = Any[nothing for _ in Ys]
    if MODE == "serial"
        t, v = timeit(() -> [score(Boolean(), nothing, D, Y) for Y in Ys])
        emit(mname, "many", "serial", 1, 0, r4(t), count(v))
    elseif MODE == "threads"
        for nt in tcounts
            P = CSets.Threaded(nt)
            t, v = timeit(() -> score(P, Boolean(), Ws, D, Ys))
            emit(mname, "many", "threads", nt, 0, r4(t), count(v))
        end
    else
        t, v = timeit(() -> CSetsGPU.score(Boolean(), Ws, D, Ys))
        emit(mname, "many", "gpu", "batched", 0, r4(t), count(v))
        t, v = timeit(() -> [CSetsGPU.score(Boolean(), nothing, D, Y) for Y in Ys])
        emit(mname, "many", "gpu", "auto-each", 0, r4(t), count(v))
    end
end
