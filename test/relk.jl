# The relk kernel and kernel = :auto against the default kernel.
# Run with threads, e.g. julia -t 4 --project=test test/relk.jl
# Small instances: 300 trials x 4 semirings, the scores must be equal
# (== for every semiring; Boolean scores are Bool). Larger instances:
# 40 trials x 3 semirings, which exercise the parallel paths.
include(joinpath(@__DIR__, "graphs.jl"))
using CSets: score, Boolean, Lawvere, Probability, Eager, Lazy
using Random

rgraph(rng, n, m) = [(rand(rng, 1:n), rand(rng, 1:n)) for _ in 1:m]
const NT = max(2, Threads.nthreads())
nok = 0; nbad = 0

function check(R, W, D, Y, tag, exact)
    a = score(R, W, D, Y)                                       # default kernel (and strategy)
    vals = Any[a]
    R isa Boolean && push!(vals, score(R, W, D, Y; strategy=Eager))
    push!(vals, score(R, W, D, Y; kernel=:relk))
    push!(vals, score(R, W, D, Y; kernel=:relk, nthreads=NT))
    push!(vals, score(R, W, D, Y; kernel=:auto, strategy=Eager))
    push!(vals, score(R, W, D, Y; kernel=:auto, strategy=Eager, nthreads=NT))
    ok = exact ? all(v -> v === a || v == a, vals) && all(v -> typeof(v) == typeof(a), vals) : all(v -> isapprox(v, a; rtol=1e-12), vals)
    if ok
        global nok += 1
    else
        global nbad += 1
        nbad < 10 && println("MISMATCH $tag $R: ", vals)
    end
end

rng = MersenneTwister(1)
for trial in 1:300
    nY = rand(rng, 1:12); mY = rand(rng, 0:30)
    Y = graph_instance(nY, rgraph(rng, nY, mY))
    k = rand(rng, 1:6); mX = rand(rng, 0:9)
    X = graph_instance(k, rgraph(rng, k, mX))
    float(mY + 1)^mX * float(nY)^k > 1e8 && continue
    D = CSets.decomposition(X, graph_support(X); alg=MF())
    for (R, W) in [(Probability(), nothing), (Boolean(), nothing),
                   (Probability(), [rand(rng, 0:2, nY, k) .* 1.0, nothing]),
                   (Lawvere(), [ifelse.(rand(rng, nY, k) .< 0.2, Inf, rand(rng, 0:5, nY, k) .* 1.0), rand(rng, 0:3, mY, mX) .* 1.0])]
        check(R, W, D, Y, "small $trial", true)
    end
end
println("small: ok $nok bad $nbad"); n1 = nok
rng = MersenneTwister(2)
for trial in 1:40
    nY = rand(rng, 50:400); mY = rand(rng, 100:2000)
    Y = graph_instance(nY, rgraph(rng, nY, mY))
    k = rand(rng, 2:5); mX = rand(rng, 1:6)
    X = graph_instance(k, rgraph(rng, k, mX))
    D = CSets.decomposition(X, graph_support(X); alg=MF())
    for (R, W) in [(Probability(), nothing), (Boolean(), nothing),
                   (Probability(), [rand(rng, 0:2, nY, k) .* 1.0, nothing])]
        # integer-valued weights: the sums are exact in Float64 below 2^53
        check(R, W, D, Y, "large $trial", true)
    end
end
println("large: ok $(nok - n1) bad $nbad")
# witnesses still use the default kernel
X = graph_instance(3, [(1, 2), (2, 3), (3, 1)]); Y = graph_instance(4, [(i, j) for i in 1:4 for j in 1:4 if i != j])
D = CSets.decomposition(X, graph_support(X); alg=MF())
@assert !isnothing(CSets.color(D, Y))
@assert score(Probability(), nothing, D, Y; kernel=:relk) == 24.0
try
    score(Probability(), nothing, D, Y; kernel=:fast); global nbad += 1
catch e
    e isa ArgumentError || (global nbad += 1)
end
# the :auto path with thresholds that select relk at every bag, and never
RK = CSets.RelKernel
for (tau, ratio) in ((0.0, 0.0), (Inf, Inf))
    RK.TAU[] = tau; RK.RATIO[] = ratio
    rng = MersenneTwister(3); n0 = nok
    for trial in 1:150
        nY = rand(rng, 1:12); mY = rand(rng, 0:30)
        Y = graph_instance(nY, rgraph(rng, nY, mY))
        k = rand(rng, 1:6); mX = rand(rng, 0:9)
        X = graph_instance(k, rgraph(rng, k, mX))
        float(mY + 1)^mX * float(nY)^k > 1e8 && continue
        D = CSets.decomposition(X, graph_support(X); alg=MF())
        check(Probability(), nothing, D, Y, "auto tau=$tau $trial", true)
        check(Lawvere(), [ifelse.(rand(rng, nY, k) .< 0.2, Inf, rand(rng, 0:5, nY, k) .* 1.0), nothing], D, Y, "auto tau=$tau $trial", true)
    end
    println("auto tau=$tau ratio=$ratio: ok $(nok - n0) bad $nbad")
end
RK.TAU[] = 1e6; RK.RATIO[] = 8.0
println(nbad == 0 ? "ALL OK $nok" : "FAILED $nbad")
