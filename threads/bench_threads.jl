include("workloads.jl")
using CSets: CSets, score, Boolean, Lawvere, Probability, sprod, sone, Threaded
include("rwutil.jl")
B = instance(boolean())
function run(name, X, S, Y, R; W=nothing)
    D = decomposition(X, S; alg=MF())
    score(R, W, D, Y); ts = minimum(@elapsed(score(R, W, D, Y)) for _ in 1:2)
    out = "$(rpad(name, 26)) $R serial=$(round(ts, digits=3))"
    for nt in (1, 4, 8, 16)
        P = Threaded(nt); score(P, R, W, D, Y)
        t = minimum(@elapsed(score(P, R, W, D, Y)) for _ in 1:2)
        out *= " t$nt=$(round(t, digits=3))"
    end
    println(out); flush(stdout)
end
rng = MersenneTwister(1)
nv, E = partialktree(rng, 2000, 3, 0.8); X = graphinstance(nv, E); Y = symmetric(30, randomgraph(rng, 30, 0.3))
for R in (Boolean(), Lawvere(), Probability()); run("pk3-n2000-G30", X, graphsupport(X), Y, R; W=randweights(rng, R, Y, nv)); end
nv, E = partialktree(rng, 2000, 6, 0.6); X = graphinstance(nv, E); run("pk6-n2000-K7", X, graphsupport(X), complete(7), Boolean())
nv, E = partialktree(rng, 20000, 4, 0.7); X = graphinstance(nv, E); run("pk4-n20000-K5", X, graphsupport(X), complete(5), Boolean())
c, = miter(10000, adder!, (c, a, b) -> buggy!(c, a, b, 5000)); X = instance(c); run("miter-bug-10k", X, support(X), B, Boolean())
nv, E = grid(6, 1000); X = graphinstance(nv, E); run("grid6x1000-K3", X, graphsupport(X), complete(3), Boolean())
