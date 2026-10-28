include("workloads.jl")
using CSets: CSets, score, color, Boolean, Threaded
B = instance(boolean())
function run(name, X, S, Y)
    D = decomposition(X, S; alg=MF())
    score(Boolean(), nothing, D, Y); ts = minimum(@elapsed(score(Boolean(), nothing, D, Y)) for _ in 1:2)
    out = "$(rpad(name, 18)) serial=$(round(ts, digits=3))"
    for nt in (1, 4, 8, 16)
        P = Threaded(nt; deep=1); score(P, Boolean(), nothing, D, Y)
        t = minimum(@elapsed(score(P, Boolean(), nothing, D, Y)) for _ in 1:2)
        out *= " c$nt=$(round(t, digits=3))"
    end
    P = Threaded(16); t = minimum(@elapsed(score(P, Boolean(), nothing, D, Y)) for _ in 1:2)
    println(out, " levelsync16=", round(t, digits=3)); flush(stdout)
end
c, = miter(10000, adder!, (c, a, b) -> buggy!(c, a, b, 5000)); X = instance(c); run("miter-bug-10k", X, support(X), B)
c, = miter(30000, adder!, (c, a, b) -> buggy!(c, a, b, 15000)); X = instance(c); run("miter-bug-30k", X, support(X), B)
nv, E = grid(4, 20000); X = graphinstance(nv, E); run("grid4x20000-K3", X, graphsupport(X), complete(3))
nv, E = grid(6, 1000); X = graphinstance(nv, E); run("grid6x1000-K3", X, graphsupport(X), complete(3))
