include("workloads.jl"); include("stats.jl")
using Metis
using CliqueTrees: MF, MMD, ND, METISND, AMD
B = instance(boolean())
for n in (10000, 30000)
    c, = miter(n, adder!, (c, a, b) -> buggy!(c, a, b, n ÷ 2)); X = instance(c); S = support(X)
    for (nm, alg) in (("MF", MF()), ("MMD", MMD()), ("AMD", AMD()), ("ND(MMD,METIS)", ND(MMD(), METISND())), ("ND(MF,METIS)", ND(MF(), METISND())))
        D = decomposition(X, S; alg); td = minimum(@elapsed(decomposition(X, S; alg)) for _ in 1:3)
        sh = shape(D); color(D, B); tc = minimum(@elapsed(color(D, B)) for _ in 1:2)
        println("n=$n $nm decomp=", round(td, digits=4), " color=", round(tc, digits=4), " maxbag=", sh.maxbag, " maxsep=", sh.maxsep, " depth=", sh.depth, " nodes=", sh.nt); flush(stdout)
    end
end
