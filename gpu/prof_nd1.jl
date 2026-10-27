include("workloads.jl"); include("stats.jl")
using Metis; import AMD as AMDpkg
using CliqueTrees: MF, MMD, ND, METISND, AMD
B = instance(boolean())
n = parse(Int, ARGS[1])
c, = miter(n, adder!, (c, a, b) -> buggy!(c, a, b, n ÷ 2)); X = instance(c); S = support(X)
for (nm, alg) in (("AMD", AMD()), ("ND(MMD,METIS)", ND(MMD(), METISND())))
    t0 = time(); D = decomposition(X, S; alg); td = time() - t0
    sh = shape(D); println("n=$n $nm decomp(first)=", round(td, digits=3), " maxbag=", sh.maxbag, " maxsep=", sh.maxsep, " depth=", sh.depth); flush(stdout)
    t0 = time(); color(D, B); println("   color(first)=", round(time() - t0, digits=3)); flush(stdout)
end
