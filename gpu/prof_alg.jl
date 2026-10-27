include("workloads.jl"); include("stats.jl")
using CliqueTrees: MF, AMF, MMD, MCS, RCMMD, BFS, AMD
B = instance(boolean())
for n in (10000, 30000, 100000)
    c, = miter(n, adder!, (c, a, b) -> buggy!(c, a, b, n ÷ 2)); X = instance(c); S = support(X)
    nv = S.nv
    for (nm, alg) in (("MF", MF()), ("AMF", AMF()), ("MMD", MMD()), ("MCS", MCS()), ("RCMMD", RCMMD()))
        try
            D = decomposition(X, S; alg); td = minimum(@elapsed(decomposition(X, S; alg)) for _ in 1:3)
            sh = shape(D); color(D, B); tc = minimum(@elapsed(color(D, B)) for _ in 1:3)
            println("n=$n $nm decomp=", round(td, digits=4), " color=", round(tc, digits=4), " maxbag=", sh.maxbag, " depth=", sh.depth)
        catch e
            println("n=$n $nm error ", sprint(showerror, e)[1:min(100, end)])
        end
    end
end
