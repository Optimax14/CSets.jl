include("workloads.jl"); include("stats.jl"); include("CSetsGPU.jl"); using .CSetsGPU
using CliqueTrees: MMD
B = instance(boolean())
for n in (10000, 30000, 100000)
    for (nm, f) in (("eq", adder2!), ("bug", (c, a, b) -> buggy!(c, a, b, n ÷ 2)))
        c, = miter(n, adder!, f); X = instance(c); S = support(X)
        D = decomposition(X, S; alg=MMD()); td = minimum(@elapsed(decomposition(X, S; alg=MMD())) for _ in 1:3)
        hc = color(D, B); tc = minimum(@elapsed(color(D, B)) for _ in 1:2)
        hg = CSetsGPU.contract(D, B); hg === :wide && (println("wide"); continue)
        tg = minimum(@elapsed(CSetsGPU.contract(D, B)) for _ in 1:3)
        ok = isnothing(hc) == isnothing(hg) && (isnothing(hg) || ishom(X, B, hg))
        println("n=$n $nm MMD decomp=", round(td, digits=4), " cpu=", round(tc, digits=4), " gpu-contract=", round(tg, digits=4),
            " search x", round(tc / tg, digits=1), " e2e x", round((td + tc) / (td + tg), digits=2), " ok=", ok); flush(stdout)
    end
end
