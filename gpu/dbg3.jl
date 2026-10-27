include("workloads.jl"); include("stats.jl"); include("CSetsGPU.jl"); using .CSetsGPU
B = instance(boolean())
for n in (4, 8), bud in (4, 64, 1<<14), nit in (1, 64, 1<<15)
    c, = miter(n, adder!, (c, a, b) -> buggy!(c, a, b, max(2, n ÷ 2))); X = instance(c)
    D = decomposition(X, support(X); alg=MF())
    hc = color(D, B); hg = CSetsGPU.color(D, B; nitems=nit, minwork=0, budget=bud)
    println("n=$n bud=$bud nit=$nit cpu=", isnothing(hc), " gpu=", isnothing(hg))
end
