include("workloads.jl"); include("stats.jl"); include("contract.jl")
function run(name, X, S, Y)
    D = decomposition(X, S; alg=MF())
    t = @elapsed hd, st = contract(D, Y)
    hc = color(D, Y)
    ok = isnothing(hc) == isnothing(hd) && (isnothing(hd) || ishom(X, Y, hd))
    println(rpad(name, 26), " levels=", maximum(depths(D)) + 1, " ", st, " ref-time=", round(t, digits=2), " ok=", ok); flush(stdout)
end
B = instance(boolean())
for n in (1000, 10000)
    c, = miter(n, adder!, adder2!); X = instance(c); run("adder≡adder2 n=$n", X, support(X), B)
    c, = miter(n, adder!, (c, a, b) -> buggy!(c, a, b, n ÷ 2)); X = instance(c); run("adder vs buggy n=$n", X, support(X), B)
end
for (w, l) in ((4, 2000), (6, 1000))
    nv, E = grid(w, l); X = graphinstance(nv, E); run("grid $w×$l →K3", X, graphsupport(X), complete(3))
end
