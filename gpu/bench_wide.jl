include("workloads.jl"); include("stats.jl"); include("CSetsGPU.jl"); using .CSetsGPU
using CliqueTrees: MMD
function run(name, X, S, Y; reps=2, alg=MF())
    D = decomposition(X, S; alg)
    td = minimum(@elapsed(decomposition(X, S; alg)) for _ in 1:reps)
    hc = color(D, Y); tc = minimum(@elapsed(color(D, Y)) for _ in 1:reps)
    C = CSetsGPU.prepare_contract(D)
    tp = @elapsed CSetsGPU.prepare_contract(D)
    hg = CSetsGPU.contract(C, Y)
    hg === :wide && (println(rpad(name, 24), " cpu=", round(tc, digits=4), " wide (memory)"); return)
    st = Dict{Symbol,Any}(); tg = Inf
    for _ in 1:reps
        s1 = Dict{Symbol,Any}(); t = @elapsed CSetsGPU.contract(C, Y; stats=s1); t < tg && (tg = t; st = s1)
    end
    ok = isnothing(hc) == isnothing(hg) && (isnothing(hg) || ishom(X, Y, hg))
    println(rpad(name, 24), " decomp=", round(td, digits=4), " cpu=", round(tc, digits=4), " prep=", round(tp, digits=4), " gpu=", round(tg, digits=4),
        " (local ", round(get(st, :local, 0.0), digits=4), " contract ", round(get(st, :contract, 0.0), digits=4), " [jumps ", round(get(st, :jumptime, 0.0), digits=4), "] witness ", round(get(st, :witness, 0.0), digits=4),
        ") x", round(tc / tg, digits=1), " maxL=", get(st, :maxL, 0), " MB=", get(st, :mbytes, 0) ÷ 2^20, " jumps=", get(st, :jumps, 0), " ok=", ok); flush(stdout)
end
B = instance(boolean())
which = isempty(ARGS) ? "all" : ARGS[1]
if which in ("all", "circ")
for n in (10000, 30000)
    c, = miter(n, adder!, (c, a, b) -> buggy!(c, a, b, n ÷ 2)); X = instance(c); run("miter-bug $n", X, support(X), B)
end
end
if which in ("all", "grid")
for (w, l, q) in ((4, 100000, 3), (5, 2000, 3), (6, 1000, 3), (6, 4000, 3), (7, 1000, 3), (5, 1000, 4))
    nv, E = grid(w, l); X = graphinstance(nv, E); run("grid $w×$l →K$q", X, graphsupport(X), complete(q))
end
end
