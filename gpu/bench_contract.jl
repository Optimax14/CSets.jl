include("workloads.jl"); include("stats.jl"); include("CSetsGPU.jl"); using .CSetsGPU
function run(name, X, S, Y; reps=3)
    td = minimum(@elapsed(decomposition(X, S; alg=MF())) for _ in 1:reps)
    D = decomposition(X, S; alg=MF())
    hc = color(D, Y)
    tc = minimum(@elapsed(color(D, Y)) for _ in 1:reps)
    hd = CSetsGPU.contract(D, Y)
    if hd === :wide
        println(rpad(name, 26), " cpu=", round(tc, digits=4), " contraction: separator domain > 64"); return
    end
    st = Dict{Symbol,Any}(); tg = Inf
    for _ in 1:reps
        s1 = Dict{Symbol,Any}()
        t = @elapsed CSetsGPU.contract(D, Y; stats=s1)
        t < tg && (tg = t; st = s1)
    end
    ok = isnothing(hc) == isnothing(hd) && (isnothing(hd) || ishom(X, Y, hd))
    println(rpad(name, 26), " decomp=", round(td, digits=4), " cpu=", round(tc, digits=4), " gpu-contract=", round(tg, digits=4),
        " (local ", round(get(st, :local, 0.0), digits=4), " [plan ", round(get(st, :lplan, 0.0), digits=4), " setup ", round(get(st, :lsetup, 0.0), digits=4), " fwd ", round(get(st, :lfwd, 0.0), digits=4), " tree ", round(get(st, :ltree, 0.0), digits=4), "] contract ", round(get(st, :contract, 0.0), digits=4), " witness ", round(get(st, :witness, 0.0), digits=4),
        ") x", round(tc / tg, digits=1), " end-to-end x", round((td + tc) / (td + tg), digits=2), " levels=", maximum(depths(D)) + 1,
        " phases=", get(st, :phases, 0), " jumps=", get(st, :jumps, 0), " maxL=", get(st, :maxL, 0), " ok=", ok); flush(stdout)
end
B = instance(boolean())
for n in (1000, 10000, 30000)
    c, = miter(n, adder!, adder2!); X = instance(c); run("adder≡adder2 n=$n", X, support(X), B)
    c, = miter(n, adder!, (c, a, b) -> buggy!(c, a, b, n ÷ 2)); X = instance(c); run("adder vs buggy n=$n", X, support(X), B)
end
for (w, l) in ((4, 2000), (4, 20000), (6, 1000))
    nv, E = grid(w, l); X = graphinstance(nv, E); run("grid $w×$l →K3", X, graphsupport(X), complete(3))
end
rng = MersenneTwister(1)
nv, E = partialktree(rng, 20000, 3, 0.8; bushy=true); X = graphinstance(nv, E); run("pk3 n=20000 →K4 bushy", X, graphsupport(X), complete(4))
nv, E = partialktree(rng, 20000, 2, 0.9; bushy=false); X = graphinstance(nv, E); run("pk2 n=20000 →K3 deep", X, graphsupport(X), complete(3))
