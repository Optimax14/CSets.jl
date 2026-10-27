# usage: julia hpg_bench2.jl wide|e2e|multi|big out.tsv
include("workloads.jl"); include("stats.jl"); include("CSetsGPU.jl"); using .CSetsGPU
using CliqueTrees: MMD
const SUITE = ARGS[1]; const OUT = ARGS[2]
B = instance(boolean())
emit(xs...) = (line = join(xs, '\t'); println(line); flush(stdout); open(io -> println(io, line), OUT, "a"))
r4(x) = round(x, digits=4)
function best(f, reps)
    t = Inf; st = nothing
    for _ in 1:reps
        s1 = Dict{Symbol,Any}(); t1 = @elapsed f(s1); t1 < t && (t = t1; st = s1)
    end
    return t, st
end
function okcheck(X, Y, hc, hg)
    return isnothing(hc) == isnothing(hg) && (isnothing(hg) || ishom(X, Y, hg))
end
if SUITE == "wide"
    cases = Any[]
    for (w, l, q) in ((4, 100000, 3), (5, 2000, 3), (6, 1000, 3), (5, 1000, 4), (6, 4000, 3), (7, 1000, 3))
        nv, E = grid(w, l); X = graphinstance(nv, E); push!(cases, ("grid$(w)x$l-K$q", X, graphsupport(X), complete(q)))
    end
    for (name, X, S, Y) in cases
        D = decomposition(X, S; alg=MF()); hc = color(D, Y); tc = minimum(@elapsed(color(D, Y)) for _ in 1:2)
        C = CSetsGPU.prepare_contract(D)
        for jump in (:mate, :wyllie)
            hg = CSetsGPU.contract(C, Y; jump)
            if hg === :wide; emit(name, jump, r4(tc), "wide"); continue; end
            tg, st = best(s -> CSetsGPU.contract(C, Y; jump, stats=s), 2)
            emit(name, jump, r4(tc), r4(tg), r4(tc / tg), okcheck(X, Y, hc, hg), "local=$(r4(st[:local])) contract=$(r4(st[:contract])) jumps=$(r4(st[:jumptime])) witness=$(r4(st[:witness])) maxL=$(st[:maxL]) MB=$(st[:mbytes] ÷ 2^20) rounds=$(st[:jumps])")
        end
    end
elseif SUITE == "e2e"
    for n in parse.(Int, split(get(ENV, "NS", "10000,30000,100000"), ",")), (alg, an) in ((MF(), "MF"), (MMD(), "MMD"))
        for (kind, f) in (("eq", adder2!), ("bug", (c, a, b) -> buggy!(c, a, b, n ÷ 2)))
            c, = miter(n, adder!, f); X = instance(c); S = support(X)
            D = decomposition(X, S; alg); td = minimum(@elapsed(decomposition(X, S; alg)) for _ in 1:3)
            hc = color(D, B); tc = minimum(@elapsed(color(D, B)) for _ in 1:2)
            tp = minimum(@elapsed(CSetsGPU.prepare_contract(D)) for _ in 1:2)
            C = CSetsGPU.prepare_contract(D)
            hg = CSetsGPU.contract(C, B)
            tg, st = best(s -> CSetsGPU.contract(C, B; stats=s), 3)
            emit("miter-$kind-$n", an, r4(td), r4(tc), r4(tp), r4(tg), "search x$(r4(tc / tg))", "e2e(uncached) x$(r4((td + tc) / (td + tp + tg)))", "e2e(cached prep) x$(r4((td + tc) / (td + tg)))", okcheck(X, B, hc, hg))
        end
    end
elseif SUITE == "multi"
    # same X, many Y: cached preparation
    rng = MersenneTwister(4)
    for (name, nv, E) in (("pk3-n20000", partialktree(rng, 20000, 3, 0.8)...), ("grid4x20000", grid(4, 20000)...))
        X = graphinstance(nv, E); S = graphsupport(X); D = decomposition(X, S; alg=MF())
        Ys = [begin ny = rand(rng, 4:9); symmetric(ny, randomgraph(rng, ny, rand(rng, 0.4:0.1:0.8))) end for _ in 1:24]
        hcs = [color(D, Y) for Y in Ys]
        tc = @elapsed for Y in Ys; color(D, Y); end
        P = CSetsGPU.prepare(D); C = CSetsGPU.prepare_contract(D)
        CSetsGPU.color(P, Ys[1]); CSetsGPU.contract(C, Ys[1])
        tl = @elapsed hl = [CSetsGPU.color(P, Y) for Y in Ys]
        tlu = @elapsed for Y in Ys; CSetsGPU.color(D, Y); end
        tk = @elapsed hk = [CSetsGPU.contract(C, Y) for Y in Ys]
        tku = @elapsed for Y in Ys; CSetsGPU.contract(D, Y); end
        ok = all(okcheck(X, Ys[i], hcs[i], hl[i]) for i in eachindex(Ys)) && all(hk[i] === :wide || okcheck(X, Ys[i], hcs[i], hk[i]) for i in eachindex(Ys))
        nsat = count(!isnothing, hcs)
        emit(name, "24 targets ($nsat sat)", "cpu=$(r4(tc))", "levelsync cached=$(r4(tl)) uncached=$(r4(tlu))", "contract cached=$(r4(tk)) uncached=$(r4(tku))", "x(best cached)=$(r4(tc / min(tl, tk)))", ok)
    end
else # big: level-synchronous with resplit
    function pk(seed, n, k, p, Yf)
        rng = MersenneTwister(seed); nv, E = partialktree(rng, n, k, p; bushy=true); X = graphinstance(nv, E)
        return X, graphsupport(X), Yf(rng)
    end
    randY(ny, py) = rng -> symmetric(ny, randomgraph(rng, ny, py))
    cpu = Dict("pk4-n20000-K5" => 1.2135, "pk6-n5000-K7" => 22.1178, "pk5-n20000-K6" => 7.7721, "pk3-n5000-G30" => 13.1786, "pk4-n1000-G40" => 47.1682, "pk4-n3000-G40" => 120.5021, "pk5-n1000-G60" => 11.6724)
    cases = Any[]
    for (n, k, q, p) in ((20000, 4, 5, 0.7), (5000, 6, 7, 0.6), (20000, 5, 6, 0.6))
        push!(cases, ("pk$k-n$n-K$q", pk(1, n, k, p, _ -> complete(q))...))
    end
    for (n, k, p, ny, py) in ((5000, 3, 0.8, 30, 0.3), (1000, 4, 0.7, 40, 0.3), (3000, 4, 0.7, 40, 0.3), (1000, 5, 0.7, 60, 0.25))
        push!(cases, ("pk$k-n$n-G$ny", pk(2, n, k, p, randY(ny, py))...))
    end
    for (name, X, S, Y) in cases
        D = decomposition(X, S; alg=MF()); P = CSetsGPU.prepare(D)
        for bud in (0, 1 << 14)
            hg = CSetsGPU.color(P, Y; budget=bud)
            tg, st = best(s -> CSetsGPU.color(P, Y; budget=bud, stats=s), 2)
            emit(name, "budget=$bud", get(cpu, name, NaN), r4(tg), r4(get(cpu, name, NaN) / tg), isnothing(hg) || ishom(X, Y, hg), "sat=$(!isnothing(hg)) rounds=$(st[:rounds]) kern=$(r4(get(st, :kernel, 0.0)))")
        end
    end
end
