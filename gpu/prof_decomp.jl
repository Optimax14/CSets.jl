include("workloads.jl")
using CSets: sccs, permute, gaifman, invspan, separators, invmap
using CliqueTrees: BipartiteGraph, cliquetree, MF, AMF, MMD
function decompparts(X, S; alg=MF())
    T = Dict{String,Float64}()
    t = time(); nu = X.dom.nu; C = sccs(X.dom); scc = [C.adjncy[nu + 1 - u] for u in 1:nu]; T["sccs"] = time() - t
    t = time(); X2 = permute(X, scc); S2 = permute(S, scc); T["permute"] = time() - t
    t = time(); G = gaifman(S2); T["gaifman"] = time() - t
    graph = BipartiteGraph(G.nv, G.nu, G.ne, G.xadj, G.adjncy)
    t = time(); perm, clique = cliquetree(graph; alg); T["cliquetree"] = time() - t
    t = time(); D = decomposition(X, S; alg); T["total"] = time() - t
    T["rest(separators etc)"] = T["total"] - T["sccs"] - T["permute"] - T["gaifman"] - T["cliquetree"]
    return T, G
end
B = instance(boolean())
for n in (10000, 30000, 100000)
    c, = miter(n, adder!, (c, a, b) -> buggy!(c, a, b, n ÷ 2)); X = instance(c); S = support(X)
    decompparts(X, S)
    T, G = decompparts(X, S)
    for _ in 1:4
        T2, _ = decompparts(X, S)
        for k in keys(T); T[k] = min(T[k], T2[k]); end
    end
    println("miter n=$n  |V(gaifman)|=", G.nv, " |E|=", G.ne, "  ", join(["$k=$(round(v, digits=4))" for (k, v) in sort(collect(T))], " "))
    for alg in (AMF(), MMD())
        graph = BipartiteGraph(G.nv, G.nu, G.ne, G.xadj, G.adjncy)
        cliquetree(graph; alg); t = @elapsed cliquetree(graph; alg)
        D = decomposition(X, S; alg); tc = @elapsed color(D, B)
        println("   alg=$alg cliquetree=", round(t, digits=4), " color=", round(tc, digits=4))
    end
end
