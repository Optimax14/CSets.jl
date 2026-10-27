include("workloads.jl")
using CSets: sccs, permute, gaifman, invspan
using CliqueTrees: BipartiteGraph, cliquetree, permutation, MMD, MF
for n in (10000, 30000)
    c, = miter(n, adder!, (c, a, b) -> buggy!(c, a, b, n ÷ 2)); X = instance(c); S = support(X)
    nu = X.dom.nu
    m5(f) = (f(); minimum(@elapsed(f()) for _ in 1:5))
    tsc = m5(() -> sccs(X.dom))
    C = sccs(X.dom); scc = [C.adjncy[nu + 1 - u] for u in 1:nu]
    tpe = m5(() -> (permute(X, scc), permute(S, scc)))
    S2 = permute(S, scc)
    tga = m5(() -> gaifman(S2))
    G = gaifman(S2); graph = BipartiteGraph(G.nv, G.nu, G.ne, G.xadj, G.adjncy)
    for (nm, alg) in (("MF", MF()), ("MMD", MMD()))
        to = m5(() -> permutation(graph; alg))
        perm, _ = permutation(graph; alg)
        tct = m5(() -> cliquetree(graph; alg=perm))
        td = m5(() -> decomposition(X, S; alg))
        println("n=$n $nm decomposition=", round(td, digits=4), " = sccs ", round(tsc, digits=4), " + permute ", round(tpe, digits=4), " + gaifman ", round(tga, digits=4),
            " + ordering ", round(to, digits=4), " + cliquetree(perm) ", round(tct, digits=4), " + rest ", round(td - tsc - tpe - tga - to - tct, digits=4)); flush(stdout)
    end
end
