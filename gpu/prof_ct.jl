include("workloads.jl")
using CSets: sccs, permute, gaifman
using CliqueTrees: BipartiteGraph, cliquetree, permutation, MF
for n in (10000, 30000, 100000)
    c, = miter(n, adder!, (c, a, b) -> buggy!(c, a, b, n ÷ 2)); X = instance(c); S = support(X)
    G = gaifman(S); graph = BipartiteGraph(G.nv, G.nu, G.ne, G.xadj, G.adjncy)
    perm, _ = permutation(graph; alg=MF()); cliquetree(graph; alg=perm); decomposition(X, S; alg=MF())
    to = minimum(@elapsed(permutation(graph; alg=MF())) for _ in 1:5)
    tc = minimum(@elapsed(cliquetree(graph; alg=perm)) for _ in 1:5)
    tall = minimum(@elapsed(cliquetree(graph; alg=MF())) for _ in 1:5)
    td = minimum(@elapsed(decomposition(X, S; alg=MF())) for _ in 1:5)
    tdp = minimum(@elapsed(decomposition(X, S; alg=perm)) for _ in 1:5)
    println("n=$n ordering(MF)=", round(to, digits=4), " cliquetree(given perm)=", round(tc, digits=4), " cliquetree(MF)=", round(tall, digits=4),
        " decomposition(MF)=", round(td, digits=4), " decomposition(given perm)=", round(tdp, digits=4))
end
