include("workloads.jl")
using CSets: CSets, sccs, permute, gaifman, invspan, invmap, separators, FVector
using CliqueTrees: BipartiteGraph, cliquetree, MMD, residual, parentindex
function parts(X, S; alg)
    T = Float64[]
    tic() = time_ns()
    t = tic(); nu = X.dom.nu; nv = S.nv; C = sccs(X.dom); scc = FVector{Int}(undef, nu); for u in 1:nu; scc[u] = C.adjncy[nu + 1 - u]; end
    X = permute(X, scc); S = permute(S, scc); push!(T, time_ns() - t)
    t = tic(); G = gaifman(S); graph = BipartiteGraph(G.nv, G.nu, G.ne, G.xadj, G.adjncy); push!(T, time_ns() - t)
    t = tic(); perm, clique = cliquetree(graph; alg); nt = length(clique); push!(T, time_ns() - t)
    t = tic(); L = invspan(permute(invspan(S), perm)); push!(T, time_ns() - t)
    t = tic()
    node = zeros(Int, nv); prnt = zeros(Int, nt); fst = zeros(Int, nu); xres = zeros(Int, nt + 1); back = zeros(Int, nu)
    for t in 1:nt
        for v in residual(clique, t); node[v] = t; end
        p = parentindex(clique, t); prnt[t] = !isnothing(p) ? p : (t < nt ? nt : 0)
    end
    push!(T, time_ns() - t)
    t = tic()
    for u in 1:nu
        if L.xadj[u] <= L.xadj[u + 1] - 1; fst[u] = L.adjncy[L.xadj[u]]; tt = node[fst[u]] else fst[u] = nv + 1; tt = nt end
        xres[tt + 1] += 1
    end
    order = invmap(nv + 1, nu, FVector{Int}(fst)).adjncy
    tree = invmap(nt, nt, FVector{Int}(prnt)); Y = permute(X, order); R = permute(L, order); push!(T, time_ns() - t)
    t = tic(); sep = separators(R, clique, nt); push!(T, time_ns() - t)
    return T ./ 1e9
end
for n in (10000, 30000)
    c, = miter(n, adder!, (c, a, b) -> buggy!(c, a, b, n ÷ 2)); X = instance(c); S = support(X)
    P = parts(X, S; alg=MMD())
    for _ in 1:4; Q = parts(X, S; alg=MMD()); P = min.(P, Q); end
    println("n=$n MMD: sccs+permute=", round(P[1], digits=4), " gaifman=", round(P[2], digits=4), " cliquetree(MMD)=", round(P[3], digits=4),
        " L=invspan/permute=", round(P[4], digits=4), " residual/parent loop=", round(P[5], digits=4), " fst/invmap/permute=", round(P[6], digits=4), " separators=", round(P[7], digits=4), " sum=", round(sum(P), digits=4))
end
