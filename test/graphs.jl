# directed multigraphs as C-sets (two sorts: vertices and edges, arcs src and tgt)
using CSets
using CSets: FVector, Span, Instance, saturate
using CliqueTrees: MF
graphschema() = Span{Int}(2, 2, 2, FVector{Int}([1, 1, 3]), FVector{Int}([1, 1]))
function graph_instance(n::Int, edges::Vector{Tuple{Int, Int}})
    m = length(edges)
    xadj = Vector{Int}(undef, n + m + 1); xadj[1:n + 1] .= 1
    adjncy = Vector{Int}(undef, 2m); arc = Vector{Int}(undef, 2m)
    for (i, (u, v)) in enumerate(edges)
        adjncy[2i - 1] = u; arc[2i - 1] = 1; adjncy[2i] = v; arc[2i] = 2; xadj[n + i + 1] = 2i + 1
    end
    vtx = [fill(1, n); fill(2, m)]
    dom = Span{Int}(n + m, n + m, 2m, FVector{Int}(xadj), FVector{Int}(adjncy))
    return Instance{Int}(dom, graphschema(), FVector{Int}(vtx), FVector{Int}(arc))
end
function graph_support(X::Instance)
    n = X.dom.nu; nv = count(==(1), X.vtx)
    return saturate(X, Span{Int}(n, nv, nv, FVector{Int}([min(x, nv + 1) for x in 1:n + 1]), FVector{Int}(collect(1:nv))))
end
