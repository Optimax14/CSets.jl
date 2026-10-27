#
# Workloads for the GPU study: the circuit miters from
# examples/circuits.jl, plus graph homomorphisms G → H where G has low
# treewidth and H is larger (colorings, random targets).
#
using CSets: FVector, Span, Instance, saturate, color, decomposition
using CliqueTrees: MF
using Random

include(joinpath(@__DIR__, "..", "examples", "circuits.jl"))

#
# The schema of graphs: V, E with src, tgt : E → V.
#
function graphschema()
    xadj = FVector{Int}([1, 1, 3])
    adjncy = FVector{Int}([1, 1])
    return Span{Int}(2, 2, 2, xadj, adjncy)
end

#
# The graph with vertices 1:n and the given edges as a C-set: vertices
# first, then one element per edge.
#
function graphinstance(n::Int, edges::Vector{Tuple{Int, Int}})
    m = length(edges)
    xadj = Vector{Int}(undef, n + m + 1)
    xadj[1:n + 1] .= 1
    adjncy = Int[]
    arc = Int[]
    vtx = [fill(1, n); fill(2, m)]

    for (i, (u, v)) in enumerate(edges)
        push!(adjncy, u); push!(arc, 1)
        push!(adjncy, v); push!(arc, 2)
        xadj[n + i + 1] = length(adjncy) + 1
    end

    dom = Span{Int}(n + m, n + m, 2m, FVector{Int}(xadj), FVector{Int}(adjncy))
    return Instance{Int}(dom, graphschema(), FVector{Int}(vtx), FVector{Int}(arc))
end

# symmetric target: both orientations of every edge
function symmetric(n::Int, edges::Vector{Tuple{Int, Int}})
    return graphinstance(n, [edges; [(v, u) for (u, v) in edges]])
end

function graphsupport(X::Instance)
    n = X.dom.nu
    nv = count(==(1), X.vtx)
    xadj = [min(x, nv + 1) for x in 1:n + 1]
    return saturate(X, Span{Int}(n, nv, nv, FVector{Int}(xadj), FVector{Int}(collect(1:nv))))
end

complete(k) = symmetric(k, [(u, v) for u in 1:k for v in u+1:k])

#
# A random partial k-tree on n vertices: grow a k-tree by attaching each
# new vertex to a random k-clique, keeping each edge with probability p.
# `bushy` attaches to a uniformly random earlier clique (shallow, bushy
# trees); otherwise to one of the last few (deep, path-like trees).
#
function partialktree(rng, n, k, p; bushy=true)
    cliques = [collect(1:k)]
    edges = Tuple{Int, Int}[(u, v) for u in 1:k for v in u+1:k]

    for v in k+1:n
        c = bushy ? rand(rng, cliques) : cliques[max(1, length(cliques) - rand(rng, 0:2))]
        for u in c
            push!(edges, (u, v))
        end
        drop = rand(rng, 1:k)
        push!(cliques, [c[1:drop-1]; c[drop+1:end]; v])
    end

    keep = [e for e in edges if rand(rng) < p]
    return n, shuffle(rng, keep)
end

function randomgraph(rng, n, p)
    return [(u, v) for u in 1:n for v in u+1:n if rand(rng) < p]
end

# grid graph w × l (treewidth w, path-like decomposition of length ~l)
function grid(w, l)
    id(i, j) = (j - 1) * w + i
    edges = Tuple{Int, Int}[]
    for j in 1:l, i in 1:w
        i < w && push!(edges, (id(i, j), id(i + 1, j)))
        j < l && push!(edges, (id(i, j), id(i, j + 1)))
    end
    return w * l, edges
end

#
# check a map h : X → Y is a homomorphism
#
function ishom(X::Instance, Y::Instance, h)
    for x in 1:X.dom.nu
        X.vtx[x] == Y.vtx[h[x]] || return false
        for p in X.dom.xadj[x]:X.dom.xadj[x + 1] - 1
            a = X.arc[p]
            z = h[X.dom.adjncy[p]]
            ok = false
            for q in Y.dom.xadj[h[x]]:Y.dom.xadj[h[x] + 1] - 1
                if Y.arc[q] == a
                    ok = Y.dom.adjncy[q] == z
                    break
                end
            end
            ok || return false
        end
    end
    return true
end
