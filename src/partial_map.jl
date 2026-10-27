struct PartialMap{I}
    #
    # We represent partial maps V ⇀ U as diagrams D: I → Set:
    #
    #   ┌──────────────────────────┐
    #   │                map       │
    #   │  U  ←  E  ↪  V  →  U + 1 │
    #   └──────────────────────────┘
    #
    # where
    #
    #   - V is a set of vertices
    #   - U is a set of vertices
    #   - E is the set of vertices in V where the map is defined
    #
    # # Attributes
    #
    # Each vertex v ∈ V maps to
    #
    #    map[v] ∈ U
    #
    # if map[v] is positive, and is undefined otherwise.
    #
    map::FVector{I}
    #
    # The preimage of the partial map:
    #
    #    U ← E ↪ V
    #
    inv::Span{I}
end

const PMap{I} = PartialMap{I}

function PMap{I}(nu::Integer, nv::Integer, map::AbstractVector) where {I}
    inv = invmap(nu, nv, map)
    return PMap{I}(map, inv)
end

function PMap(nu::I, nv::I, map::AbstractVector{I}) where {I}
    return PMap{I}(nu, nv, map)
end

function invmap(nu::I, nv::I, map::AbstractVector{I}) where {I}
    xadj   = FVector{I}(undef, nu + one(I))
    adjncy = FVector{I}(undef, nv)

    for u in oneto(nu)
        xadj[u + one(I)] = zero(I)
    end

    for v in oneto(nv)
        u = map[v]

        if zero(I) < u < nu
            xadj[u + two(I)] += one(I)
        end
    end

    xadj[one(I)] = p = one(I)

    for u in oneto(nu)
        xadj[u + one(I)] = p += xadj[u + one(I)]
    end

    for v in oneto(nv)
        u = map[v]

        if ispositive(u)
            adjncy[xadj[u + one(I)]] = v
            xadj[u + one(I)] += one(I)
        end
    end

    ne = xadj[nu + one(I)] - one(I)
    return Span{I}(nu, nv, ne, xadj, adjncy)
end
