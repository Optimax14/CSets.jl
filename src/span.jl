struct Span{I}
    #
    # We represent spans as diagrams D: I → Set:
    #
    #   ┌──────────────────┐
    #   │    adj  adjncy   │
    #   │  U  ←  E  →  V   │
    #   └──────────────────┘
    #
    # where
    #
    #   - U is a set of left vertices
    #   - V is a set of right vertices
    #   - E is a set of edges
    #
    # # Attributes
    #
    # The number of left vertices:
    #
    #    | U |
    #
    nu::I
    #
    # The number of right vertices:
    #
    #    | V |
    #
    nv::I
    #
    # The number of edges:
    #
    #    | E |
    #
    ne::I
    #
    # Each left vertex u ∈ U is incident to the edges
    #
    #    xadj[u] ... xadj[u + 1] - 1 ∈ E
    #
    xadj::FVector{I}
    #
    # Each edge e ∈ E targets a right vertex
    #
    #    adjncy[e] ∈ V
    #
    adjncy::FVector{I}
end

function invspan(S::Span{I}) where {I}
    nu = S.nu
    nv = S.nv
    ne = S.ne

    xadj   = FVector{I}(undef, nv + one(I))
    adjncy = FVector{I}(undef, ne)

    for v in oneto(nv)
        xadj[v + one(I)] = zero(I)
    end

    for p in oneto(ne)
        v = S.adjncy[p]

        if v < nv
            xadj[v + two(I)] += one(I)
        end
    end

    xadj[one(I)] = q = one(I)

    for v in oneto(nv)
        xadj[v + one(I)] = q += xadj[v + one(I)]
    end

    for u in oneto(nu)
        pstrt = S.xadj[u]
        pstop = S.xadj[u + one(I)] - one(I)

        for p in pstrt:pstop
            v = S.adjncy[p]
            adjncy[xadj[v + one(I)]] = u
            xadj[v + one(I)] += one(I)
        end
    end

    return Span{I}(nv, nu, ne, xadj, adjncy)
end

function permute(S::Span{I}, order::AbstractVector{I}) where {I}
    nu = S.nu
    nv = S.nv
    ne = S.ne

    xadj   = FVector{I}(undef, nu + one(I))
    adjncy = FVector{I}(undef, ne)

    q = one(I)

    for u in oneto(nu)
        w = order[u]
        pstrt = S.xadj[w]
        pstop = S.xadj[w + one(I)] - one(I)
        xadj[u] = q

        for p in pstrt:pstop
            adjncy[q] = S.adjncy[p]
            q += one(I)
        end
    end

    xadj[nu + one(I)] = q
    return Span{I}(nu, nv, ne, xadj, adjncy)
end

#
# Compute the strongly connected components of a graph using
# Tarjan's algorithm. The components are the columns of the
# result, listed in reverse topological order: each component
# appears after every component reachable from it.
#
function sccs(S::Span{I}) where {I}
    n = S.nu
    np1 = n + one(I)

    low = FVector{I}(undef, n)
    arc = FVector{I}(undef, n)
    ptr = FVector{I}(undef, np1)
    tgt = FVector{I}(undef, n)

    for u in oneto(n)
        low[u] = zero(I)
    end

    r = np1
    c = zero(I)
    i = one(I)

    for u in oneto(n)
        if iszero(low[u])
            r -= one(I)
            tgt[r] = u
            l = low[u] = np1 - r
            d = one(I)
            ptr[np1 - d] = r
            v = u
            p = S.xadj[v]
            pstop = S.xadj[v + one(I)]

            while true
                if p < pstop
                    w = S.adjncy[p]
                    p += one(I)
                    m = low[w]

                    if iszero(m)
                        low[v] = l
                        arc[d] = p
                        r -= one(I)
                        tgt[r] = w
                        l = low[w] = np1 - r
                        d += one(I)
                        ptr[np1 - d] = r
                        v = w
                        p = S.xadj[v]
                        pstop = S.xadj[v + one(I)]
                    else
                        l = min(l, m)
                    end
                else
                    rstop = ptr[np1 - d]

                    if l + rstop == np1
                        c += one(I)
                        ptr[c] = i

                        for j in r:rstop
                            w = tgt[i] = tgt[j]
                            low[w] = np1
                            i += one(I)
                        end

                        r = rstop + one(I)
                    else
                        low[v] = l
                    end

                    d -= one(I)

                    if iszero(d)
                        break
                    end

                    v = tgt[ptr[np1 - d]]
                    l = min(low[v], l)
                    p = arc[d]
                    pstop = S.xadj[v + one(I)]
                end
            end
        end
    end

    ptr[c + one(I)] = i
    return Span{I}(c, n, n, ptr, tgt)
end
