struct Decomposition{I}
    #
    # We represent structured decompositions as diagrams D: I → Set:
    #
    #   ┌─────────────────────┐
    #   │    C     E          │
    #   │    ⇊     ⇊          │
    #   │    T  ←  R  →  V    │
    #   │       ↖  ↑          │
    #   │          S          │
    #   └─────────────────────┘
    #
    # where
    #
    #   - T is a set of nodes
    #   - V is a set of vertices
    #   - E is a set of edges
    #   - C is a set of tree edges
    #   - R is a set of residual incidences
    #   - S is a set of separator incidences
    #
    # The map R → V is a bijection, so R is a relabeling of V. Everything
    # else is expressed in terms of R.
    #
    # # Attributes
    #
    # The instance:
    #
    #    E ⇉ R
    #
    inst::Instance{I}
    #
    # The residuals:
    #
    #    T ← R → V
    #
    res::Span{I}
    #
    # The separators:
    #
    #    T ← S → R
    #
    sep::Span{I}
    #
    # The tree:
    #
    #    T ← C → T
    #
    tree::Span{I}
end

function gaifman(S::Span{I}) where {I}
    nv = S.nv
    T = invspan(S)

    marker = FVector{I}(undef, nv)

    for v in oneto(nv)
        marker[v] = zero(I)
    end

    ne = zero(I)

    for v in oneto(nv)
        tag = v
        pstrt = T.xadj[v]
        pstop = T.xadj[v + one(I)] - one(I)

        for p in pstrt:pstop
            u = T.adjncy[p]
            qstrt = S.xadj[u]
            qstop = S.xadj[u + one(I)] - one(I)

            for q in qstrt:qstop
                w = S.adjncy[q]

                if w != v && marker[w] < tag
                    marker[w] = tag
                    ne += one(I)
                end
            end
        end
    end

    xadj   = FVector{I}(undef, nv + one(I))
    adjncy = FVector{I}(undef, ne)

    r = one(I)

    for v in oneto(nv)
        tag = nv + v
        pstrt = T.xadj[v]
        pstop = T.xadj[v + one(I)] - one(I)
        xadj[v] = r

        for p in pstrt:pstop
            u = T.adjncy[p]
            qstrt = S.xadj[u]
            qstop = S.xadj[u + one(I)] - one(I)

            for q in qstrt:qstop
                w = S.adjncy[q]

                if w != v && marker[w] < tag
                    marker[w] = tag
                    adjncy[r] = w
                    r += one(I)
                end
            end
        end
    end

    xadj[nv + one(I)] = r
    return Span{I}(nv, nv, ne, xadj, adjncy)
end

function decomposition(X::Instance{I}, S::Span{I}; kw...) where {I}
    nu = X.dom.nu
    nv = S.nv
    #
    # relabel the elements in topological order of their strongly
    # connected components, sources first
    #
    C = sccs(X.dom)
    scc = FVector{I}(undef, nu)

    for u in oneto(nu)
        scc[u] = C.adjncy[nu + one(I) - u]
    end

    X = permute(X, scc)
    S = permute(S, scc)

    G = gaifman(S)
    graph = BipartiteGraph(G.nv, G.nu, G.ne, G.xadj, G.adjncy)
    perm, clique = cliquetree(graph; kw...)
    nt = convert(I, length(clique))

    L = invspan(permute(invspan(S), perm))

    node = FVector{I}(undef, nv)
    prnt = FVector{I}(undef, nt)
    fst  = FVector{I}(undef, nu)
    xres = FVector{I}(undef, nt + one(I))
    back = FVector{I}(undef, nu)

    for t in oneto(nt)
        for v in residual(clique, t)
            node[v] = t
        end

        p = parentindex(clique, t)

        if !isnothing(p)
            prnt[t] = p
        elseif t < nt
            prnt[t] = nt
        else
            prnt[t] = zero(I)
        end
    end

    for t in oneto(nt + one(I))
        xres[t] = zero(I)
    end

    for u in oneto(nu)
        pstrt = L.xadj[u]
        pstop = L.xadj[u + one(I)] - one(I)

        if pstrt <= pstop
            fst[u] = L.adjncy[pstrt]
            t = node[fst[u]]
        else
            fst[u] = nv + one(I)
            t = nt
        end

        xres[t + one(I)] += one(I)
    end

    xres[one(I)] = q = one(I)

    for t in oneto(nt)
        xres[t + one(I)] = q += xres[t + one(I)]
    end

    order = invmap(nv + one(I), nu, fst).adjncy

    for u in oneto(nu)
        back[u] = scc[order[u]]
    end

    res = Span{I}(nt, nu, nu, xres, back)
    tree = invmap(nt, nt, prnt)
    Y = permute(X, order)
    R = permute(L, order)
    sep = separators(R, clique, nt)
    return Decomposition{I}(Y, res, sep, tree)
end

function separators(R::Span{I}, clique, nt::I) where {I}
    nu = R.nu
    nv = R.nv

    C = CliqueTrees.separators(clique)
    xfst = FVector{I}(undef, nv + one(I))

    for v in oneto(nv + one(I))
        xfst[v] = zero(I)
    end

    for u in oneto(nu)
        pstrt = R.xadj[u]
        pstop = R.xadj[u + one(I)] - one(I)

        if pstrt <= pstop
            xfst[R.adjncy[pstrt] + one(I)] += one(I)
        end
    end

    xfst[one(I)] = q = one(I)

    for v in oneto(nv)
        xfst[v + one(I)] = q += xfst[v + one(I)]
    end

    nempty = nu - xfst[nv + one(I)] + one(I)
    ne = zero(I)

    for t in oneto(nt)
        kstrt = pointers(C)[t]
        kstop = pointers(C)[t + one(I)] - one(I)

        for k in kstrt:kstop
            v = targets(C)[k]
            ustrt = xfst[v]
            ustop = xfst[v + one(I)] - one(I)

            for u in ustrt:ustop
                pstrt = R.xadj[u] + one(I)
                pstop = R.xadj[u + one(I)] - one(I)
                j = k + one(I)
                found = true

                for p in pstrt:pstop
                    w = R.adjncy[p]

                    while j <= kstop && targets(C)[j] < w
                        j += one(I)
                    end

                    if j > kstop || targets(C)[j] != w
                        found = false
                        break
                    end

                    j += one(I)
                end

                if found
                    ne += one(I)
                end
            end
        end

        if t < nt
            ne += nempty
        end
    end

    xadj   = FVector{I}(undef, nt + one(I))
    adjncy = FVector{I}(undef, ne)

    q = one(I)

    for t in oneto(nt)
        kstrt = pointers(C)[t]
        kstop = pointers(C)[t + one(I)] - one(I)
        xadj[t] = q

        for k in kstrt:kstop
            v = targets(C)[k]
            ustrt = xfst[v]
            ustop = xfst[v + one(I)] - one(I)

            for u in ustrt:ustop
                pstrt = R.xadj[u] + one(I)
                pstop = R.xadj[u + one(I)] - one(I)
                j = k + one(I)
                found = true

                for p in pstrt:pstop
                    w = R.adjncy[p]

                    while j <= kstop && targets(C)[j] < w
                        j += one(I)
                    end

                    if j > kstop || targets(C)[j] != w
                        found = false
                        break
                    end

                    j += one(I)
                end

                if found
                    adjncy[q] = u
                    q += one(I)
                end
            end
        end

        if t < nt
            ustrt = xfst[nv + one(I)]

            for u in ustrt:nu
                adjncy[q] = u
                q += one(I)
            end
        end
    end

    xadj[nt + one(I)] = q
    return Span{I}(nt, nu, ne, xadj, adjncy)
end
