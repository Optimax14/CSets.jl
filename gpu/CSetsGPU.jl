#
# Level-synchronous GPU homomorphism search over a structured
# decomposition (design (a) + (b) in GPU_NOTES.md).
#
# The search of each bag is the backtracking search of
# CSets.bagsearch!, run by one GPU thread. A bag is split into many
# work items by fixing a prefix of its separator: the items of a bag
# are its partial assignments of positions 1 ... d, in the order the
# CPU search visits them, so concatenating their columns in item order
# reproduces the CPU table exactly.
#
module CSetsGPU

using Base: oneto
using Adapt
using CUDA
using CSets: CSets, Decomposition, Instance, Span, FVector, invmap, invvtx, normalize, two

const J = Int32

#
# The plan of each bag: everything bagsearch! computes before it
# starts searching, flattened over the bags.
#
struct Plan{V}
    m::V      # bag size
    s::V      # separator size
    xpos::V   # positions of bag t are xpos[t] + 1 ... xpos[t] + m[t]
    ord::V    # element at each position
    vtx::V    # its type
    bknd::V   # candidate kind (1, 2, 3)
    bpos::V   # position determining the candidates
    barc::V   # arc determining the candidates
    xchk::V   # checks at position k: xchk[xpos[t] + t - 1 + k] ...
    chka::V
    chkp::V
    chkd::V
    xnar::V   # narrowings at position k: xnar[xpos[t] + t - 1 + k] ...
    narc::V
    nari::V
    naro::V
    xsl::V    # child slots of bag t: xsl[t] ... xsl[t + 1] - 1
    slc::V    # child
    slo::V    # slot offset
end

Adapt.@adapt_structure Plan

struct Target{V}
    xadj::V     # Y.dom.xadj
    adjncy::V   # Y.dom.adjncy
    vtx::V      # Y.vtx
    codx::V     # Y.cod.xadj
    coda::V     # Y.cod.adjncy
    typx::V     # typ.xadj
    base::V
    prex::V     # pre.xadj
    prea::V     # pre.adjncy
    nty::Int32  # number of types (typx has nty + 1 entries per target)
    na::Int32   # number of arcs (base has na + 1 entries per target)
end

Adapt.@adapt_structure Target

function plan(D::Decomposition{I}) where {I}
    X = D.inst
    nu = X.dom.nu
    nt = D.tree.nu
    #
    # sizes
    #
    npos = zero(I)
    nchk = zero(I)
    nnar = zero(I)
    nsl = zero(I)

    for t in oneto(nt)
        rstrt = D.res.xadj[t]; rstop = D.res.xadj[t + one(I)] - one(I)
        sstrt = D.sep.xadj[t]; sstop = D.sep.xadj[t + one(I)] - one(I)
        npos += sstop - sstrt + rstop - rstrt + two(I)

        for x in rstrt:rstop
            nchk += X.dom.xadj[x + one(I)] - X.dom.xadj[x]
        end

        for p in sstrt:sstop
            x = D.sep.adjncy[p]
            nchk += X.dom.xadj[x + one(I)] - X.dom.xadj[x]
        end

        for j in D.tree.xadj[t]:D.tree.xadj[t + one(I)] - one(I)
            c = D.tree.adjncy[j]
            nnar += D.sep.xadj[c + one(I)] - D.sep.xadj[c]
            nsl += one(I)
        end
    end

    pos  = zeros(I, nu)
    m    = Vector{J}(undef, nt)
    s    = Vector{J}(undef, nt)
    xpos = Vector{J}(undef, nt)
    ord  = Vector{J}(undef, npos)
    vtx  = Vector{J}(undef, npos)
    bknd = Vector{J}(undef, npos)
    bpos = Vector{J}(undef, npos)
    barc = Vector{J}(undef, npos)
    xchk = Vector{J}(undef, npos + nt)
    chka = Vector{J}(undef, nchk)
    chkp = Vector{J}(undef, nchk)
    chkd = Vector{J}(undef, nchk)
    xnar = Vector{J}(undef, npos + nt)
    narc = Vector{J}(undef, nnar)
    nari = Vector{J}(undef, nnar)
    naro = Vector{J}(undef, nnar)
    xsl  = Vector{J}(undef, nt + one(I))
    slc  = Vector{J}(undef, nsl)
    slo  = Vector{J}(undef, nsl)
    maxo = zero(I)
    o = zero(I)
    q = one(I)
    l = one(I)
    r = one(I)

    for t in oneto(nt)
        rstrt = D.res.xadj[t]; rstop = D.res.xadj[t + one(I)] - one(I)
        sstrt = D.sep.xadj[t]; sstop = D.sep.xadj[t + one(I)] - one(I)
        cstrt = D.tree.xadj[t]; cstop = D.tree.xadj[t + one(I)] - one(I)
        st = sstop - sstrt + one(I)
        mt = st + rstop - rstrt + one(I)
        m[t] = mt; s[t] = st; xpos[t] = o
        pp = o + t - one(I)
        k = o

        for p in sstop:-one(I):sstrt
            k += one(I)
            ord[k] = D.sep.adjncy[p]
        end

        for x in rstop:-one(I):rstrt
            k += one(I)
            ord[k] = x
        end

        for k in oneto(mt)
            x = ord[o + k]
            pos[x] = k
            vtx[o + k] = X.vtx[x]
            bknd[o + k] = one(J)
            bpos[o + k] = zero(J)
            barc[o + k] = zero(J)
            xchk[pp + k + one(I)] = zero(J)
            xnar[pp + k + one(I)] = zero(J)
        end
        #
        # checks, as in bagsearch!
        #
        for k in oneto(mt)
            w = ord[o + k]

            for p in X.dom.xadj[w]:X.dom.xadj[w + one(I)] - one(I)
                j = max(k, pos[X.dom.adjncy[p]])

                if j < mt
                    xchk[pp + j + two(I)] += one(J)
                end
            end
        end

        xchk[pp + one(I)] = q

        for k in oneto(mt)
            xchk[pp + k + one(I)] = q += xchk[pp + k + one(I)]
        end

        for k in oneto(mt)
            w = ord[o + k]

            for p in X.dom.xadj[w]:X.dom.xadj[w + one(I)] - one(I)
                a = X.arc[p]
                j = pos[X.dom.adjncy[p]]

                if j <= k
                    i = xchk[pp + k + one(I)]
                    chka[i] = a; chkp[i] = j; chkd[i] = one(J)
                    xchk[pp + k + one(I)] += one(J)

                    if j < k && isone(bknd[o + k])
                        bknd[o + k] = two(J); bpos[o + k] = j; barc[o + k] = a
                    end
                else
                    i = xchk[pp + j + one(I)]
                    chka[i] = a; chkp[i] = k; chkd[i] = zero(J)
                    xchk[pp + j + one(I)] += one(J)

                    if bknd[o + j] != J(3)
                        bknd[o + j] = J(3); bpos[o + j] = k; barc[o + j] = a
                    end
                end
            end
        end

        q = xchk[pp + mt + one(I)]
        #
        # child narrowings, as in bagsearch!
        #
        for j in cstrt:cstop
            c = D.tree.adjncy[j]

            for p in D.sep.xadj[c]:D.sep.xadj[c + one(I)] - one(I)
                k = pos[D.sep.adjncy[p]]

                if k < mt
                    xnar[pp + k + two(I)] += one(J)
                end
            end
        end

        xnar[pp + one(I)] = l

        for k in oneto(mt)
            xnar[pp + k + one(I)] = l += xnar[pp + k + one(I)]
        end

        xsl[t] = r
        so = one(I)

        for j in cstrt:cstop
            c = D.tree.adjncy[j]
            pstrt = D.sep.xadj[c]; pstop = D.sep.xadj[c + one(I)] - one(I)
            slc[r] = c; slo[r] = so
            r += one(I)
            i = zero(I)

            for p in pstop:-one(I):pstrt
                i += one(I)
                k = pos[D.sep.adjncy[p]]
                e = xnar[pp + k + one(I)]
                narc[e] = c; nari[e] = i; naro[e] = so
                xnar[pp + k + one(I)] += one(J)
            end

            so += pstop - pstrt + two(I)
        end

        l = xnar[pp + mt + one(I)]
        maxo = max(maxo, so - one(I))
        o += mt
    end

    xsl[nt + one(I)] = r
    P = Plan(m, s, xpos, ord, vtx, bknd, bpos, barc, xchk, chka, chkp, chkd, xnar, narc, nari, naro, xsl, slc, slo)
    return P, Int(maximum(m; init=0)), Int(maxo)
end

#
# The (normalized) targets Y_1 ... Y_k as one instance: element y of
# Y_q is oy[q] + y. Per target, typx and base are blocks of nty + 1 and
# na + 1 entries; base also carries the offset of Y_q's preimage slots.
#
function target(Ys::Vector{Instance{I}}) where {I}
    xadj = J[]; adjncy = J[]; vtx = J[]; typx = J[]; base = J[]; prex = J[]; prea = J[]
    oy = 0; of = 0; op = 0
    Y1 = first(Ys)
    nty = Y1.cod.nu; na = Y1.cod.ne

    for Y in Ys
        ny = Y.dom.nu; nf = Y.dom.ne
        ysrc = Vector{I}(undef, nf)

        for y in oneto(ny), p in Y.dom.xadj[y]:Y.dom.xadj[y + one(I)] - one(I)
            ysrc[p] = y
        end

        typ = invvtx(Y)
        bq = Vector{I}(undef, na + one(I))
        bq[1] = 1

        for a in oneto(na)
            d = Y.cod.adjncy[a]
            bq[a + 1] = bq[a] + typ.xadj[d + 1] - typ.xadj[d]
        end

        ykey = Vector{I}(undef, nf)

        for e in oneto(nf)
            a = Y.arc[e]
            ykey[e] = bq[a] + Y.dom.adjncy[e] - typ.xadj[Y.cod.adjncy[a]]
        end

        nslot = bq[na + 1] - one(I)
        pre = invmap(nslot, nf, ykey)

        for y in oneto(ny)
            push!(xadj, Y.dom.xadj[y] + of)
            push!(vtx, Y.vtx[y])
        end

        for e in oneto(nf)
            push!(adjncy, Y.dom.adjncy[e] + oy)
        end

        for u in 1:nty + 1
            push!(typx, typ.xadj[u] + oy)
        end

        for a in 1:na + 1
            push!(base, bq[a] + op)
        end

        for l in oneto(nslot)
            push!(prex, pre.xadj[l] + of)
        end

        for p in oneto(nf)
            push!(prea, ysrc[pre.adjncy[p]] + oy)
        end

        oy += ny; of += nf; op += nslot
    end

    push!(xadj, of + 1)
    push!(prex, of + 1)
    v(x) = CuArray{J}(collect(x))
    return Target(v(xadj), v(isempty(adjncy) ? J[0] : adjncy), v(vtx), v(Y1.cod.xadj), v(Y1.cod.adjncy),
        v(typx), v(base), v(prex), v(isempty(prea) ? J[0] : prea), J(nty), J(na))
end

@inline function image(Y, y, a)
    @inbounds return Y.adjncy[Y.xadj[y] + a - Y.codx[Y.vtx[y]]]
end

#
# The work items of a round: item i searches bag ibag[i] with positions
# 1 ... idep[i] fixed to ipre[(i - 1) * W + 1] ...
#
# Modes:
#   0  forward: emit every separator assignment (columns of width s)
#   1  expand:  emit the partial assignments of positions 1 ... d + 1
#   2  backward: positions 1 ... s are read from h; write the first
#      extension to h
#
function searchkernel!(P, Y, tab, xtab, ncol, h, ibag, idep, ipre, W, nitem, next,
        rkey, rdep, rval, cap, cnt, scr, maxm, maxo, mode, icnt, budget, abort, aflag, iq, ntJ, nuJ,
        S, wo, wv, cval, xval, rvv)
    tid = (blockIdx().x - one(J)) * blockDim().x + threadIdx().x
    nthr = gridDim().x * blockDim().x
    one_ = one(J)

    @inbounds while true
        i = CUDA.atomic_add!(pointer(next), one_) + one_

        if i > nitem
            break
        end

        t = ibag[i]
        d = idep[i]
        #
        # target q: its tables, its h, its blocks of typx and base
        #
        q = iq[i]
        qo = (q - one_) * ntJ
        tq = qo + t
        ho = (q - one_) * nuJ
        yo = (q - one_) * (Y.nty + one_)
        ao = (q - one_) * (Y.na + one_)
        m = P.m[t]
        s = P.s[t]
        po = P.xpos[t]
        pp = po + t - one_
        #
        # scratch: kind, arg, cur, stop, hp at k, rlo, rhi at slot o
        #
        KIND = zero(J); ARG = J(maxm); CUR = J(2maxm); STOP = J(3maxm); HP = J(4maxm)
        RLO = J(5maxm); RHI = J(5maxm + maxo); BEST = J(5maxm + 2maxo)
        woq = (q - one_) * (nuJ + one_)
        # sc(f, k) = scr[(f + k - 1) * nthr + tid]

        if mode == J(1)
            if d >= s || (budget > zero(J) && iszero(abort[i]))
                emit!(rkey, rdep, rval, cap, cnt, ncol, ipre, W, i, zero(J), tq, d, d, scr, nthr, tid, HP, true, rvv, sone_(S))
                icnt[i] = one_
                continue
            end

            mstop = d + one_
            E = d + one_
        else
            mstop = m
            E = s
        end

        for l in P.xsl[t]:P.xsl[t + one_] - one_
            o = P.slo[l]
            c = P.slc[l]
            scr[(RLO + o - one_) * nthr + tid] = one_
            scr[(RHI + o - one_) * nthr + tid] = ncol[qo + c]
        end

        for k in one_:d
            if mode == J(2)
                v = h[ho + P.ord[po + k]]
            else
                v = ipre[(i - one_) * W + k]
            end

            scr[(HP + k - one_) * nthr + tid] = v
            narrow!(P, tab, xtab, scr, nthr, tid, RLO, RHI, pp, k, v, zero(J), qo)
        end

        if mstop <= d
            if mode != J(2)
                emit!(rkey, rdep, rval, cap, cnt, ncol, ipre, W, i, zero(J), tq, E, E, scr, nthr, tid, HP, false, rvv, mode == zero(J) ? fullval(S, P, Y, scr, nthr, tid, HP, RLO, wo, wv, woq, yo, po, s, m, t, cval, xval, qo) : sone_(S))
                icnt[i] = one_
            end

            continue
        end

        kmin = d + one_
        k = kmin
        fresh = true
        seq = zero(J)
        steps = zero(J)
        aborted = false
        acc = szero_(S)

        while kmin <= k
            if fresh
                fresh = false
                kd = P.bknd[po + k]
                ag = zero(J)
                cu = zero(J)
                st = zero(J)

                if kd == J(3)
                    ag = image(Y, scr[(HP + P.bpos[po + k] - one_) * nthr + tid], P.barc[po + k])
                    cu = one_
                    st = one_
                else
                    if kd == J(2)
                        a = P.barc[po + k]
                        l = Y.base[ao + a] + scr[(HP + P.bpos[po + k] - one_) * nthr + tid] - Y.typx[yo + Y.coda[a]]
                        cu = Y.prex[l]
                        st = Y.prex[l + one_] - one_
                    else
                        u = P.vtx[po + k]
                        cu = Y.typx[yo + u]
                        st = Y.typx[yo + u + one_] - one_
                    end

                    for j in P.xnar[pp + k]:P.xnar[pp + k + one_] - one_
                        ii = P.nari[j]
                        o = P.naro[j]
                        lo = scr[(RLO + o + ii - J(2)) * nthr + tid]
                        hi = scr[(RHI + o + ii - J(2)) * nthr + tid]

                        if kd != J(4) || hi - lo < st - cu
                            kd = J(4)
                            ag = j
                            cu = lo
                            st = hi
                        end
                    end
                end

                scr[(KIND + k - one_) * nthr + tid] = kd
                scr[(ARG + k - one_) * nthr + tid] = ag
                scr[(CUR + k - one_) * nthr + tid] = cu
                scr[(STOP + k - one_) * nthr + tid] = st
            end

            kd = scr[(KIND + k - one_) * nthr + tid]
            ag = scr[(ARG + k - one_) * nthr + tid]
            cu = scr[(CUR + k - one_) * nthr + tid]
            st = scr[(STOP + k - one_) * nthr + tid]
            found = false

            while cu <= st
                steps += one_
                c = cu
                cu += one_

                if kd == one_
                    y = c
                elseif kd == J(2)
                    y = Y.prea[c]
                elseif kd == J(3)
                    y = ag
                else
                    j = ag
                    l = P.narc[j]
                    ii = P.nari[j]
                    o = P.naro[j]
                    w = P.s[l]
                    b = xtab[qo + l] + ii - one_
                    y = tab[b + (c - one_) * w]
                    lo = c
                    hi = c + one_
                    step = one_

                    while hi <= st && tab[b + (hi - one_) * w] <= y
                        lo = hi
                        hi = c + step
                        step += step
                    end

                    hi = min(hi, st + one_)

                    while lo + one_ < hi
                        mid = (lo + hi) >>> one_

                        if tab[b + (mid - one_) * w] <= y
                            lo = mid
                        else
                            hi = mid
                        end
                    end

                    scr[(RLO + o + ii - one_) * nthr + tid] = c
                    scr[(RHI + o + ii - one_) * nthr + tid] = lo
                    cu = lo + one_
                end

                if y <= zero(J)
                    continue
                end

                scr[(HP + k - one_) * nthr + tid] = y
                valid = true

                for q in P.xchk[pp + k]:P.xchk[pp + k + one_] - one_
                    z = scr[(HP + P.chkp[q] - one_) * nthr + tid]

                    if P.chkd[q] == one_
                        valid = image(Y, y, P.chka[q]) == z
                    else
                        valid = image(Y, z, P.chka[q]) == y
                    end

                    if !valid
                        break
                    end
                end

                if valid && allowed(S, wo, wv, woq, P.ord[po + k], y, Y.typx[yo + P.vtx[po + k]]) == false
                    valid = false
                end

                if valid && P.xnar[pp + k] < P.xnar[pp + k + one_]
                    skip = kd == J(4) ? ag : zero(J)
                    valid = narrow!(P, tab, xtab, scr, nthr, tid, RLO, RHI, pp, k, y, skip, qo)
                end

                if valid
                    found = true
                    break
                end
            end

            scr[(CUR + k - one_) * nthr + tid] = cu

            if mode == zero(J) && budget > zero(J) && d < s && steps > budget
                aborted = true
                break
            end

            if found
                if k < mstop
                    k += one_
                    fresh = true
                elseif mode == J(2)
                    if S === nothing
                        for kk in s + one_:m
                            h[ho + P.ord[po + kk]] = scr[(HP + kk - one_) * nthr + tid]
                        end

                        break
                    end
                    #
                    # selective semiring: keep the best extension
                    #
                    val = fullval(S, P, Y, scr, nthr, tid, HP, RLO, wo, wv, woq, yo, po, s, m, t, cval, xval, qo)
                    new = S.add(acc, val)

                    if val != S.zero && (acc == S.zero || (new == val && new != acc))
                        acc = new

                        for kk in s + one_:m
                            scr[(BEST + kk - one_) * nthr + tid] = scr[(HP + kk - one_) * nthr + tid]
                        end
                    end
                elseif S === nothing || m == s || mode == J(1)
                    emit!(rkey, rdep, rval, cap, cnt, ncol, ipre, W, i, seq, tq, E, E, scr, nthr, tid, HP, false, rvv,
                        mode == zero(J) ? fullval(S, P, Y, scr, nthr, tid, HP, RLO, wo, wv, woq, yo, po, s, m, t, cval, xval, qo) : sone_(S))
                    seq += one_

                    if iszero(E)
                        break
                    end

                    k = E
                else
                    #
                    # semiring: sum the extensions of the separator
                    # assignment; emit it when the search leaves the
                    # residual (below)
                    #
                    acc = S.add(acc, fullval(S, P, Y, scr, nthr, tid, HP, RLO, wo, wv, woq, yo, po, s, m, t, cval, xval, qo))
                end
            else
                k -= one_

                if S !== nothing && mode == zero(J) && k == s && m > s
                    if acc != S.zero
                        emit!(rkey, rdep, rval, cap, cnt, ncol, ipre, W, i, seq, tq, E, E, scr, nthr, tid, HP, false, rvv, acc)
                        seq += one_
                    end

                    acc = S.zero
                end
            end
        end

        if S !== nothing && mode == J(2) && acc != S.zero
            for kk in s + one_:m
                h[ho + P.ord[po + kk]] = scr[(BEST + kk - one_) * nthr + tid]
            end
        end

        if mode == zero(J) && budget > zero(J)
            abort[i] = aborted ? one_ : zero(J)

            if aborted
                aflag[one_] = one_
            end
        end

        if mode != J(2)
            icnt[i] = seq
        end
    end

    return
end

@inline function narrow!(P, tab, xtab, scr, nthr, tid, RLO, RHI, pp, k, v, skip, qo)
    one_ = one(J)

    @inbounds for j in P.xnar[pp + k]:P.xnar[pp + k + one_] - one_
        if j == skip
            continue
        end

        c = P.narc[j]
        i = P.nari[j]
        o = P.naro[j]
        s = P.s[c]
        b = xtab[qo + c] + i - one_
        lo = scr[(RLO + o + i - J(2)) * nthr + tid]
        hi = scr[(RHI + o + i - J(2)) * nthr + tid] + one_

        while lo < hi
            mid = (lo + hi) >>> one_

            if tab[b + (mid - one_) * s] < v
                lo = mid + one_
            else
                hi = mid
            end
        end

        strt = lo
        hi = scr[(RHI + o + i - J(2)) * nthr + tid] + one_

        while lo < hi
            mid = (lo + hi) >>> one_

            if tab[b + (mid - one_) * s] <= v
                lo = mid + one_
            else
                hi = mid
            end
        end

        if strt == lo
            return false
        end

        scr[(RLO + o + i - one_) * nthr + tid] = strt
        scr[(RHI + o + i - one_) * nthr + tid] = lo - one_
    end

    return true
end

@inline function emit!(rkey, rdep, rval, cap, cnt, ncol, ipre, W, i, seq, tq, E, d, scr, nthr, tid, HP, passthrough, rvv, val)
    one_ = one(J)
    r = CUDA.atomic_add!(pointer(cnt), one_) + one_
    CUDA.atomic_add!(pointer(ncol, tq), one_)

    @inbounds if r <= cap
        rkey[r] = (UInt64(i) << 32) | UInt64(seq)
        rdep[r] = d
        rvv[r] = val

        for k in one_:E
            if passthrough
                rval[(r - one_) * W + k] = ipre[(i - one_) * W + k]
            else
                rval[(r - one_) * W + k] = scr[(HP + k - one_) * nthr + tid]
            end
        end
    end

    return
end

#
# Semiring helpers. S === nothing is the Boolean search (no values).
#
@inline sone_(::Nothing) = zero(J)
@inline sone_(S) = S.one
@inline szero_(::Nothing) = false
@inline szero_(S) = S.zero

# may element e (of sort start ty in the normalized target) map to y?
@inline function allowed(S, wo, wv, woq, e, y, ty)
    @inbounds if wo[woq + e] < wo[woq + e + one(J)]
        return wv[wo[woq + e] + y - ty] != wzero(S, wv)
    end

    return true
end

@inline wzero(::Nothing, wv) = zero(eltype(wv))
@inline wzero(S, wv) = S.zero

#
# the value of a full assignment of bag t (in HP): the weights of its
# residual elements times the values of the matching child columns
#
@inline fullval(::Nothing, P, Y, scr, nthr, tid, HP, RLO, wo, wv, woq, yo, po, s, m, t, cval, xval, qo) = zero(J)

@inline function fullval(S, P, Y, scr, nthr, tid, HP, RLO, wo, wv, woq, yo, po, s, m, t, cval, xval, qo)
    one_ = one(J)
    r = S.one

    @inbounds for k in s + one_:m
        e = P.ord[po + k]

        if wo[woq + e] < wo[woq + e + one_]
            y = scr[(HP + k - one_) * nthr + tid]
            r = S.mul(r, wv[wo[woq + e] + y - Y.typx[yo + P.vtx[po + k]]])
        end
    end

    @inbounds for l in P.xsl[t]:P.xsl[t + one_] - one_
        o = P.slo[l]
        c = P.slc[l]
        col = scr[(RLO + o + P.s[c] - one_) * nthr + tid]
        r = S.mul(r, cval[xval[qo + c] + col])
    end

    return r
end

#
# Record rec, the seq-th column of item i, has rank
#
#    iscan[i - 1] + seq + 1
#
# in the round's output: items in order, each item's columns in order.
#
function gatherkernel!(iscan, rkey, rdep, rval, W, R, ibag, P, tab, xtab, first, nbag, ndep, npre, mode, iq, niq, ntJ, rvv, cval, xval, S)
    rec = (blockIdx().x - one(J)) * blockDim().x + threadIdx().x
    one_ = one(J)

    @inbounds if rec <= R
        key = rkey[rec]
        i = J(key >> 32)
        seq = J(key & 0xffffffff)
        r = seq + one_

        if i > one_
            r += iscan[i - one_]
        end

        t = ibag[i]
        q = iq[i]
        tq = (q - one_) * ntJ + t

        if mode == zero(J)
            #
            # column r - first[t] of the table of t (for target q)
            #
            s = P.s[t]
            b = xtab[tq] + (r - one_ - first[tq]) * s

            for k in one_:s
                tab[b + k - one_] = rval[(rec - one_) * W + k]
            end

            if S !== nothing
                cval[xval[tq] + r - first[tq]] = rvv[rec]
            end
        else
            #
            # a new item
            #
            nbag[r] = t
            niq[r] = q
            ndep[r] = rdep[rec]

            for k in one_:rdep[rec]
                npre[(r - one_) * W + k] = rval[(rec - one_) * W + k]
            end
        end
    end

    return
end

#
# Reset the counters of a round: ncol of its bags, the record count,
# and the item cursor.
#
function zerokernel!(ncol, bags, n, cnt, next)
    r = (blockIdx().x - one(J)) * blockDim().x + threadIdx().x

    @inbounds if r <= n
        ncol[bags[r]] = zero(J)
    end

    @inbounds if r == one(J)
        cnt[one(J)] = zero(J)
        next[one(J)] = zero(J)
    end

    return
end

#
# Copy what the host needs after a round: the column count of each bag
# and the number of records.
#
function reportkernel!(out, ncol, bags, n, cnt)
    r = (blockIdx().x - one(J)) * blockDim().x + threadIdx().x

    @inbounds if r <= n
        out[r] = ncol[bags[r]]
    end

    @inbounds if r == one(J)
        out[n + one(J)] = cnt[one(J)]
    end

    return
end

function placekernel!(xtab, first, bags, n, buf, xval, vbase)
    r = (blockIdx().x - one(J)) * blockDim().x + threadIdx().x

    @inbounds if r <= n
        xtab[bags[r]] = buf[r]
        first[bags[r]] = buf[n + r]
        xval[bags[r]] = vbase + buf[n + r]
    end

    return
end

#
# the record capacity that sufficed so far: a round that overflows is
# run again with more
#
const STAGE = Ref(1 << 16)

mutable struct Stage{T}
    cap::Int
    rkey::CuVector{UInt64}
    rdep::CuVector{J}
    rval::CuVector{J}
    rvv::CuVector{T}      # the value of each record (semiring search)
end

function Stage(cap, W, ::Type{T}=J) where {T}
    return Stage{T}(cap, CuVector{UInt64}(undef, cap), CuVector{J}(undef, cap), CuVector{J}(undef, cap * W), CuVector{T}(undef, cap))
end

function flatten(levels)
    ptr = [1]

    for l in levels
        push!(ptr, ptr[end] + length(l))
    end

    flat = Vector{J}(undef, ptr[end] - 1)

    for (l, lev) in enumerate(levels)
        copyto!(flat, ptr[l], lev, 1, length(lev))
    end

    return flat, ptr
end

#
# Find a homomorphism D.inst → Y, or nothing. Same answer as
# CSets.color(D, Y), with the same tables.
#
#   - nitems:   split the bags of a level until it has this many items
#               (default: 8 per GPU thread)
#   - minwork:  ... unless the level is smaller than this
#
#
# Everything that depends only on the decomposition: the plan of every
# bag and the levels, on the device. Reuse it for many targets.
#
struct Prepared{I, A, V}
    D::Decomposition{I}
    Pc::Plan{Vector{J}}
    P::A
    maxm::Int
    maxo::Int
    W::Int
    up::Vector{Vector{J}}
    upptr::Vector{Int}
    dnptr::Vector{Int}
    upd::V
    dnd::V
    dns::V
    mlev::Int
    dnflat::Vector{J}
end

function prepare(D::Decomposition{I}) where {I}
    nt = D.tree.nu
    Pc, maxm, maxo = plan(D)
    P = adapt(CuArray, Pc)
    W = max(1, Int(maximum(Pc.s; init=0)))
    #
    # forward levels: by height; backward levels: by depth
    #
    hgt = zeros(Int, nt)
    dep = zeros(Int, nt)

    for t in oneto(nt), j in D.tree.xadj[t]:D.tree.xadj[t + one(I)] - one(I)
        hgt[t] = max(hgt[t], hgt[D.tree.adjncy[j]] + 1)
    end

    for t in nt:-one(I):one(I), j in D.tree.xadj[t]:D.tree.xadj[t + one(I)] - one(I)
        dep[D.tree.adjncy[j]] = dep[t] + 1
    end

    up = [J[] for _ in 0:maximum(hgt; init=0)]
    down = [J[] for _ in 0:maximum(dep; init=0)]

    for t in oneto(nt)
        push!(up[hgt[t] + 1], t)
        push!(down[dep[t] + 1], t)
    end

    upflat, upptr = flatten(up)
    dnflat, dnptr = flatten(down)
    upd = CuVector{J}(upflat)
    dnd = CuVector{J}(dnflat)
    dns = CuVector{J}(J[Pc.s[t] for t in dnflat])
    mlev = max(maximum(length, up), maximum(length, down))
    return Prepared{I, typeof(P), typeof(upd)}(D, Pc, P, maxm, maxo, W, up, upptr, dnptr, upd, dnd, dns, mlev, dnflat)
end

function color(D::Decomposition{I}, Y::Instance{I}; kw...) where {I}
    return color(prepare(D), Y; kw...)
end

function color(prep::Prepared{I}, Y::Instance{I}; kw...) where {I}
    r = color(prep, [Y]; kw...)
    return r isa Vector ? r[1] : r
end

#
# Many targets in one pass: every work item carries a target q, and
# bag t has a table for each target (index (q - 1) * nt + t). Returns
# one result per target (a homomorphism or nothing).
#
#
# With a semiring S (a Semiring, see contractgpu.jl) the tables carry
# values: the sum over the extensions of each separator assignment of
# the products of the weights Ws[q] (upstream's convention) and of the
# child values. Then the result per target is (value, h), with h only
# when witness=true (S selective).
#
function color(prep::Prepared{I}, Ys::Vector{<:Instance{I}}; nitems=0, maxround=8, minwork=256, threads=128, budget=1 << 14, maxsplit=6, stats=nothing, keep=false,
        S=nothing, Ws=nothing, witness=true) where {I}
    tst = time_ns()
    D = prep.D
    Pc, P, maxm, maxo, W = prep.Pc, prep.P, prep.maxm, prep.maxo, prep.W
    up, upptr, dnptr, mlev = prep.up, prep.upptr, prep.dnptr, prep.mlev
    k = length(Ys)
    nz = [normalize(Y) for Y in Ys]
    Yn = Instance{I}[z[1] for z in nz]
    yords = [z[2] for z in nz]
    oys = cumsum([0; [Y.dom.nu for Y in Yn]])
    X = D.inst
    nu = X.dom.nu
    nt = D.tree.nu
    tpl = time_ns() - tst
    T = target(Yn)

    dev = device()
    nsm = attribute(dev, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    blocks = nsm * 8
    nthr = blocks * threads
    nitems = iszero(nitems) ? 8nthr : nitems
    scr = CuVector{J}(undef, (6maxm + 2maxo) * nthr)
    VT = S === nothing ? J : typeof(S.zero)
    #
    # weights: per target a block of nu + 1 offsets into wv
    #
    woh = Vector{J}(undef, k * (nu + 1))
    wvh = VT[]

    for q in 1:k
        Wq = Ws === nothing ? nothing : Ws[q]

        if Wq === nothing
            for x in 1:nu + 1
                woh[(q - 1) * (nu + 1) + x] = length(wvh) + 1
            end
        else
            wo1, wv1, _ = weightcols(D, Yn[q], Wq, S === nothing ? Bool : VT)
            base = length(wvh)

            for x in 1:nu + 1
                woh[(q - 1) * (nu + 1) + x] = base + wo1[x]
            end

            append!(wvh, VT.(wv1[1:wo1[nu + 1] - 1]))
        end
    end

    wo = CuVector{J}(woh)
    wv = CuVector{VT}(isempty(wvh) ? VT[zero(VT)] : wvh)
    cval = CuVector{VT}(undef, 1 << 16)
    xval = CUDA.zeros(J, k * nt)
    nval = 0
    ncol = CUDA.zeros(J, k * nt)
    xtab = CUDA.zeros(J, k * nt)
    first = CUDA.zeros(J, k * nt)
    tab = CuVector{J}(undef, 1 << 20)
    h = CUDA.zeros(J, k * nu)
    cnt = CUDA.zeros(J, 1)
    next = CUDA.zeros(J, 1)
    mp = mlev * k
    rep = CuVector{J}(undef, 2mp + 1)
    reph = Vector{J}(undef, 2mp + 1)
    st = Stage(STAGE[], W, VT)
    icap = max(1 << 16, mp)
    ibag = CuVector{J}(undef, icap); idep = CuVector{J}(undef, icap); ipre = CuVector{J}(undef, icap * W); iq = CuVector{J}(undef, icap)
    nbag = CuVector{J}(undef, icap); ndep = CuVector{J}(undef, icap); npre = CuVector{J}(undef, icap * W); niq = CuVector{J}(undef, icap)
    icnt = CuVector{J}(undef, icap); iscan = CuVector{J}(undef, icap)
    abort = CUDA.zeros(J, icap); aflag = CUDA.zeros(J, 1); status = Vector{J}(undef, 1)
    ntab = 0
    sh = Vector(Pc.s)
    ncolh = zeros(Int, k * nt)
    dead = falses(k)
    ntyp = [count(==(u), Yn[1].vtx) for u in 1:Yn[1].cod.nu]
    nkern = 0
    nround = 0
    ev1 = CuEvent(); ev2 = CuEvent()
    tker = 0.0
    ntJ = J(nt); nuJ = J(nu)
    t0 = time_ns()

    for l in eachindex(up)
        lev = up[l]
        nb = length(lev)
        np = nb * k
        #
        # the items of the level: (t, q) for every target, target-major
        #
        hb = Vector{J}(undef, np); hq = Vector{J}(undef, np); hp = Vector{J}(undef, np)

        for q in 1:k, (j, t) in enumerate(lev)
            e = (q - 1) * nb + j
            hb[e] = t; hq[e] = q; hp[e] = (q - 1) * nt + t
        end

        bagsd = CuVector{J}(hp)
        copyto!(ibag, 1, hb, 1, np)
        copyto!(iq, 1, hq, 1, np)
        fill!(view(idep, 1:np), zero(J))
        nitem = np
        round = 0
        #
        # estimated work: the sizes of the child tables, or for a leaf the
        # number of assignments of its separator
        #
        work = 0

        for t in lev
            cstrt = D.tree.xadj[t]
            cstop = D.tree.xadj[t + one(I)] - one(I)

            if cstrt <= cstop
                for j in cstrt:cstop, q in 1:k
                    work += ncolh[(q - 1) * nt + D.tree.adjncy[j]]
                end
            else
                w = 1

                for kk in 1:sh[t]
                    w = min(w * ntyp[Pc.vtx[Pc.xpos[t] + kk]], minwork)
                end

                work += k * w
            end
        end

        resplit = false
        nsplit = 0
        bud = budget

        while true
            expand = !resplit && nitem < nitems && round < maxround && work >= minwork && any(t -> sh[t] > round, lev)
            mode = expand || resplit ? 1 : 0
            #
            # a final round runs with a step budget while the level is
            # still small; items that exceed it are split and run again
            #
            kb = resplit ? one(J) : mode == 0 && nsplit < maxsplit && nitem < 4nitems ? J(bud) : zero(J)
            R = 0

            while true
                @cuda threads=256 blocks=cld(np, 256) zerokernel!(ncol, bagsd, J(np), cnt, next)
                isnothing(stats) || CUDA.record(ev1)
                @cuda threads=threads blocks=min(blocks, cld(nitem, threads)) searchkernel!(P, T, tab, xtab, ncol, h, ibag, idep, ipre, J(W), J(nitem), next,
                    st.rkey, st.rdep, st.rval, J(st.cap), cnt, scr, J(maxm), J(maxo), J(mode), icnt, kb, abort, aflag, iq, ntJ, nuJ,
                    S, wo, wv, cval, xval, st.rvv)
                isnothing(stats) || CUDA.record(ev2)
                @cuda threads=256 blocks=cld(np + 1, 256) reportkernel!(rep, ncol, bagsd, J(np), cnt)
                copyto!(reph, 1, rep, 1, np + 1)
                nkern += 1
                isnothing(stats) || (tker += CUDA.elapsed(ev1, ev2))
                R = Int(reph[np + 1])
                R <= st.cap && break
                st = Stage(max(R, 2st.cap), W, VT)
                STAGE[] = max(STAGE[], st.cap)
            end

            nround += 1

            if mode == 0 && kb > zero(J)
                copyto!(status, aflag)

                if isone(status[1])
                    aflag .= zero(J)
                    resplit = true
                    nsplit += 1
                    bud *= 4
                    continue
                end
            end
            #
            # a target with an empty table has no homomorphism
            #
            for e in 1:np
                if iszero(reph[e])
                    dead[hq[e]] = true
                end
            end

            if all(dead)
                if !isnothing(stats)
                    stats[:fwd] = (time_ns() - t0) / 1e9; stats[:ntab] = ntab; stats[:nkern] = nkern; stats[:levels] = length(up); stats[:rounds] = nround; stats[:kernel] = tker
                end

                return keep ? nothing : S === nothing ? fill(nothing, k) : [(S.zero, nothing) for _ in 1:k]
            end

            accumulate!(+, view(iscan, 1:nitem), view(icnt, 1:nitem))

            if mode == 1
                if R > icap
                    icap = max(R, 2icap)
                    nbag = CuVector{J}(undef, icap); ndep = CuVector{J}(undef, icap); npre = CuVector{J}(undef, icap * W); niq = CuVector{J}(undef, icap)
                end

                @cuda threads=256 blocks=cld(R, 256) gatherkernel!(iscan, st.rkey, st.rdep, st.rval, J(W), J(R), ibag, P, tab, xtab, first, nbag, ndep, npre, J(1), iq, niq, ntJ, st.rvv, cval, xval, S)
                ibag, nbag = nbag, ibag; idep, ndep = ndep, idep; ipre, npre = npre, ipre; iq, niq = niq, iq

                if length(nbag) < icap
                    nbag = CuVector{J}(undef, icap); ndep = CuVector{J}(undef, icap); npre = CuVector{J}(undef, icap * W); niq = CuVector{J}(undef, icap)
                end

                if length(icnt) < icap
                    icnt = CuVector{J}(undef, icap); iscan = CuVector{J}(undef, icap)
                end

                if length(abort) < icap
                    abort = CUDA.zeros(J, icap)
                end

                nitem = R
                round += resplit ? 0 : 1
                resplit = false
            else
                acc = 0

                for e in 1:np
                    tq = hp[e]
                    ncolh[tq] = reph[e]
                    reph[e] = ntab + 1
                    reph[np + e] = acc
                    ntab += ncolh[tq] * sh[hb[e]]
                    acc += ncolh[tq]
                end

                if ntab > length(tab)
                    tab2 = CuVector{J}(undef, max(ntab, 2length(tab)))
                    copyto!(tab2, 1, tab, 1, length(tab))
                    tab = tab2
                end

                if nval + acc > length(cval)
                    cval2 = CuVector{VT}(undef, max(nval + acc, 2length(cval)))
                    copyto!(cval2, 1, cval, 1, length(cval))
                    cval = cval2
                end

                copyto!(rep, 1, reph, 1, 2np)
                @cuda threads=256 blocks=cld(np, 256) placekernel!(xtab, first, bagsd, J(np), rep, xval, J(nval))
                nval += acc
                R > 0 && @cuda threads=256 blocks=cld(R, 256) gatherkernel!(iscan, st.rkey, st.rdep, st.rval, J(W), J(R), ibag, P, tab, xtab, first, nbag, ndep, npre, J(0), iq, niq, ntJ, st.rvv, cval, xval, S)
                break
            end
        end
    end

    CUDA.synchronize()
    t1 = time_ns()

    if keep
        if !isnothing(stats)
            stats[:plan] = tpl / 1e9; stats[:setup] = (t0 - tst) / 1e9; stats[:fwd] = (t1 - t0) / 1e9; stats[:rounds] = nround
        end

        return (; tab, xtab, ncol, Pc, yord=yords[1], Y=Yn[1])
    end
    #
    # backward, for the live targets
    #
    live = [q for q in 1:k if !dead[q]]
    dnflat = prep.dnflat
    vals = fill(S === nothing ? zero(VT) : S.zero, k)

    if S !== nothing
        xvh = Array(xval)

        for q in live
            vals[q] = Array(view(cval, xvh[(q - 1) * nt + nt] + 1:xvh[(q - 1) * nt + nt] + 1))[1]
        end

        live = [q for q in live if vals[q] != S.zero]

        if !witness
            return [(vals[q], nothing) for q in 1:k]
        end
    end

    for l in 1:length(dnptr) - 1
        lev = view(dnflat, dnptr[l]:dnptr[l + 1] - 1)
        nb = length(lev)
        np = nb * length(live)
        hb = Vector{J}(undef, np); hq = Vector{J}(undef, np); hd = Vector{J}(undef, np)

        for (qi, q) in enumerate(live), (j, t) in enumerate(lev)
            e = (qi - 1) * nb + j
            hb[e] = t; hq[e] = q; hd[e] = sh[t]
        end

        copyto!(ibag, 1, hb, 1, np)
        copyto!(iq, 1, hq, 1, np)
        copyto!(idep, 1, hd, 1, np)
        next .= zero(J)
        @cuda threads=threads blocks=min(blocks, cld(np, threads)) searchkernel!(P, T, tab, xtab, ncol, h, ibag, idep, ipre, J(W), J(np), next,
            st.rkey, st.rdep, st.rval, J(st.cap), cnt, scr, J(maxm), J(maxo), J(2), icnt, zero(J), abort, aflag, iq, ntJ, nuJ,
            S, wo, wv, cval, xval, st.rvv)
        nkern += 1
    end

    hh = Array(h)
    t2 = time_ns()
    outs = Vector{Union{Nothing, Vector{I}}}(nothing, k)

    for q in live
        out = Vector{I}(undef, nu)

        for r in oneto(nu)
            out[D.res.adjncy[r]] = yords[q][hh[(q - 1) * nu + r] - oys[q]]
        end

        outs[q] = out
    end

    if !isnothing(stats)
        stats[:plan] = tpl / 1e9; stats[:setup] = (t0 - tst) / 1e9; stats[:fwd] = (t1 - t0) / 1e9; stats[:bwd] = (t2 - t1) / 1e9; stats[:ntab] = ntab; stats[:nkern] = nkern
        stats[:levels] = length(up); stats[:rounds] = nround; stats[:kernel] = tker
        if haskey(stats, :wanttab)
            stats[:ncol] = Array(ncol); stats[:tab] = Array(tab)[1:ntab]; stats[:xtab] = Array(xtab)
        end
    end

    return S === nothing ? outs : [(vals[q], outs[q]) for q in 1:k]
end

include("contractgpu.jl")

#
# Pick a method: contraction when every separator domain fits in 64
# assignments, else level-synchronous search when the tree is shallow,
# else the CPU.
#
function auto(D::Decomposition{I}, Y::Instance{I}; maxlevels=256, kw...) where {I}
    h = contract(D, Y; kw...)

    if h !== :wide
        return h
    end

    nt = D.tree.nu
    hgt = zeros(Int, nt)

    for t in oneto(nt), j in D.tree.xadj[t]:D.tree.xadj[t + one(I)] - one(I)
        hgt[t] = max(hgt[t], hgt[D.tree.adjncy[j]] + 1)
    end

    if hgt[nt] < maxlevels
        return color(D, Y; kw...)
    else
        return CSets.color(D, Y)
    end
end

end # module
