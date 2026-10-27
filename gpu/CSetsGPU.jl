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

function target(Y::Instance{I}) where {I}
    ny = Y.dom.nu; nf = Y.dom.ne; na = Y.cod.ne
    ysrc = Vector{I}(undef, nf)

    for y in oneto(ny), p in Y.dom.xadj[y]:Y.dom.xadj[y + one(I)] - one(I)
        ysrc[p] = y
    end

    typ = invvtx(Y)
    base = Vector{I}(undef, na + one(I))
    base[1] = 1

    for a in oneto(na)
        d = Y.cod.adjncy[a]
        base[a + 1] = base[a] + typ.xadj[d + 1] - typ.xadj[d]
    end

    ykey = Vector{I}(undef, nf)

    for e in oneto(nf)
        a = Y.arc[e]
        ykey[e] = base[a] + Y.dom.adjncy[e] - typ.xadj[Y.cod.adjncy[a]]
    end

    pre = invmap(base[na + 1] - one(I), nf, ykey)
    prea = [ysrc[pre.adjncy[p]] for p in oneto(nf)]
    v(x) = CuArray{J}(collect(x))
    return Target(v(Y.dom.xadj), v(Y.dom.adjncy), v(Y.vtx), v(Y.cod.xadj), v(Y.cod.adjncy),
        v(typ.xadj), v(base), v(pre.xadj), v(prea))
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
        rkey, rdep, rval, cap, cnt, scr, maxm, maxo, mode, icnt)
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
        m = P.m[t]
        s = P.s[t]
        po = P.xpos[t]
        pp = po + t - one_
        #
        # scratch: kind, arg, cur, stop, hp at k, rlo, rhi at slot o
        #
        KIND = zero(J); ARG = J(maxm); CUR = J(2maxm); STOP = J(3maxm); HP = J(4maxm)
        RLO = J(5maxm); RHI = J(5maxm + maxo)
        # sc(f, k) = scr[(f + k - 1) * nthr + tid]

        if mode == J(1)
            if d >= s
                emit!(rkey, rdep, rval, cap, cnt, ncol, ipre, W, i, zero(J), t, d, d, scr, nthr, tid, HP, true)
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
            scr[(RHI + o - one_) * nthr + tid] = ncol[c]
        end

        for k in one_:d
            if mode == J(2)
                v = h[P.ord[po + k]]
            else
                v = ipre[(i - one_) * W + k]
            end

            scr[(HP + k - one_) * nthr + tid] = v
            narrow!(P, tab, xtab, scr, nthr, tid, RLO, RHI, pp, k, v, zero(J))
        end

        if mstop <= d
            if mode != J(2)
                emit!(rkey, rdep, rval, cap, cnt, ncol, ipre, W, i, zero(J), t, E, E, scr, nthr, tid, HP, false)
                icnt[i] = one_
            end

            continue
        end

        kmin = d + one_
        k = kmin
        fresh = true
        seq = zero(J)

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
                        l = Y.base[a] + scr[(HP + P.bpos[po + k] - one_) * nthr + tid] - Y.typx[Y.coda[a]]
                        cu = Y.prex[l]
                        st = Y.prex[l + one_] - one_
                    else
                        u = P.vtx[po + k]
                        cu = Y.typx[u]
                        st = Y.typx[u + one_] - one_
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
                    b = xtab[l] + ii - one_
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

                if valid && P.xnar[pp + k] < P.xnar[pp + k + one_]
                    skip = kd == J(4) ? ag : zero(J)
                    valid = narrow!(P, tab, xtab, scr, nthr, tid, RLO, RHI, pp, k, y, skip)
                end

                if valid
                    found = true
                    break
                end
            end

            scr[(CUR + k - one_) * nthr + tid] = cu

            if found
                if k < mstop
                    k += one_
                    fresh = true
                elseif mode == J(2)
                    for kk in s + one_:m
                        h[P.ord[po + kk]] = scr[(HP + kk - one_) * nthr + tid]
                    end

                    break
                else
                    emit!(rkey, rdep, rval, cap, cnt, ncol, ipre, W, i, seq, t, E, E, scr, nthr, tid, HP, false)
                    seq += one_

                    if iszero(E)
                        break
                    end

                    k = E
                end
            else
                k -= one_
            end
        end

        if mode != J(2)
            icnt[i] = seq
        end
    end

    return
end

@inline function narrow!(P, tab, xtab, scr, nthr, tid, RLO, RHI, pp, k, v, skip)
    one_ = one(J)

    @inbounds for j in P.xnar[pp + k]:P.xnar[pp + k + one_] - one_
        if j == skip
            continue
        end

        c = P.narc[j]
        i = P.nari[j]
        o = P.naro[j]
        s = P.s[c]
        b = xtab[c] + i - one_
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

@inline function emit!(rkey, rdep, rval, cap, cnt, ncol, ipre, W, i, seq, t, E, d, scr, nthr, tid, HP, passthrough)
    one_ = one(J)
    r = CUDA.atomic_add!(pointer(cnt), one_) + one_
    CUDA.atomic_add!(pointer(ncol, t), one_)

    @inbounds if r <= cap
        rkey[r] = (UInt64(i) << 32) | UInt64(seq)
        rdep[r] = d

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
# Record rec, the seq-th column of item i, has rank
#
#    iscan[i - 1] + seq + 1
#
# in the round's output: items in order, each item's columns in order.
#
function gatherkernel!(iscan, rkey, rdep, rval, W, R, ibag, P, tab, xtab, first, nbag, ndep, npre, mode)
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

        if mode == zero(J)
            #
            # column r - first[t] of the table of t
            #
            s = P.s[t]
            b = xtab[t] + (r - one_ - first[t]) * s

            for k in one_:s
                tab[b + k - one_] = rval[(rec - one_) * W + k]
            end
        else
            #
            # a new item
            #
            nbag[r] = t
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

function placekernel!(xtab, first, bags, n, buf)
    r = (blockIdx().x - one(J)) * blockDim().x + threadIdx().x

    @inbounds if r <= n
        xtab[bags[r]] = buf[r]
        first[bags[r]] = buf[n + r]
    end

    return
end

mutable struct Stage
    cap::Int
    rkey::CuVector{UInt64}
    rdep::CuVector{J}
    rval::CuVector{J}
end

function Stage(cap, W)
    return Stage(cap, CuVector{UInt64}(undef, cap), CuVector{J}(undef, cap), CuVector{J}(undef, cap * W))
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
#   - minwork:  ... unless the level is smaller than this
#
function color(D::Decomposition{I}, Y::Instance{I}; nitems=1 << 15, maxround=6, minwork=256, threads=128, stats=nothing, keep=false) where {I}
    tst = time_ns()
    Y, yord = normalize(Y)
    X = D.inst
    nu = X.dom.nu
    nt = D.tree.nu
    ny = Y.dom.nu
    Pc, maxm, maxo = plan(D)
    tpl = time_ns() - tst
    P = adapt(CuArray, Pc)
    T = target(Y)
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

    dev = device()
    nsm = attribute(dev, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    blocks = nsm * 8
    nthr = blocks * threads
    scr = CuVector{J}(undef, (5maxm + 2maxo) * nthr)
    ncol = CUDA.zeros(J, nt)
    xtab = CUDA.zeros(J, nt)
    first = CUDA.zeros(J, nt)
    tab = CuVector{J}(undef, 1 << 20)
    h = CUDA.zeros(J, nu)
    cnt = CUDA.zeros(J, 1)
    next = CUDA.zeros(J, 1)
    rep = CuVector{J}(undef, 2mlev + 1)
    reph = Vector{J}(undef, 2mlev + 1)
    st = Stage(1 << 16, W)
    icap = max(1 << 16, mlev)
    ibag = CuVector{J}(undef, icap); idep = CuVector{J}(undef, icap); ipre = CuVector{J}(undef, icap * W)
    nbag = CuVector{J}(undef, icap); ndep = CuVector{J}(undef, icap); npre = CuVector{J}(undef, icap * W)
    icnt = CuVector{J}(undef, icap); iscan = CuVector{J}(undef, icap)
    ntab = 0
    sh = Vector(Pc.s)
    ncolh = zeros(Int, nt)
    ntyp = [count(==(u), Y.vtx) for u in 1:Y.cod.nu]
    nkern = 0
    nround = 0
    ev1 = CuEvent(); ev2 = CuEvent()
    tker = 0.0
    t0 = time_ns()

    for l in eachindex(up)
        lev = up[l]
        nb = length(lev)
        bagsd = view(upd, upptr[l]:upptr[l + 1] - 1)
        copyto!(ibag, 1, upd, upptr[l], nb)
        fill!(view(idep, 1:nb), zero(J))
        nitem = nb
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
                for j in cstrt:cstop
                    work += ncolh[D.tree.adjncy[j]]
                end
            else
                w = 1

                for k in 1:sh[t]
                    w = min(w * ntyp[Pc.vtx[Pc.xpos[t] + k]], minwork)
                end

                work += w
            end
        end

        while true
            expand = nitem < nitems && round < maxround && work >= minwork && any(t -> sh[t] > round, lev)
            mode = expand ? 1 : 0
            R = 0

            while true
                @cuda threads=256 blocks=cld(nb, 256) zerokernel!(ncol, bagsd, J(nb), cnt, next)
                isnothing(stats) || CUDA.record(ev1)
                @cuda threads=threads blocks=min(blocks, cld(nitem, threads)) searchkernel!(P, T, tab, xtab, ncol, h, ibag, idep, ipre, J(W), J(nitem), next,
                    st.rkey, st.rdep, st.rval, J(st.cap), cnt, scr, J(maxm), J(maxo), J(mode), icnt)
                isnothing(stats) || CUDA.record(ev2)
                @cuda threads=256 blocks=cld(nb + 1, 256) reportkernel!(rep, ncol, bagsd, J(nb), cnt)
                copyto!(reph, 1, rep, 1, nb + 1)
                nkern += 1
                isnothing(stats) || (tker += CUDA.elapsed(ev1, ev2))
                R = Int(reph[nb + 1])
                R <= st.cap && break
                st = Stage(max(R, 2st.cap), W)
            end

            nround += 1

            for k in 1:nb
                if iszero(reph[k])
                    if !isnothing(stats)
                        stats[:fwd] = (time_ns() - t0) / 1e9; stats[:ntab] = ntab; stats[:nkern] = nkern; stats[:levels] = length(up); stats[:rounds] = nround; stats[:kernel] = tker
                    end

                    return
                end
            end

            accumulate!(+, view(iscan, 1:nitem), view(icnt, 1:nitem))

            if expand
                if R > icap
                    icap = max(R, 2icap)
                    nbag = CuVector{J}(undef, icap); ndep = CuVector{J}(undef, icap); npre = CuVector{J}(undef, icap * W)
                end

                @cuda threads=256 blocks=cld(R, 256) gatherkernel!(iscan, st.rkey, st.rdep, st.rval, J(W), J(R), ibag, P, tab, xtab, first, nbag, ndep, npre, J(1))
                ibag, nbag = nbag, ibag; idep, ndep = ndep, idep; ipre, npre = npre, ipre

                if length(nbag) < icap
                    nbag = CuVector{J}(undef, icap); ndep = CuVector{J}(undef, icap); npre = CuVector{J}(undef, icap * W)
                end

                if length(icnt) < icap
                    icnt = CuVector{J}(undef, icap); iscan = CuVector{J}(undef, icap)
                end

                nitem = R
                round += 1
            else
                acc = 0

                for k in 1:nb
                    t = lev[k]
                    ncolh[t] = reph[k]
                    reph[k] = ntab + 1
                    reph[nb + k] = acc
                    ntab += ncolh[t] * sh[t]
                    acc += ncolh[t]
                end

                if ntab > length(tab)
                    tab2 = CuVector{J}(undef, max(ntab, 2length(tab)))
                    copyto!(tab2, 1, tab, 1, length(tab))
                    tab = tab2
                end

                copyto!(rep, 1, reph, 1, 2nb)
                @cuda threads=256 blocks=cld(nb, 256) placekernel!(xtab, first, bagsd, J(nb), rep)
                R > 0 && @cuda threads=256 blocks=cld(R, 256) gatherkernel!(iscan, st.rkey, st.rdep, st.rval, J(W), J(R), ibag, P, tab, xtab, first, nbag, ndep, npre, J(0))
                break
            end
        end
    end

    CUDA.synchronize()
    t1 = time_ns()

    if keep
        return (; tab, xtab, ncol, Pc, yord, Y)
    end

    for l in eachindex(down)
        nb = dnptr[l + 1] - dnptr[l]
        copyto!(ibag, 1, dnd, dnptr[l], nb)
        copyto!(idep, 1, dns, dnptr[l], nb)
        @cuda threads=256 blocks=cld(nb, 256) zerokernel!(rep, view(dnd, dnptr[l]:dnptr[l + 1] - 1), J(0), cnt, next)
        @cuda threads=threads blocks=min(blocks, cld(nb, threads)) searchkernel!(P, T, tab, xtab, ncol, h, ibag, idep, ipre, J(W), J(nb), next,
            st.rkey, st.rdep, st.rval, J(st.cap), cnt, scr, J(maxm), J(maxo), J(2), icnt)
        nkern += 1
    end

    hh = Array(h)
    t2 = time_ns()
    out = Vector{I}(undef, nu)

    for r in oneto(nu)
        out[D.res.adjncy[r]] = yord[hh[r]]
    end

    if !isnothing(stats)
        stats[:plan] = tpl / 1e9; stats[:setup] = (t0 - tst) / 1e9; stats[:fwd] = (t1 - t0) / 1e9; stats[:bwd] = (t2 - t1) / 1e9; stats[:ntab] = ntab; stats[:nkern] = nkern
        stats[:levels] = length(up); stats[:rounds] = nround; stats[:kernel] = tker
        stats[:ncol] = Array(ncol); stats[:tab] = Array(tab)[1:ntab]; stats[:xtab] = Array(xtab)
    end

    return out
end

include("contractgpu.jl")

end # module
