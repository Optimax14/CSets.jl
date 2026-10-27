#
# Design (d) on the GPU, for separator domains of at most 64
# assignments: tree contraction with 64 × 64 Boolean matrices, one
# UInt64 per row. See contract.jl for the reference and GPU_NOTES.md.
#
# Every kernel has one thread per bag.
#
const B = 64

@inline function bit(w::UInt64, b)
    return isone((w >> (b - one(b))) & one(UInt64))
end

@inline function setbit(b)
    return one(UInt64) << (b - one(b))
end

# the bag table of D' (the bag is its own separator): column x of bag t
@inline function full(tab, xtab, m, t, x, k)
    @inbounds return tab[xtab[t] + (x - one(J)) * m[t] + k - one(J)]
end

function lockernel!(tab, xtab, F, m, s, xF, proj, locfirst, L, flag, nt)
    t = (blockIdx().x - one(J)) * blockDim().x + threadIdx().x

    @inbounds if t <= nt
        n = zero(J)

        for x in one(J):F[t]
            new = x == one(J)

            if !new
                for k in one(J):s[t]
                    if full(tab, xtab, m, t, x, k) != full(tab, xtab, m, t, x - one(J), k)
                        new = true
                        break
                    end
                end
            end

            if new
                n += one(J)

                if n <= J(B)
                    locfirst[(t - one(J)) * J(B) + n] = x
                end
            end

            proj[xF[t] + x] = n
        end

        L[t] = n

        if n > J(B)
            flag[one(J)] = one(J)
        end
    end

    return
end

function cprojkernel!(tab, xtab, F, m, s, L, locfirst, xch, chl, cpoff, cpos, xC, cproj, nt)
    t = (blockIdx().x - one(J)) * blockDim().x + threadIdx().x

    @inbounds if t <= nt
        nch = xch[t + one(J)] - xch[t]

        for j in one(J):nch
            l = xch[t] + j - one(J)
            c = chl[l]

            for x in one(J):F[t]
                lo = one(J)
                hi = L[c]
                r = zero(J)

                while lo <= hi
                    mid = (lo + hi) >>> one(J)
                    y = locfirst[(c - one(J)) * J(B) + mid]
                    cmp = zero(J)

                    for i in one(J):s[c]
                        a = full(tab, xtab, m, t, x, cpos[cpoff[l] + i - one(J)])
                        b = full(tab, xtab, m, c, y, i)

                        if a != b
                            cmp = b < a ? -one(J) : one(J)
                            break
                        end
                    end

                    if iszero(cmp)
                        r = mid
                        break
                    elseif cmp < zero(J)
                        lo = mid + one(J)
                    else
                        hi = mid - one(J)
                    end
                end

                cproj[xC[t] + (x - one(J)) * nch + j] = r
            end
        end
    end

    return
end

#
# is column x of bag t consistent with its done children?
#
@inline function ok(xch, chl, xC, cproj, done, v, t, x, skip)
    nch = xch[t + one(J)] - xch[t]

    @inbounds for j in one(J):nch
        c = chl[xch[t] + j - one(J)]

        if c != skip && isone(done[c])
            p = cproj[xC[t] + (x - one(J)) * nch + j]

            if iszero(p) || !bit(v[c], p)
                return false
            end
        end
    end

    return true
end

#
# chain bags (one undone child c): M_t[a] = the b reachable from a ∈ Loc_c
#
function chainkernel!(F, xF, proj, xch, chl, xC, cproj, done, nund, v, chain, ptr, M, nt)
    t = (blockIdx().x - one(J)) * blockDim().x + threadIdx().x

    @inbounds if t <= nt
        if iszero(done[t]) && isone(nund[t])
            chain[t] = one(J)
            nch = xch[t + one(J)] - xch[t]
            jc = zero(J)
            c = zero(J)

            for j in one(J):nch
                if iszero(done[chl[xch[t] + j - one(J)]])
                    jc = j
                    c = chl[xch[t] + j - one(J)]
                end
            end

            ptr[t] = c

            for a in one(J):J(B)
                M[(t - one(J)) * J(B) + a] = zero(UInt64)
            end

            for x in one(J):F[t]
                if ok(xch, chl, xC, cproj, done, v, t, x, c)
                    p = cproj[xC[t] + (x - one(J)) * nch + jc]

                    if p > zero(J)
                        M[(t - one(J)) * J(B) + p] |= setbit(proj[xF[t] + x])
                    end
                end
            end
        else
            chain[t] = zero(J)
        end
    end

    return
end

function jumpkernel!(chain, ptr, ptr2, M, M2, flag, nt)
    t = (blockIdx().x - one(J)) * blockDim().x + threadIdx().x

    @inbounds if t <= nt && isone(chain[t])
        u = ptr[t]
        o = (t - one(J)) * J(B)

        if isone(chain[u])
            ou = (u - one(J)) * J(B)

            for a in one(J):J(B)
                w = M[ou + a]
                r = zero(UInt64)

                while !iszero(w)
                    b = J(trailing_zeros(w)) + one(J)
                    r |= M[o + b]
                    w &= w - one(UInt64)
                end

                M2[o + a] = r
            end

            ptr2[t] = ptr[u]
            flag[one(J)] = one(J)
        else
            for a in one(J):J(B)
                M2[o + a] = M[o + a]
            end

            ptr2[t] = u
        end
    end

    return
end

function rakekernel!(F, xF, proj, xch, chl, xC, cproj, done, nund, v, newd, nt)
    t = (blockIdx().x - one(J)) * blockDim().x + threadIdx().x

    @inbounds if t <= nt
        newd[t] = zero(J)

        if iszero(done[t]) && iszero(nund[t])
            r = zero(UInt64)

            for x in one(J):F[t]
                if ok(xch, chl, xC, cproj, done, v, t, x, zero(J))
                    r |= setbit(proj[xF[t] + x])
                end
            end

            v[t] = r
            newd[t] = one(J)
        end
    end

    return
end

function finkernel!(chain, ptr, done, v, M, newd, nt)
    t = (blockIdx().x - one(J)) * blockDim().x + threadIdx().x

    @inbounds if t <= nt
        newd[t] = zero(J)

        if isone(chain[t]) && isone(done[ptr[t]])
            w = v[ptr[t]]
            r = zero(UInt64)

            while !iszero(w)
                a = J(trailing_zeros(w)) + one(J)
                r |= M[(t - one(J)) * J(B) + a]
                w &= w - one(UInt64)
            end

            v[t] = r
            newd[t] = one(J)
        end
    end

    return
end

function updatekernel!(newd, done, nund, parent, v, flag, nt)
    t = (blockIdx().x - one(J)) * blockDim().x + threadIdx().x

    @inbounds if t <= nt && isone(newd[t])
        done[t] = one(J)
        p = parent[t]

        if p > zero(J)
            CUDA.atomic_sub!(pointer(nund, p), one(J))
        end

        if iszero(v[t])
            flag[one(J)] = one(J)
        end
    end

    return
end

function xselkernel!(F, xF, proj, xch, chl, xC, cproj, done, v, xsel, nt)
    t = (blockIdx().x - one(J)) * blockDim().x + threadIdx().x

    @inbounds if t <= nt
        o = (t - one(J)) * J(B)

        for b in one(J):J(B)
            xsel[o + b] = zero(J)
        end

        for x in one(J):F[t]
            b = proj[xF[t] + x]

            if bit(v[t], b) && iszero(xsel[o + b]) && ok(xch, chl, xC, cproj, done, v, t, x, zero(J))
                xsel[o + b] = x
            end
        end
    end

    return
end

function fkernel!(parent, slot, xch, xC, cproj, xsel, f, nt)
    t = (blockIdx().x - one(J)) * blockDim().x + threadIdx().x

    @inbounds if t <= nt
        p = parent[t]
        o = (t - one(J)) * J(B)

        if p > zero(J)
            nch = xch[p + one(J)] - xch[p]

            for b in one(J):J(B)
                x = xsel[(p - one(J)) * J(B) + b]
                f[o + b] = iszero(x) ? zero(J) : cproj[xC[p] + (x - one(J)) * nch + slot[t]]
            end
        end
    end

    return
end

function fjumpkernel!(anc, anc2, f, g, root, flag, nt)
    t = (blockIdx().x - one(J)) * blockDim().x + threadIdx().x

    @inbounds if t <= nt
        a = anc[t]
        o = (t - one(J)) * J(B)

        if a > zero(J) && a != root
            oa = (a - one(J)) * J(B)

            for b in one(J):J(B)
                y = f[oa + b]
                g[o + b] = iszero(y) ? zero(J) : f[o + y]
            end

            anc2[t] = anc[a]
            flag[one(J)] = one(J)
        else
            for b in one(J):J(B)
                g[o + b] = f[o + b]
            end

            anc2[t] = a
        end
    end

    return
end

function hkernel!(tab, xtab, m, s, ord, xpos, xsel, f, v, h, root, nt)
    t = (blockIdx().x - one(J)) * blockDim().x + threadIdx().x

    @inbounds if t <= nt
        br = J(trailing_zeros(v[root])) + one(J)
        b = t == root ? br : f[(t - one(J)) * J(B) + br]
        x = xsel[(t - one(J)) * J(B) + b]

        for k in s[t] + one(J):m[t]
            h[ord[xpos[t] + k]] = full(tab, xtab, m, t, x, k)
        end
    end

    return
end

#
# Find a homomorphism D.inst → Y, or nothing, by tree contraction.
# Returns :wide if some separator domain has more than 64 assignments.
#
function contract(D::Decomposition{I}, Y::Instance{I}; stats=nothing, kw...) where {I}
    tst = time_ns()
    X = D.inst
    nt = D.tree.nu
    nu = X.dom.nu
    #
    # D': every bag is its own separator, no tree
    #
    xadj = Vector{I}(undef, nt + one(I))
    adjncy = Vector{I}(undef, nu + D.sep.ne)
    q = one(I)

    for t in oneto(nt)
        xadj[t] = q

        for x in D.res.xadj[t]:D.res.xadj[t + one(I)] - one(I)
            adjncy[q] = x
            q += one(I)
        end

        for p in D.sep.xadj[t]:D.sep.xadj[t + one(I)] - one(I)
            adjncy[q] = D.sep.adjncy[p]
            q += one(I)
        end
    end

    xadj[nt + one(I)] = q
    sep = Span{I}(nt, nu, q - one(I), FVector{I}(xadj), FVector{I}(adjncy))
    res = Span{I}(nt, nu, nu, FVector{I}(ones(I, nt + one(I))), D.res.adjncy)
    T0 = Span{I}(nt, nt, zero(I), FVector{I}(ones(I, nt + one(I))), FVector{I}(I[]))
    Dl = Decomposition{I}(X, res, sep, T0)
    st0 = Dict{Symbol, Any}()
    R = color(Dl, Y; keep=true, stats=st0, kw...)
    tlocal = time_ns()

    if isnothing(R)
        return
    end

    tab, xtab, Fd, Pc, yord = R.tab, R.xtab, R.ncol, R.Pc, R.yord
    F = Array(Fd)
    #
    # the tree of D
    #
    sD = J[D.sep.xadj[t + one(I)] - D.sep.xadj[t] for t in oneto(nt)]
    nch = J[D.tree.xadj[t + one(I)] - D.tree.xadj[t] for t in oneto(nt)]
    xch = J.(collect(D.tree.xadj))
    chl = J[D.tree.adjncy[p] for p in 1:D.tree.xadj[nt + one(I)] - one(I)]
    parent = zeros(J, nt)
    slot = zeros(J, nt)
    xF = zeros(J, nt)
    xC = zeros(J, nt)
    cpoff = Vector{J}(undef, length(chl))
    cpos = J[]
    pos = zeros(J, nu)
    accF = 0
    accC = 0

    for t in oneto(nt)
        xF[t] = accF
        xC[t] = accC
        accF += F[t]
        accC += F[t] * nch[t]
        po = Pc.xpos[t]

        for k in 1:Pc.m[t]
            pos[Pc.ord[po + k]] = k
        end

        for j in 1:nch[t]
            l = xch[t] + j - 1
            c = chl[l]
            parent[c] = t
            slot[c] = j
            cpoff[l] = length(cpos) + 1
            sstop = D.sep.xadj[c + one(I)] - one(I)

            for i in 1:sD[c]
                push!(cpos, pos[D.sep.adjncy[sstop - i + 1]])
            end
        end
    end

    d(x) = CuVector{J}(x)
    md = d(Pc.m); sd = d(sD); xFd = d(xF); xCd = d(xC); xchd = d(xch); chld = d(chl)
    cpoffd = d(cpoff); cposd = d(isempty(cpos) ? J[0] : cpos); parentd = d(parent); slotd = d(slot)
    ordd = d(Pc.ord); xposd = d(Pc.xpos)
    proj = CuVector{J}(undef, max(accF, 1))
    cproj = CuVector{J}(undef, max(accC, 1))
    locfirst = CuVector{J}(undef, B * nt)
    L = CuVector{J}(undef, nt)
    flag = CUDA.zeros(J, 1)
    th = 256
    bl = cld(nt, th)
    @cuda threads=th blocks=bl lockernel!(tab, xtab, Fd, md, sd, xFd, proj, locfirst, L, flag, J(nt))

    if isone(Array(flag)[1])
        return :wide
    end

    @cuda threads=th blocks=bl cprojkernel!(tab, xtab, Fd, md, sd, L, locfirst, xchd, chld, cpoffd, cposd, xCd, cproj, J(nt))
    #
    # contraction
    #
    done = CUDA.zeros(J, nt); nund = d(nch); v = CUDA.zeros(UInt64, nt)
    chain = CUDA.zeros(J, nt); ptr = CUDA.zeros(J, nt); ptr2 = CUDA.zeros(J, nt)
    M = CuVector{UInt64}(undef, B * nt); M2 = CuVector{UInt64}(undef, B * nt)
    newd = CUDA.zeros(J, nt)
    root = J(nt)
    nphase = 0
    njump = 0
    status = Vector{J}(undef, 1)

    while true
        nphase += 1
        nphase > nt + 1 && error("contraction did not finish")
        @cuda threads=th blocks=bl chainkernel!(Fd, xFd, proj, xchd, chld, xCd, cproj, done, nund, v, chain, ptr, M, J(nt))

        while true
            flag .= zero(J)
            @cuda threads=th blocks=bl jumpkernel!(chain, ptr, ptr2, M, M2, flag, J(nt))
            ptr, ptr2 = ptr2, ptr
            M, M2 = M2, M
            njump += 1
            copyto!(status, flag)
            iszero(status[1]) && break
        end

        flag .= zero(J)
        @cuda threads=th blocks=bl rakekernel!(Fd, xFd, proj, xchd, chld, xCd, cproj, done, nund, v, newd, J(nt))
        @cuda threads=th blocks=bl updatekernel!(newd, done, nund, parentd, v, flag, J(nt))
        @cuda threads=th blocks=bl finkernel!(chain, ptr, done, v, M, newd, J(nt))
        @cuda threads=th blocks=bl updatekernel!(newd, done, nund, parentd, v, flag, J(nt))
        copyto!(status, flag)

        if isone(status[1])
            stats === nothing || (stats[:phases] = nphase; stats[:jumps] = njump)
            return
        end

        if isone(Array(view(done, nt:nt))[1])
            break
        end
    end

    tfwd = time_ns()
    #
    # witness
    #
    xsel = CuVector{J}(undef, B * nt); f = CUDA.zeros(J, B * nt); g = CUDA.zeros(J, B * nt)
    anc = parentd; anc2 = CUDA.zeros(J, nt)
    @cuda threads=th blocks=bl xselkernel!(Fd, xFd, proj, xchd, chld, xCd, cproj, done, v, xsel, J(nt))
    @cuda threads=th blocks=bl fkernel!(parentd, slotd, xchd, xCd, cproj, xsel, f, J(nt))
    anc = copy(parentd)

    while true
        flag .= zero(J)
        @cuda threads=th blocks=bl fjumpkernel!(anc, anc2, f, g, root, flag, J(nt))
        anc, anc2 = anc2, anc
        f, g = g, f
        njump += 1
        copyto!(status, flag)
        iszero(status[1]) && break
    end

    h = CUDA.zeros(J, nu)
    @cuda threads=th blocks=bl hkernel!(tab, xtab, md, sd, ordd, xposd, xsel, f, v, h, root, J(nt))
    hh = Array(h)
    out = Vector{I}(undef, nu)

    for r in oneto(nu)
        out[D.res.adjncy[r]] = yord[hh[r]]
    end

    tend = time_ns()

    if !isnothing(stats)
        stats[:local] = (tlocal - tst) / 1e9; stats[:contract] = (tfwd - tlocal) / 1e9; stats[:witness] = (tend - tfwd) / 1e9
        stats[:phases] = nphase; stats[:jumps] = njump; stats[:maxL] = maximum(Array(L))
    end

    return out
end
