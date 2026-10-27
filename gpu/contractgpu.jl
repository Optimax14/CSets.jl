#
# Design (d) on the GPU: tree contraction. See contract.jl for the
# sequential reference and GPU_NOTES.md.
#
# Each bag t has a local table Full_t (its consistent assignments,
# ignoring its children) and a separator domain Loc_t (their
# projections). A message is a vector over Loc_t; a chain bag (one
# undone child c) is a matrix M_t : Loc_c × Loc_t; chains collapse by
# matrix products (pointer jumping).
#
# The algebra of messages is a parameter:
#
#   - BoolPacked(): Boolean, 64 entries per UInt64 word (existence,
#     with a witness);
#
#   - Semiring(zero, one, add, mul, atomic): dense entries of the type
#     of zero, for any commutative semiring. `atomic(ptr, x)` must do
#     *ptr = add(*ptr, x) atomically (CUDA.atomic_or! for Booleans,
#     CUDA.atomic_add! for counting, CUDA.atomic_min! for min-plus).
#     contract returns the value at the root (and a witness when
#     witness=true and the semiring is Boolean-like: nonzero = exists).
#
# The value of a column x of Full_t is the product of the messages of
# its done children at its projections, so the message of t is
#
#    v_t[b] = ⊕ { value(x) : proj(x) = b }
#
# and, with one child c undone,
#
#    M_t[a, b] = ⊕ { value(x) : proj_c(x) = a, proj(x) = b }.
#
struct BoolPacked end

struct Semiring{T, A, M, F}
    zero::T
    one::T
    add::A
    mul::M
    atomic::F
end

boolean() = Semiring(zero(UInt32), one(UInt32), |, &, CUDA.atomic_or!)

counting(::Type{T}=Float64) where {T} = Semiring(zero(T), one(T), +, *, CUDA.atomic_add!)

eltype_(::BoolPacked) = UInt64
eltype_(S::Semiring{T}) where {T} = T

# words per message of a bag with L entries
width(::BoolPacked, L) = cld(L, 64)
width(::Semiring, L) = L

@inline function bit(w::UInt64, b)
    return isone((w >> ((b - one(b)) & 63)) & one(UInt64))
end

@inline function setbit(b)
    return one(UInt64) << ((b - one(b)) & 63)
end

@inline function word(b)
    return ((b - one(b)) >> 6) + one(b)
end

# column x of bag t of the local tables
@inline function full(tab, xtab, m, t, x, k)
    @inbounds return tab[xtab[t] + (x - one(J)) * m[t] + k - one(J)]
end

# the bag holding entry g of a CSR-like offset vector off (off[1] = 0)
@inline function owner(off, g, n)
    lo = one(J)
    hi = n

    @inbounds while lo < hi
        mid = (lo + hi + one(J)) >>> one(J)

        if off[mid] < g
            lo = mid
        else
            hi = mid - one(J)
        end
    end

    return lo
end

#
# entry b of the message of bag c
#
@inline function msg(::BoolPacked, v, voff, c, b)
    @inbounds return bit(v[voff[c] + word(b)], b)
end

@inline function msg(::Semiring, v, voff, c, b)
    @inbounds return v[voff[c] + b]
end

@inline function isnz(::BoolPacked, x::Bool)
    return x
end

@inline function isnz(S::Semiring, x)
    return x != S.zero
end

#
# the value of column x of bag t: the product over its done children
# (except skip) of their messages
#
@inline function colval(alg::BoolPacked, xch, chl, xC, cproj, done, v, voff, t, x, skip)
    nch = xch[t + one(J)] - xch[t]

    @inbounds for j in one(J):nch
        c = chl[xch[t] + j - one(J)]

        if c != skip && isone(done[c])
            p = cproj[xC[t] + (x - one(J)) * nch + j]

            if iszero(p) || !msg(alg, v, voff, c, p)
                return false
            end
        end
    end

    return true
end

@inline function colval(alg::Semiring, xch, chl, xC, cproj, done, v, voff, t, x, skip)
    nch = xch[t + one(J)] - xch[t]
    r = alg.one

    @inbounds for j in one(J):nch
        c = chl[xch[t] + j - one(J)]

        if c != skip && isone(done[c])
            p = cproj[xC[t] + (x - one(J)) * nch + j]

            if iszero(p)
                return alg.zero
            end

            r = alg.mul(r, msg(alg, v, voff, c, p))
        end
    end

    return r
end

#
# accumulate x into entry (row, b) of a matrix or message at offset o
# with row stride rs
#
@inline function accum!(::BoolPacked, A, o, rs, row, b, x)
    if x
        CUDA.atomic_or!(pointer(A, o + (row - one(row)) * rs + word(b)), setbit(b))
    end

    return
end

@inline function accum!(S::Semiring, A, o, rs, row, b, x)
    if x != S.zero
        S.atomic(pointer(A, o + (row - one(row)) * rs + b), x)
    end

    return
end

#
# Loc_t: runs of equal separators among the columns of Full_t
#
function lockernel!(tab, xtab, F, m, s, xF, proj, locfirst, L, nt)
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
                locfirst[xF[t] + n] = x
            end

            proj[xF[t] + x] = n
        end

        L[t] = n
    end

    return
end

function cprojkernel!(tab, xtab, F, m, s, L, locfirst, xch, chl, cpoff, cpos, xC, cproj, xbag, xF, nx)
    g = (blockIdx().x - one(J)) * blockDim().x + threadIdx().x

    @inbounds if g <= nx
        t = xbag[g]
        x = g - xF[t]
        nch = xch[t + one(J)] - xch[t]

        for j in one(J):nch
            l = xch[t] + j - one(J)
            c = chl[l]
            lo = one(J)
            hi = L[c]
            r = zero(J)

            while lo <= hi
                mid = (lo + hi) >>> one(J)
                y = locfirst[xF[c] + mid]
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

    return
end

#
# chain bags: undone, exactly one undone child (ptr, at slot chj)
#
function chainkernel!(xch, chl, done, nund, chain, ptr, chj, nt)
    t = (blockIdx().x - one(J)) * blockDim().x + threadIdx().x

    @inbounds if t <= nt
        if iszero(done[t]) && isone(nund[t])
            chain[t] = one(J)

            for j in one(J):xch[t + one(J)] - xch[t]
                c = chl[xch[t] + j - one(J)]

                if iszero(done[c])
                    ptr[t] = c
                    chj[t] = j
                end
            end
        else
            chain[t] = zero(J)
        end
    end

    return
end

function buildkernel!(alg, M, moff, rs, proj, xbag, xF, xch, chl, xC, cproj, done, v, voff, chain, ptr, chj, nx)
    g = (blockIdx().x - one(J)) * blockDim().x + threadIdx().x

    @inbounds if g <= nx
        t = xbag[g]

        if isone(chain[t])
            x = g - xF[t]
            nch = xch[t + one(J)] - xch[t]
            p = cproj[xC[t] + (x - one(J)) * nch + chj[t]]

            if p > zero(J)
                val = colval(alg, xch, chl, xC, cproj, done, v, voff, t, x, ptr[t])
                accum!(alg, M, moff[t], rs[t], p, proj[g], val)
            end
        end
    end

    return
end

#
# one thread per word (packed) or entry (dense) of every matrix:
# M2_t = M_u ⋅ M_t if u = ptr[t] is a chain bag
#
function jumpkernel!(alg, chain, ptr, ptr2, L, M, M2, moff, rs, flag, nt, total)
    g = (Int(blockIdx().x) - 1) * Int(blockDim().x) + Int(threadIdx().x)

    @inbounds if g <= total
        t = owner(moff, g, nt)

        if isone(chain[t])
            e = g - moff[t]                   # entry within M_t
            a = J((e - one(e)) ÷ rs[t]) + one(J)   # row
            w = J(e - (a - one(J)) * rs[t])   # word or column
            u = ptr[t]

            if isone(chain[u])
                if a <= L[ptr[u]]
                    M2[g] = jumpentry(alg, M, moff, rs, L, t, u, a, w)
                else
                    M2[g] = zero(eltype(M2))
                end

                if isone(a) && isone(w)
                    ptr2[t] = ptr[u]
                    flag[one(J)] = one(J)
                end
            else
                M2[g] = M[g]

                if isone(a) && isone(w)
                    ptr2[t] = u
                end
            end
        end
    end

    return
end

@inline function jumpentry(::BoolPacked, M, moff, rs, L, t, u, a, w)
    r = zero(UInt64)
    ou = moff[u] + (a - one(J)) * rs[u]

    @inbounds for k in one(J):rs[u]
        x = M[ou + k]

        while !iszero(x)
            b = J(trailing_zeros(x)) + (k - one(J)) * J(64) + one(J)
            r |= M[moff[t] + (b - one(J)) * rs[t] + w]
            x &= x - one(UInt64)
        end
    end

    return r
end

@inline function jumpentry(S::Semiring, M, moff, rs, L, t, u, a, w)
    r = S.zero
    ou = moff[u] + (a - one(J)) * rs[u]

    @inbounds for b in one(J):L[u]
        x = M[ou + b]

        if x != S.zero
            r = S.add(r, S.mul(x, M[moff[t] + (b - one(J)) * rs[t] + w]))
        end
    end

    return r
end

#
# bags to rake (all children done): zero their messages
#
function rakemarkkernel!(done, nund, newd, v, voff, rs, nt)
    t = (blockIdx().x - one(J)) * blockDim().x + threadIdx().x

    @inbounds if t <= nt
        if iszero(done[t]) && iszero(nund[t])
            newd[t] = one(J)

            for k in one(J):rs[t]
                v[voff[t] + k] = zero(eltype(v))
            end
        else
            newd[t] = zero(J)
        end
    end

    return
end

function rakekernel!(alg, v, voff, rs, proj, xbag, xF, xch, chl, xC, cproj, done, newd, nx)
    g = (blockIdx().x - one(J)) * blockDim().x + threadIdx().x

    @inbounds if g <= nx
        t = xbag[g]

        if isone(newd[t])
            val = colval(alg, xch, chl, xC, cproj, done, v, voff, t, g - xF[t], zero(J))
            accum!(alg, v, voff[t], zero(J), one(J), proj[g], val)
        end
    end

    return
end

function finmarkkernel!(chain, ptr, done, newd, nt)
    t = (blockIdx().x - one(J)) * blockDim().x + threadIdx().x

    @inbounds if t <= nt
        newd[t] = isone(chain[t]) && isone(done[ptr[t]]) ? one(J) : zero(J)
    end

    return
end

# one thread per word (entry) of every message: v_t = v_ptr ⋅ M_t
function finkernel!(alg, newd, ptr, L, v, voff, M, moff, rs, nt, total)
    g = (Int(blockIdx().x) - 1) * Int(blockDim().x) + Int(threadIdx().x)

    @inbounds if g <= total
        t = owner(voff, g, nt)

        if isone(newd[t])
            w = J(g - voff[t])
            v[g] = finentry(alg, v, voff, M, moff, rs, L, ptr[t], t, w)
        end
    end

    return
end

@inline function finentry(::BoolPacked, v, voff, M, moff, rs, L, c, t, w)
    r = zero(UInt64)

    @inbounds for k in one(J):rs[c]
        x = v[voff[c] + k]

        while !iszero(x)
            a = J(trailing_zeros(x)) + (k - one(J)) * J(64) + one(J)
            r |= M[moff[t] + (a - one(J)) * rs[t] + w]
            x &= x - one(UInt64)
        end
    end

    return r
end

@inline function finentry(S::Semiring, v, voff, M, moff, rs, L, c, t, w)
    r = S.zero

    @inbounds for a in one(J):L[c]
        x = v[voff[c] + a]

        if x != S.zero
            r = S.add(r, S.mul(x, M[moff[t] + (a - one(J)) * rs[t] + w]))
        end
    end

    return r
end

function updatekernel!(alg, newd, done, nund, parent, v, voff, rs, L, flag, nt)
    t = (blockIdx().x - one(J)) * blockDim().x + threadIdx().x

    @inbounds if t <= nt && isone(newd[t])
        done[t] = one(J)
        p = parent[t]

        if p > zero(J)
            CUDA.atomic_sub!(pointer(nund, p), one(J))
        end

        empty = true

        for k in one(J):rs[t]
            if v[voff[t] + k] != zero(eltype(v))
                empty = false
            end
        end

        if empty
            flag[one(J)] = one(J)
        end
    end

    return
end

#
# witness
#
function xselkernel!(alg, xsel, v, voff, proj, xbag, xF, xch, chl, xC, cproj, done, nx)
    g = (blockIdx().x - one(J)) * blockDim().x + threadIdx().x

    @inbounds if g <= nx
        t = xbag[g]
        x = g - xF[t]
        b = proj[g]

        if isnz(alg, msg(alg, v, voff, t, b)) && isnz(alg, colval(alg, xch, chl, xC, cproj, done, v, voff, t, x, zero(J)))
            CUDA.atomic_min!(pointer(xsel, xF[t] + b), x)
        end
    end

    return
end

function fkernel!(parent, slot, xch, xC, cproj, xF, L, xsel, f, maxL, nt)
    g = (blockIdx().x - one(J)) * blockDim().x + threadIdx().x
    t = (g - one(J)) ÷ maxL + one(J)
    b = g - (t - one(J)) * maxL

    @inbounds if t <= nt
        p = parent[t]

        if p > zero(J) && b <= L[p]
            nch = xch[p + one(J)] - xch[p]
            x = xsel[xF[p] + b]
            f[g] = x == typemax(J) ? zero(J) : cproj[xC[p] + (x - one(J)) * nch + slot[t]]
        else
            f[g] = zero(J)
        end
    end

    return
end

function fjumpkernel!(anc, anc2, f, g_, root, flag, maxL, nt)
    g = (blockIdx().x - one(J)) * blockDim().x + threadIdx().x
    t = (g - one(J)) ÷ maxL + one(J)
    b = g - (t - one(J)) * maxL

    @inbounds if t <= nt
        a = anc[t]
        o = (t - one(J)) * maxL

        if a > zero(J) && a != root
            y = f[(a - one(J)) * maxL + b]
            g_[g] = iszero(y) ? zero(J) : f[o + y]

            if isone(b)
                anc2[t] = anc[a]
                flag[one(J)] = one(J)
            end
        else
            g_[g] = f[g]

            if isone(b)
                anc2[t] = a
            end
        end
    end

    return
end

function hkernel!(tab, xtab, m, s, ord, xpos, xF, xsel, f, h, br, root, maxL, nt)
    t = (blockIdx().x - one(J)) * blockDim().x + threadIdx().x

    @inbounds if t <= nt
        b = t == root ? br : f[(t - one(J)) * maxL + br]
        x = xsel[xF[t] + b]

        for k in s[t] + one(J):m[t]
            h[ord[xpos[t] + k]] = full(tab, xtab, m, t, x, k)
        end
    end

    return
end

#
# Random-mate compress (linear work): a chain bag t with heads absorbs
# its chain child u = ptr[t] with tails, M_t = M_u ⋅ M_t; u is frozen
# with its matrix and pointer, and resolved after the chain's bottom.
#
@inline function coin(t, r)
    x = UInt32(t) * 0x9e3779b1 + UInt32(r) * 0x85ebca77
    x ⊻= x >> 15
    x *= 0x2c1b3c6d
    x ⊻= x >> 12
    return isone(x & one(UInt32))
end

function absorbkernel!(chain, rem, ptr, absorb, flag, r, nt)
    t = (blockIdx().x - one(J)) * blockDim().x + threadIdx().x

    @inbounds if t <= nt
        absorb[t] = zero(J)

        if isone(chain[t]) && iszero(rem[t])
            u = ptr[t]

            if isone(chain[u])
                flag[one(J)] = one(J)

                if coin(t, r) && !coin(u, r)
                    absorb[t] = one(J)
                end
            end
        end
    end

    return
end

# one thread per (absorbing bag, row): the new row into M2
function absorbrowkernel!(alg, alist, na, ptr, L, M, M2, moff, rs, maxL)
    i = (Int(blockIdx().x) - 1) * Int(blockDim().x) + Int(threadIdx().x)
    k = J((i - 1) ÷ maxL) + one(J)
    a = J(i - (k - one(J)) * maxL)

    @inbounds if k <= na
        t = alist[k]
        u = ptr[t]
        o = moff[t] + (a - one(J)) * rs[t]

        for w in one(J):rs[t]
            if a <= L[ptr[u]]
                M2[o + w] = jumpentry(alg, M, moff, rs, L, t, u, a, w)
            else
                M2[o + w] = zero(eltype(M2))
            end
        end
    end

    return
end

function absorbcopykernel!(alist, na, M, M2, moff, rs, maxL)
    i = (Int(blockIdx().x) - 1) * Int(blockDim().x) + Int(threadIdx().x)
    k = J((i - 1) ÷ maxL) + one(J)
    a = J(i - (k - one(J)) * maxL)

    @inbounds if k <= na
        t = alist[k]
        o = moff[t] + (a - one(J)) * rs[t]

        for w in one(J):rs[t]
            M[o + w] = M2[o + w]
        end
    end

    return
end

function absorbptrkernel!(alist, na, ptr, rem, rptr, r)
    k = (blockIdx().x - one(J)) * blockDim().x + threadIdx().x

    @inbounds if k <= na
        t = alist[k]
        u = ptr[t]
        rem[u] = r
        rptr[u] = ptr[u]
        ptr[t] = ptr[u]
    end

    return
end

# the same, one thread per (bag, row) over all bags (no compaction)
function absorbrowallkernel!(alg, absorb, ptr, L, M, M2, moff, rs, maxL, nt)
    i = (Int(blockIdx().x) - 1) * Int(blockDim().x) + Int(threadIdx().x)
    t = J((i - 1) ÷ maxL) + one(J)
    a = J(i - (t - one(J)) * maxL)

    @inbounds if t <= nt && isone(absorb[t])
        u = ptr[t]
        o = moff[t] + (a - one(J)) * rs[t]

        for w in one(J):rs[t]
            if a <= L[ptr[u]]
                M2[o + w] = jumpentry(alg, M, moff, rs, L, t, u, a, w)
            else
                M2[o + w] = zero(eltype(M2))
            end
        end
    end

    return
end

function absorbcopyallkernel!(absorb, M, M2, moff, rs, maxL, nt)
    i = (Int(blockIdx().x) - 1) * Int(blockDim().x) + Int(threadIdx().x)
    t = J((i - 1) ÷ maxL) + one(J)
    a = J(i - (t - one(J)) * maxL)

    @inbounds if t <= nt && isone(absorb[t])
        o = moff[t] + (a - one(J)) * rs[t]

        for w in one(J):rs[t]
            M[o + w] = M2[o + w]
        end
    end

    return
end

function absorbptrallkernel!(absorb, ptr, rem, rptr, r, nt)
    t = (blockIdx().x - one(J)) * blockDim().x + threadIdx().x

    @inbounds if t <= nt && isone(absorb[t])
        u = ptr[t]
        rem[u] = r
        rptr[u] = ptr[u]
        ptr[t] = ptr[u]
    end

    return
end

function unsplicemarkkernel!(rem, rptr, done, newd, r, nt)
    t = (blockIdx().x - one(J)) * blockDim().x + threadIdx().x

    @inbounds if t <= nt
        newd[t] = rem[t] == r && isone(done[rptr[t]]) ? one(J) : zero(J)
    end

    return
end

function finmark2kernel!(chain, rem, ptr, done, newd, nt)
    t = (blockIdx().x - one(J)) * blockDim().x + threadIdx().x

    @inbounds if t <= nt
        newd[t] = isone(chain[t]) && iszero(rem[t]) && isone(done[ptr[t]]) ? one(J) : zero(J)
    end

    return
end

#
# Everything that depends only on the decomposition.
#
struct ContractPrep{I, P, V}
    D::Decomposition{I}
    Dl::P          # Prepared for D' (every bag its own separator)
    sD::V
    sDh::Vector{J}
    nch::Vector{J}
    xch::V
    chl::V
    parent::V
    slot::V
    cpoff::V
    cpos::V
end

function prepare_contract(D::Decomposition{I}) where {I}
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
    Dl = prepare(Decomposition{I}(X, res, sep, T0))
    Pc = Dl.Pc
    sD = J[D.sep.xadj[t + one(I)] - D.sep.xadj[t] for t in oneto(nt)]
    nch = J[D.tree.xadj[t + one(I)] - D.tree.xadj[t] for t in oneto(nt)]
    xch = J.(collect(D.tree.xadj))
    chl = J[D.tree.adjncy[p] for p in 1:D.tree.xadj[nt + one(I)] - one(I)]
    parent = zeros(J, nt)
    slot = zeros(J, nt)
    cpoff = Vector{J}(undef, length(chl))
    cpos = J[]
    pos = zeros(J, nu)

    for t in oneto(nt)
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

    d(x) = CuVector{J}(isempty(x) ? J[0] : x)
    return ContractPrep(D, Dl, d(sD), sD, nch, d(xch), d(chl), d(parent), d(slot), d(cpoff), d(cpos))
end

function contract(D::Decomposition{I}, Y::Instance{I}; kw...) where {I}
    return contract(prepare_contract(D), Y; kw...)
end

#
# Find a homomorphism D.inst → Y, or nothing, by tree contraction.
# Returns :wide if the matrices would not fit in maxbytes of device
# memory. With a Semiring alg and witness=false, returns the message of
# the root (for counting: the number of homomorphisms).
#
function contract(C::ContractPrep{I}, Y::Instance{I}; alg=BoolPacked(), witness=alg isa BoolPacked,
        maxbytes=CUDA.total_memory() ÷ 4, jump=:mate, batch=8, stats=nothing, kw...) where {I}
    tst = time_ns()
    D = C.D
    X = D.inst
    nt = D.tree.nu
    nu = X.dom.nu
    st0 = Dict{Symbol, Any}()
    R = color(C.Dl, Y; keep=true, stats=st0, kw...)
    tlocal = time_ns()

    if isnothing(R)
        return witness ? nothing : alg.zero
    end

    tab, xtab, Fd, Pc, yord = R.tab, R.xtab, R.ncol, R.Pc, R.yord
    F = Array(Fd)
    nch = C.nch
    xF = Vector{J}(undef, nt)
    xC = Vector{J}(undef, nt)
    nx = Int(sum(F))
    xbag = Vector{J}(undef, nx)
    accF = 0
    accC = 0

    for t in oneto(nt)
        for x in 1:F[t]
            xbag[accF + x] = t
        end

        xF[t] = accF
        xC[t] = accC
        accF += F[t]
        accC += F[t] * nch[t]
    end

    d(x) = CuVector{J}(x)
    md = d(Pc.m); xFd = d(xF); xCd = d(xC); xbagd = d(xbag)
    ordd = d(Pc.ord); xposd = d(Pc.xpos)
    sd, xchd, chld, parentd, slotd, cpoffd, cposd = C.sD, C.xch, C.chl, C.parent, C.slot, C.cpoff, C.cpos
    proj = CuVector{J}(undef, max(nx, 1))
    cproj = CuVector{J}(undef, max(accC, 1))
    locfirst = CuVector{J}(undef, max(nx, 1))
    L = CuVector{J}(undef, nt)
    flag = CUDA.zeros(J, 1)
    th = 256
    bl = cld(nt, th)
    bx = cld(max(nx, 1), th)
    @cuda threads=th blocks=bl lockernel!(tab, xtab, Fd, md, sd, xFd, proj, locfirst, L, J(nt))
    Lh = Array(L)
    maxL = Int(maximum(Lh))
    #
    # storage: M_t has maxL rows (its row domain is the Loc of a
    # descendant) of width(L_t) words; messages have width(L_t)
    #
    E = eltype_(alg)
    rsh = J[width(alg, Lh[t]) for t in 1:nt]
    moffh = zeros(Int, nt + 1)
    voffh = zeros(Int, nt + 1)

    for t in 1:nt
        moffh[t + 1] = moffh[t] + maxL * rsh[t]
        voffh[t + 1] = voffh[t] + rsh[t]
    end

    Mtot = moffh[nt + 1]
    Vtot = voffh[nt + 1]

    if (2Mtot + Vtot) * sizeof(E) + 3 * nt * maxL * sizeof(J) > maxbytes
        return :wide
    end

    @cuda threads=th blocks=bx cprojkernel!(tab, xtab, Fd, md, sd, L, locfirst, xchd, chld, cpoffd, cposd, xCd, cproj, xbagd, xFd, J(nx))
    rs = d(rsh); moff = CuVector{Int}(moffh); voff = CuVector{Int}(voffh)
    #
    # contraction
    #
    done = CUDA.zeros(J, nt); nund = d(nch); v = CUDA.zeros(E, max(Vtot, 1))
    chain = CUDA.zeros(J, nt); ptr = CUDA.zeros(J, nt); ptr2 = CUDA.zeros(J, nt); chj = CUDA.zeros(J, nt)
    M = CuVector{E}(undef, max(Mtot, 1)); M2 = CuVector{E}(undef, max(Mtot, 1))
    newd = CUDA.zeros(J, nt)
    rem = CUDA.zeros(J, nt); rptr = CUDA.zeros(J, nt); absorb = CUDA.zeros(J, nt)
    nmate = 0
    root = J(nt)
    nphase = 0
    njump = 0
    status = Vector{J}(undef, 1)
    bm = cld(max(Mtot, 1), th)
    bv = cld(max(Vtot, 1), th)
    tjump = 0.0

    while true
        nphase += 1
        nphase > nt + 1 && error("contraction did not finish")
        @cuda threads=th blocks=bl chainkernel!(xchd, chld, done, nund, chain, ptr, chj, J(nt))
        fill!(M, zero(E))
        @cuda threads=th blocks=bx buildkernel!(alg, M, moff, rs, proj, xbagd, xFd, xchd, chld, xCd, cproj, done, v, voff, chain, ptr, chj, J(nx))
        tj = time_ns()

        if jump == :wyllie
            while true
                flag .= zero(J)
                @cuda threads=th blocks=bm jumpkernel!(alg, chain, ptr, ptr2, L, M, M2, moff, rs, flag, J(nt), Mtot)
                ptr, ptr2 = ptr2, ptr
                M, M2 = M2, M
                njump += 1
                copyto!(status, flag)
                iszero(status[1]) && break
            end
        else
            fill!(rem, zero(J))
            r = 0
            #
            # rounds in batches of `batch` with one host check per batch;
            # a round with nothing to absorb is a no-op
            #
            ba = cld(nt * maxL, th)

            while true
                flag .= zero(J)

                for _ in 1:batch
                    r += 1
                    @cuda threads=th blocks=bl absorbkernel!(chain, rem, ptr, absorb, flag, J(r + 1000nphase), J(nt))
                    @cuda threads=th blocks=ba absorbrowallkernel!(alg, absorb, ptr, L, M, M2, moff, rs, J(maxL), J(nt))
                    @cuda threads=th blocks=ba absorbcopyallkernel!(absorb, M, M2, moff, rs, J(maxL), J(nt))
                    @cuda threads=th blocks=bl absorbptrallkernel!(absorb, ptr, rem, rptr, J(r), J(nt))
                end

                njump += batch
                copyto!(status, flag)
                iszero(status[1]) && break
            end

            r += 1
            nmate = r - 1
        end

        tjump += time_ns() - tj
        flag .= zero(J)
        @cuda threads=th blocks=bl rakemarkkernel!(done, nund, newd, v, voff, rs, J(nt))
        @cuda threads=th blocks=bx rakekernel!(alg, v, voff, rs, proj, xbagd, xFd, xchd, chld, xCd, cproj, done, newd, J(nx))
        @cuda threads=th blocks=bl updatekernel!(alg, newd, done, nund, parentd, v, voff, rs, L, flag, J(nt))

        if jump == :wyllie
            @cuda threads=th blocks=bl finmarkkernel!(chain, ptr, done, newd, J(nt))
            @cuda threads=th blocks=bv finkernel!(alg, newd, ptr, L, v, voff, M, moff, rs, J(nt), Vtot)
            @cuda threads=th blocks=bl updatekernel!(alg, newd, done, nund, parentd, v, voff, rs, L, flag, J(nt))
        else
            @cuda threads=th blocks=bl finmark2kernel!(chain, rem, ptr, done, newd, J(nt))
            @cuda threads=th blocks=bv finkernel!(alg, newd, ptr, L, v, voff, M, moff, rs, J(nt), Vtot)
            @cuda threads=th blocks=bl updatekernel!(alg, newd, done, nund, parentd, v, voff, rs, L, flag, J(nt))

            for r in nmate:-1:1
                @cuda threads=th blocks=bl unsplicemarkkernel!(rem, rptr, done, newd, J(r), J(nt))
                @cuda threads=th blocks=bv finkernel!(alg, newd, rptr, L, v, voff, M, moff, rs, J(nt), Vtot)
                @cuda threads=th blocks=bl updatekernel!(alg, newd, done, nund, parentd, v, voff, rs, L, flag, J(nt))
            end
        end

        copyto!(status, flag)

        if isone(status[1]) && witness
            stats === nothing || (stats[:phases] = nphase; stats[:jumps] = njump)
            return
        end

        if isone(Array(view(done, nt:nt))[1])
            break
        end
    end

    tfwd = time_ns()

    if !witness
        vr = Array(view(v, voffh[nt] + 1:voffh[nt] + rsh[nt]))
        stats === nothing || (stats[:phases] = nphase; stats[:jumps] = njump)
        return isempty(vr) ? alg.zero : vr[1]
    end
    #
    # witness
    #
    xsel = CUDA.fill(typemax(J), max(nx, 1))
    f = CuVector{J}(undef, nt * maxL); g = CuVector{J}(undef, nt * maxL)
    anc2 = CUDA.zeros(J, nt)
    bf = cld(nt * maxL, th)
    @cuda threads=th blocks=bx xselkernel!(alg, xsel, v, voff, proj, xbagd, xFd, xchd, chld, xCd, cproj, done, J(nx))
    @cuda threads=th blocks=bf fkernel!(parentd, slotd, xchd, xCd, cproj, xFd, L, xsel, f, J(maxL), J(nt))
    anc = copy(parentd)

    while true
        flag .= zero(J)
        @cuda threads=th blocks=bf fjumpkernel!(anc, anc2, f, g, root, flag, J(maxL), J(nt))
        anc, anc2 = anc2, anc
        f, g = g, f
        njump += 1
        copyto!(status, flag)
        iszero(status[1]) && break
    end

    vroot = Array(view(v, voffh[nt] + 1:voffh[nt] + rsh[nt]))
    br = J(alg isa BoolPacked ? trailing_zeros(vroot[1]) + 1 : findfirst(!=(alg.zero), vroot))
    h = CUDA.zeros(J, nu)
    @cuda threads=th blocks=bl hkernel!(tab, xtab, md, sd, ordd, xposd, xFd, xsel, f, h, br, root, J(maxL), J(nt))
    hh = Array(h)
    out = Vector{I}(undef, nu)

    for r in oneto(nu)
        out[D.res.adjncy[r]] = yord[hh[r]]
    end

    tend = time_ns()

    if !isnothing(stats)
        stats[:lplan] = st0[:plan]; stats[:lsetup] = st0[:setup]; stats[:lfwd] = st0[:fwd]
        stats[:local] = (tlocal - tst) / 1e9; stats[:contract] = (tfwd - tlocal) / 1e9; stats[:witness] = (tend - tfwd) / 1e9
        stats[:jumptime] = tjump / 1e9
        stats[:phases] = nphase; stats[:jumps] = njump; stats[:maxL] = maxL; stats[:mbytes] = (2Mtot + Vtot) * sizeof(E)
    end

    return out
end
