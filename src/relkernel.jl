#
# An output-sensitive local solver for CSets.score: a drop-in for the
# default kernel (CSets.contract!) for large targets.
#
# The default kernel assigns a bag separator-first, so a leaf bag whose
# separator elements have no arcs between them enumerates the product
# of their sorts: a bag {c, p1, p2} with separator (c, p2) costs
# |Y(Message)| x |Y(Person)| before the residual is looked at. That is
# fine for small targets (truth tables, K_q) and fatal for databases.
#
# This kernel assigns the whole bag (residual and separator together)
# in a dynamic order: at every step it picks the unassigned element
# with the fewest candidates, drawn from
#
#   1. the image of an assigned element pointing to it        (1 candidate)
#   2. the elements pointing to two assigned elements          (pair index)
#   3. the preimages of an assigned element it points to       (|preimages|)
#   4. the distinct values of a child table, narrowed by its
#      already-assigned trailing ports                         (|range|)
#   5. all elements of its sort                                (|Y(sort)|)
#
# checks every arc and every complete child key as soon as possible,
# and emits (separator assignment, weight) for every complete
# assignment. The rows are then sorted colexicographically and equal
# keys are summed, which is exactly the table contract! writes. It is
# a worst-case-optimal-style join per bag (generic join with the
# smallest-candidate rule), with Yannakakis-style aggregation between
# bags done by CSets itself.
#
# Score only (no witness): fac must be nothing.
#
module RelKernel

using ..CSets: AbstractSemiring, Boolean, Instance, Span, sone, szero, splus, sprod, image

export relkernel, prefer

#
# Pair index: for a sort with exactly two arcs a1, a2, the preimages of
# y under a1, sorted by their image under a2 (img holds those images).
#
struct PairIndex{I}
    adj::Vector{I}
    img::Vector{I}
end

mutable struct Kernel{I}
    Y::Any
    pairs::Dict{Tuple{I, I}, PairIndex{I}}
    nthreads::Int
    pmap::Matrix{I}               # (a1, a2) -> index into plist, 0 if not built
    plist::Vector{PairIndex{I}}
    # column-oriented arcs: the image of the i-th element of the source
    # sort of arc a is aimg[a][i], and that element is aoff[a] + i
    aimg::Vector{Vector{I}}
    aoff::Vector{I}
end

relkernel(::Type{I}=Int; nthreads=1) where {I} = Kernel{I}(nothing, Dict{Tuple{I, I}, PairIndex{I}}(), nthreads, zeros(I, 0, 0), PairIndex{I}[], Vector{I}[], I[])

#
# Per-target indexes, built on the first call with a new target.
#
function prepare!(K::Kernel{I}, Y, typ) where {I}
    K.Y === Y && return
    K.Y = Y
    empty!(K.pairs)
    na = Y.cod.ne
    K.pmap = zeros(I, na, na)
    empty!(K.plist)
    K.aimg = [I[] for _ in 1:na]
    K.aoff = zeros(I, na)
    for c in 1:Y.cod.nu
        st = typ.xadj[c]; sp = typ.xadj[c + 1] - one(I)
        for a in Y.cod.xadj[c]:Y.cod.xadj[c + 1] - 1
            v = Vector{I}(undef, sp - st + 1)
            r = a - Y.cod.xadj[c]
            Threads.@threads for y in st:sp
                @inbounds v[y - st + 1] = Y.dom.adjncy[Y.dom.xadj[y] + r]
            end
            K.aimg[a] = v
            K.aoff[a] = st - one(I)
        end
    end
    return
end

@inline function aimage(K::Kernel{I}, y, a) where {I}
    @inbounds return K.aimg[a][y - K.aoff[a]]
end

@inline function pairindex(K::Kernel{I}, Y, pre, base, typ, a1::I, a2::I) where {I}
    if K.Y === Y
        @inbounds q = K.pmap[a1, a2]
        q > 0 && return @inbounds K.plist[q]
    end
    return buildpair(K, Y, pre, base, typ, a1, a2)
end

@noinline function buildpair(K::Kernel{I}, Y, pre, base, typ, a1::I, a2::I) where {I}
    prepare!(K, Y, typ)
    P = get!(K.pairs, (a1, a2)) do
        d1 = Y.cod.adjncy[a1]
        lo = base[a1]
        hi = base[a1] + typ.xadj[d1 + one(I)] - typ.xadj[d1] - one(I)
        pstrt = pre.xadj[lo]
        pstop = pre.xadj[hi + one(I)] - one(I)
        adj = pre.adjncy[pstrt:pstop]
        img = similar(adj)
        Threads.@threads for l in lo:hi
            s = pre.xadj[l] - pstrt + one(I)
            e = pre.xadj[l + one(I)] - pstrt
            seg = view(adj, s:e)
            sort!(seg; by = x -> image(Y, x, a2))
            for q in s:e
                img[q] = image(Y, adj[q], a2)
            end
        end
        # offsets are pre.xadj shifted by pstrt - 1
        PairIndex{I}(adj, img)
    end
    push!(K.plist, P)
    K.pmap[a1, a2] = length(K.plist)
    return P
end

#
# The per-call state of the search.
#
struct Bag{I, T}
    m::I
    nr::I
    s::I
    # arcs of each element: out (arc, target position), in (source position, arc)
    xout::Vector{I}; oarc::Vector{I}; otgt::Vector{I}
    xin::Vector{I}; iarc::Vector{I}; isrc::Vector{I}
    # children: ports of child j are cport[xc[j]] ... (key order)
    xc::Vector{I}; cport::Vector{I}
    # children containing element k: kc[xk[k]] ...
    xk::Vector{I}; kc::Vector{I}
    # state
    h::Vector{I}
    asg::Vector{Bool}
    # child j: its trailing ports w, ..., w - clev[j] + 1 are matched,
    # and the matching columns are crl[clev[j] + 1, j] ... crh[clev[j] + 1, j]
    clev::Vector{I}
    crl::Matrix{I}
    crh::Matrix{I}
    ustk::Vector{I}       # undo stack of clev
    stop::Base.RefValue{Int}   # depth at which the search records a frontier instead (m + 1: never)
    acc0::Base.RefValue{T}     # the sum, when the separator is empty
    cap::Base.RefValue{Int}    # compact the row buffer when it reaches cap rows
    # one-port children with a dense index: column of element y is cmap[j][y - cmoff[j]] (0: none)
    cmap::Vector{Vector{Int32}}
    cmoff::Vector{I}
    # iterative search state, per depth
    kd::Vector{I}; cur::Vector{I}; stp::Vector{I}; sk1::Vector{I}; sk2::Vector{I}; aux::Vector{I}
    applied::Vector{Bool}
end

function fresh(B::Bag{I, T}) where {I, T}
    return Bag{I, T}(B.m, B.nr, B.s, B.xout, B.oarc, B.otgt, B.xin, B.iarc, B.isrc, B.xc, B.cport, B.xk, B.kc,
        zeros(I, B.m), falses(B.m), copy(B.clev), copy(B.crl), copy(B.crh), I[], Ref(Int(B.m) + 1), Ref(B.acc0[]), Ref(1 << 22), B.cmap, B.cmoff,
        zeros(I, B.m + 1), zeros(I, B.m + 1), zeros(I, B.m + 1), zeros(I, B.m + 1), zeros(I, B.m + 1), zeros(I, B.m + 1), falses(B.m + 1))
end

function Bag(C, ::Type{T}, ccol) where {T}
    I = typeof(C.nr)
    X = C.inst
    m = X.dom.nu
    nr = C.nr
    xout = Vector{I}(undef, m + 1); oarc = I[]; otgt = I[]
    xout[1] = 1
    for k in 1:m
        for p in X.dom.xadj[k]:X.dom.xadj[k + 1] - 1
            push!(oarc, X.arc[p]); push!(otgt, X.dom.adjncy[p])
        end
        xout[k + 1] = length(oarc) + 1
    end
    cnt = zeros(I, m + 1)
    for z in otgt; cnt[z + 1] += 1; end
    xin = cumsum([one(I); cnt[2:end]])
    iarc = Vector{I}(undef, length(oarc)); isrc = similar(iarc)
    fill_ = copy(xin)
    for k in 1:m, p in xout[k]:xout[k + 1] - 1
        z = otgt[p]
        iarc[fill_[z]] = oarc[p]; isrc[fill_[z]] = k; fill_[z] += 1
    end
    S = C.sep
    nc = S.nu
    xc = Vector{I}(undef, nc + 1); cport = I[]
    xc[1] = 1
    for j in 1:nc
        for p in S.xadj[j]:S.xadj[j + 1] - 1
            push!(cport, S.adjncy[p])
        end
        xc[j + 1] = length(cport) + 1
    end
    kcnt = zeros(I, m + 1)
    for k in cport; kcnt[k + 1] += 1; end
    xk = cumsum([one(I); kcnt[2:end]])
    kc = Vector{I}(undef, length(cport)); f2 = copy(xk)
    for j in 1:nc, p in xc[j]:xc[j + 1] - 1
        k = cport[p]; kc[f2[k]] = j; f2[k] += 1
    end
    wmax = nc == 0 ? 0 : maximum(xc[j + 1] - xc[j] for j in 1:nc)
    crl = zeros(I, wmax + 1, nc); crh = zeros(I, wmax + 1, nc)
    for j in 1:nc
        crl[1, j] = one(I); crh[1, j] = ccol[j]
    end
    return Bag{I, T}(m, nr, m - nr, xout, oarc, otgt, xin, iarc, isrc, xc, cport, xk, kc,
        zeros(I, m), falses(m), zeros(I, nc), crl, crh, I[], Ref(Int(m) + 1), Ref(zero(T)), Ref(1 << 22), [Int32[] for _ in 1:nc], zeros(I, nc),
        zeros(I, m + 1), zeros(I, m + 1), zeros(I, m + 1), zeros(I, m + 1), zeros(I, m + 1), zeros(I, m + 1), falses(m + 1))
end

#
# The narrowed range of child j: columns whose trailing ports (w, w-1,
# ..., i+1) equal the assigned values, where i is the highest
# unassigned port. Returns (lo, hi, i); i = 0 means fully assigned.
#
function crange(B::Bag{I}, j, ckey, ccol, key) where {I}
    pstrt = B.xc[j]; w = B.xc[j + 1] - pstrt
    lo = one(I); hi = ccol[j]
    b = ckey[j]
    i = w
    while i >= 1
        k = B.cport[pstrt + i - 1]
        B.asg[k] || break
        v = B.h[k]
        # first column in lo:hi with port i >= v
        l = lo; r = hi + one(I)
        while l < r
            mid = (l + r) >>> 1
            if key[b + (mid - 1) * w + i - 1] < v
                l = mid + one(I)
            else
                r = mid
            end
        end
        s = l
        r = hi + one(I)
        while l < r
            mid = (l + r) >>> 1
            if key[b + (mid - 1) * w + i - 1] <= v
                l = mid + one(I)
            else
                r = mid
            end
        end
        lo = s; hi = l - one(I)
        lo > hi && return (lo, hi, i)
        i -= 1
    end
    return (lo, hi, i)
end

#
# Full lookup of child j (all ports assigned): the column, or 0.
#
function clookup(B::Bag{I}, j, ckey, ccol, key) where {I}
    lo, hi, i = crange(B, j, ckey, ccol, key)
    (lo > hi) && return zero(I)
    # all ports assigned, so lo == hi (keys are distinct)
    return lo
end

#
# The search: depth-first with the fewest-candidates rule.
#
struct Ctx{I, T, R, Yt, P, Wx, Wt, V1, V2, V3}
    R::R
    Y::Yt
    pre::P
    base::V1
    typ::Span{I}
    vtx::V2
    xwgt::Wx
    wgt::Wt
    ckey::V3
    cval::V3
    ccol::V3
    key::Vector{I}
    val::Vector{T}
    K::Kernel{I}
end

@inline function wzero(cx::Ctx{I, T}, k, y) where {I, T}
    xw = cx.xwgt
    xw === nothing && return false
    l = xw[k]
    l < xw[k + 1] || return false
    return cx.wgt[l + y - cx.typ.xadj[cx.vtx[k]]] == szero(cx.R, eltype(cx.wgt))
end

# narrow the children containing k (always undone by uncheck!);
# returns false if some child has no matching column
@inline function children!(cx::Ctx{I}, B::Bag{I}, k) where {I}
    qstrt = B.xk[k]; qstop = B.xk[k + 1] - 1
    for q in qstrt:qstop
        push!(B.ustk, B.clev[B.kc[q]])
    end
    key = cx.key
    @inbounds for q in qstrt:qstop
        j = B.kc[q]
        pstrt = B.xc[j]; w = B.xc[j + 1] - pstrt
        b = cx.ckey[j]
        lev = B.clev[j]
        if w == 1 && lev == 0 && !isempty(B.cmap[j])
            col = B.cmap[j][B.h[B.cport[pstrt]] - B.cmoff[j]]
            B.clev[j] = one(I)
            B.crl[2, j] = col == 0 ? one(I) : I(col); B.crh[2, j] = col == 0 ? zero(I) : I(col)
            col == 0 && return false
            continue
        end
        while lev < w
            i = w - lev
            kk = B.cport[pstrt + i - 1]
            B.asg[kk] || break
            v = B.h[kk]
            lo = B.crl[lev + 1, j]; hi = B.crh[lev + 1, j]
            l = lo; r = hi + one(I)
            while l < r
                mid = (l + r) >>> 1
                if key[b + (mid - 1) * w + i - 1] < v
                    l = mid + one(I)
                else
                    r = mid
                end
            end
            st = l
            r = hi + one(I)
            while l < r
                mid = (l + r) >>> 1
                if key[b + (mid - 1) * w + i - 1] <= v
                    l = mid + one(I)
                else
                    r = mid
                end
            end
            lev += one(I)
            B.crl[lev + 1, j] = st; B.crh[lev + 1, j] = l - one(I)
            B.clev[j] = lev
            st > l - one(I) && return false
        end
    end
    return true
end

@inline function uncheck!(B::Bag{I}, k) where {I}
    for q in B.xk[k + 1] - 1:-1:B.xk[k]
        B.clev[B.kc[q]] = pop!(B.ustk)
    end
end

#
# A static order of the bag, chosen greedily by estimated fanout from
# the sizes of the sorts of Y (the sources of each element are still
# chosen at run time, smallest first).
#
#
# The fanout model of plan: the estimated number of candidates of
# element k given the assigned set S (a bit mask), and the selectivity
# of the other constraints. Reads only cx.typ, cx.vtx and cx.ccol.
#
function fan(cx, B::Bag{I}, k, S; pairs::Bool=true) where {I}
    sz(c) = Float64(cx.typ.xadj[c + 1] - cx.typ.xadj[c])
    inS(x) = (S >> (x - 1)) & 1 == 1
    n = sz(cx.vtx[k])
    gen = n; sel = 1.0
    nimg = 0
    for p in B.xin[k]:B.xin[k + 1] - 1
        w = B.isrc[p]
        if inS(w)
            nimg += 1
        end
    end
    if nimg > 0
        gen = 1.0
        sel *= (1 / n)^(nimg - 1)
    end
    tz = Float64[]
    for p in B.xout[k]:B.xout[k + 1] - 1
        z = B.otgt[p]
        inS(z) && push!(tz, sz(cx.vtx[z]))
    end
    if !isempty(tz)
        sort!(tz; rev=true)
        if nimg == 0
            if length(tz) >= 2 && pairs
                gen = max(n / (tz[1] * tz[2]), 1e-9)
                rest = tz[3:end]
            elseif length(tz) >= 2
                # no pair index (contract!): scan the preimages of one
                # assigned element and filter by the others
                gen = n / tz[1]
                rest = tz[2:end]
            else
                gen = n / tz[1]
                rest = Float64[]
            end
            for t in rest; sel /= t; end
        else
            for t in tz; sel /= t; end
        end
    end
    # children: generation by a child range, filtering by completed children
    for q in B.xk[k]:B.xk[k + 1] - 1
        j = B.kc[q]
        pstrt = B.xc[j]; w = B.xc[j + 1] - pstrt
        d = Float64(cx.ccol[j]); i = w
        while i >= 1 && inS(B.cport[pstrt + i - 1])
            d /= sz(cx.vtx[B.cport[pstrt + i - 1]]); i -= 1
        end
        if i >= 1 && B.cport[pstrt + i - 1] == k
            d = max(d, 1.0)
            if d < gen
                sel *= gen > 0 ? 1.0 : 1.0
                gen = d
            end
        end
        # if k completes child j, filter by its density
        if all(x -> x == k || inS(x), view(B.cport, pstrt:pstrt + w - 1))
            dens = Float64(cx.ccol[j])
            for x in view(B.cport, pstrt:pstrt + w - 1); dens /= sz(cx.vtx[x]); end
            sel *= min(1.0, dens * (gen == Float64(cx.ccol[j]) ? 1.0 : 1.0))
        end
    end
    return gen, sel
end

function plan(cx, B::Bag{I}) where {I}
    m = Int(B.m)
    m == 0 && return I[], 0.0, 1.0
    if m > 16
        order = greedyplan(cx, B)
        return order, ordercost(cx, B, order)...
    end
    N = 1 << m
    cost = fill(Inf, N); card = zeros(N); prev = zeros(Int, N); last = zeros(Int, N)
    cost[1] = 0.0; card[1] = 1.0
    for S in 0:N - 1
        isfinite(cost[S + 1]) || continue
        for k in 1:m
            (S >> (k - 1)) & 1 == 1 && continue
            gen, sel = fan(cx, B, k, S)
            visits = card[S + 1] * max(gen, 1.0)
            c2 = card[S + 1] * gen * sel
            T = S | (1 << (k - 1))
            nc = cost[S + 1] + visits
            if nc < cost[T + 1] || (nc == cost[T + 1] && c2 < card[T + 1])
                cost[T + 1] = nc; card[T + 1] = c2; prev[T + 1] = S; last[T + 1] = k
            end
        end
    end
    order = I[]
    S = N - 1
    while S != 0
        pushfirst!(order, last[S + 1]); S = prev[S + 1]
    end
    return order, cost[N], card[N]
end

#
# The estimated cost (visited partial assignments) of assigning the bag
# in the given order, under the model of plan.
#
function ordercost(cx, B::Bag{I}, order; pairs::Bool=true) where {I}
    cost = 0.0; card = 1.0; S = 0
    for k in order
        gen, sel = fan(cx, B, k, S; pairs)
        cost += card * max(gen, 1.0)
        card *= gen * sel
        S |= 1 << (k - 1)
    end
    return cost, card
end

#
# For kernel = :auto: should the bag C use this kernel instead of
# contract!? contract! assigns the bag from the end (k = m, ..., 1:
# the separator first, then the residual). Its cost under the fanout
# model of plan is compared with the cost of the planned order; this
# kernel is used when the default would visit more than tau partial
# assignments and ratio times more than the plan (defaults TAU[] and
# RATIO[]).
#
# A cheap bound runs first, with no allocation: the default order's
# cost counting only arcs (an element costs one candidate when an
# assigned element points to it, |Y(k)| / |Y(z)| when it points to an
# assigned z, and |Y(k)| otherwise). Child tables only lower the cost,
# so a bag below tau there keeps the default without building a Bag.
#
const TAU = Ref(1e6)
const RATIO = Ref(4.0)
# decisions of prefer: [bags asked, bags sent to this kernel] (not thread-safe; for diagnostics)
const DECISIONS = zeros(Int, 2)
# the cost of emitting and aggregating one row, in visited assignments
const AGG = Ref(4.0)
# only bags with a target sort at least this large are considered (small,
# dense targets such as K_q or G(40, 0.3) keep contract!, whose
# separator-first order follows the children's key order)
const MINSORT = Ref(2048)

function prefer(C, typ, ccol, ::Type{T}; tau::Float64=TAU[], ratio::Float64=RATIO[]) where {T}
    DECISIONS[1] += 1
    X = C.inst
    m = Int(X.dom.nu)
    (m == 0 || m > 62) && return false
    sz(k) = Float64(typ.xadj[X.vtx[k] + 1] - typ.xadj[X.vtx[k]])
    maximum(sz, 1:m) < MINSORT[] && return false
    cost = 0.0; card = 1.0; S = 0
    for k in m:-1:1
        gen = sz(k)
        for p in X.dom.xadj[k]:X.dom.xadj[k + 1] - 1
            z = X.dom.adjncy[p]
            (S >> (z - 1)) & 1 == 1 && (gen = min(gen, max(sz(k) / sz(z), 1.0)))
        end
        if gen > 1.0
            for x in 1:m
                (S >> (x - 1)) & 1 == 1 || continue
                for p in X.dom.xadj[x]:X.dom.xadj[x + 1] - 1
                    X.dom.adjncy[p] == k && (gen = 1.0)
                end
            end
        end
        cost += card * gen
        card *= gen
        S |= 1 << (k - 1)
        cost > tau && break
    end
    cost <= tau && return false
    B = Bag(C, T, ccol)
    cx = (; typ, vtx=X.vtx, ccol)
    cdef, _ = ordercost(cx, B, m:-1:1; pairs=false)
    cdef <= tau && return false
    order, crel, rows = plan(cx, B)
    #
    # with a separator, this kernel emits a row per complete assignment
    # and aggregates them (a sort or a histogram): charge each row
    #
    if m > Int(C.nr)
        crel += AGG[] * rows
    end
    r = cdef > ratio * crel
    DECISIONS[2] += r
    return r
end

function greedyplan(cx, B::Bag{I}) where {I}
    sz(c) = Float64(cx.typ.xadj[c + 1] - cx.typ.xadj[c])
    asg = falses(B.m)
    order = I[]
    for _ in 1:B.m
        bk = 0; be = Inf
        for k in 1:B.m
            asg[k] && continue
            e = sz(cx.vtx[k])
            for p in B.xin[k]:B.xin[k + 1] - 1
                asg[B.isrc[p]] && (e = 1.0)
            end
            nas = 0
            for p in B.xout[k]:B.xout[k + 1] - 1
                if asg[B.otgt[p]]
                    nas += 1
                    e = min(e, max(1.0, sz(cx.vtx[k]) / sz(cx.vtx[B.otgt[p]])))
                end
            end
            nas >= 2 && (e = min(e, 1.0))
            if e < be
                be = e; bk = k
            end
        end
        asg[bk] = true
        push!(order, bk)
    end
    return order
end

#
# Sort rows colexicographically and sum equal keys into okey / oval.
#
#
# Dense accumulation for separators of one or two elements: the table
# is a histogram over the separator's sorts, read out in colex order.
#
function densereduce!(R, s, rk::Vector{I}, rv::Vector{T}, okey::Vector{I}, oval::Vector{T}, ko::I, vo::I, typ, srt, nd) where {I, T}
    acc = fill(szero(R, T), nd)
    hit = falses(nd)
    o1 = typ.xadj[srt[1]]
    n1 = typ.xadj[srt[1] + 1] - o1
    o2 = s == 2 ? typ.xadj[srt[2]] : zero(I)
    @inbounds for r in eachindex(rv)
        if s == 1
            q = rk[r] - o1 + 1
        else
            q = (rk[2r] - o2) * n1 + (rk[2r - 1] - o1) + 1
        end
        acc[q] = splus(R, acc[q], rv[r])
        hit[q] = true
    end
    n = zero(I)
    need = Base.count(hit)
    length(okey) < ko + need * s && resize!(okey, max(2 * length(okey), ko + need * s))
    !(R isa Boolean) && length(oval) < vo + need && resize!(oval, max(2 * length(oval), vo + need))
    @inbounds for q in 1:nd
        hit[q] || continue
        acc[q] == szero(R, T) && continue
        if s == 1
            okey[ko + n + 1] = o1 + q - 1
        else
            okey[ko + 2n + 1] = o1 + (q - 1) % n1
            okey[ko + 2n + 2] = o2 + (q - 1) ÷ n1
        end
        if !(R isa Boolean)
            oval[vo + n + 1] = acc[q]
        end
        n += one(I)
    end
    return n
end

function reduce!(R, B::Bag{I}, rk::Vector{I}, rv::Vector{T}, okey::Vector{I}, oval::Vector{T}, ko::I, vo::I, typ, cx_sort) where {I, T}
    s = B.s
    nrow = length(rv)
    if s == 0
        nrow == 0 && return zero(I)
        acc = rv[1]
        for r in 2:nrow; acc = splus(R, acc, rv[r]); end
        acc == szero(R, T) && return zero(I)
        if !(R isa Boolean)
            length(oval) < vo + 1 && resize!(oval, max(2 * length(oval), vo + 1))
            oval[vo + 1] = acc
        end
        return one(I)
    end
    if s <= 2
        c1 = cx_sort[1]; n1 = typ.xadj[c1 + 1] - typ.xadj[c1]
        nd = n1
        if s == 2
            c2 = cx_sort[2]; n2 = typ.xadj[c2 + 1] - typ.xadj[c2]
            nd = n1 * n2
        end
        if nd <= 4 * nrow + (1 << 20)
            return densereduce!(R, s, rk, rv, okey, oval, ko, vo, typ, cx_sort, nd)
        end
    end
    lt = function (x, y)
        @inbounds for i in s:-1:1
            a = rk[(x - 1) * s + i]; b = rk[(y - 1) * s + i]
            a != b && return a < b
        end
        return false
    end
    mx = isempty(rk) ? 0 : maximum(rk)
    bits = 64 - leading_zeros(UInt64(mx))
    if s * bits <= 64
        packed = Vector{UInt64}(undef, nrow)
        @inbounds for r in 1:nrow
            v = zero(UInt64)
            for i in s:-1:1
                v = (v << bits) | UInt64(rk[(r - 1) * s + i])
            end
            packed[r] = v
        end
        perm = sortperm(packed)
    elseif s * bits <= 128
        packed2 = Vector{UInt128}(undef, nrow)
        @inbounds for r in 1:nrow
            v = zero(UInt128)
            for i in s:-1:1
                v = (v << bits) | UInt128(rk[(r - 1) * s + i])
            end
            packed2[r] = v
        end
        perm = sortperm(packed2)
    else
        perm = collect(1:nrow)
        sort!(perm; lt = lt)
    end
    n = zero(I)
    r = 1
    while r <= nrow
        x = perm[r]
        acc = rv[x]
        r2 = r + 1
        while r2 <= nrow && !lt(x, perm[r2])
            acc = splus(R, acc, rv[perm[r2]])
            r2 += 1
        end
        if acc != szero(R, T)
            l = ko + n * s
            length(okey) < l + s && resize!(okey, max(2 * length(okey), l + s))
            for i in 1:s
                okey[l + i] = rk[(x - 1) * s + i]
            end
            if !(R isa Boolean)
                length(oval) < vo + n + 1 && resize!(oval, max(2 * length(oval), vo + n + 1))
                oval[vo + n + 1] = acc
            end
            n += one(I)
        end
        r = r2
    end
    return n
end

function (K::Kernel{I})(R::AbstractSemiring, C, Y, pre, base, typ, xwgt, wgt, ckey, cval, ccol,
        key::Vector{I}, val::Vector{T}, okey::Vector{I}, oval::Vector{T}, ko::I, vo::I, fac, fo::I) where {I, T}
    @assert fac === nothing "RelKernel computes scores only"
    prepare!(K, Y, typ)
    B = Bag(C, T, ccol)
    cx = Ctx(R, Y, pre, base, typ, C.inst.vtx, xwgt, wgt, ckey, cval, ccol, key, val, K)
    densemaps!(B, cx)
    rk = I[]; rv = T[]
    srt = [C.inst.vtx[B.nr + i] for i in 1:B.s]
    if B.m > 0
        G = program(cx, B, first(plan(cx, B)))
        B.acc0[] = szero(R, T)
        if K.nthreads > 1
            outk, outv, accs = psearch(cx, B, G, K.nthreads)
            if B.s == 0
                B.acc0[] = foldl((a, b) -> splus(R, a, b), accs; init=szero(R, T))
                return writescalar!(R, B.acc0[], oval, vo), zero(I)
            end
            nrow = sum(length, outv; init=0)
            if 1 <= B.s <= 2
                nd = typ.xadj[srt[1] + 1] - typ.xadj[srt[1]]
                B.s == 2 && (nd *= typ.xadj[srt[2] + 1] - typ.xadj[srt[2]])
                if nd <= 4 * nrow + (1 << 20)
                    return pdensereduce!(R, B.s, outk, outv, okey, oval, ko, vo, typ, srt, nd, K.nthreads), zero(I)
                end
            end
            if B.s >= 1
                return pbucketreduce!(R, Int(B.s), outk, outv, okey, oval, ko, vo, typ, srt, K.nthreads), zero(I)
            end
            rk = isempty(outk) ? I[] : reduce(vcat, outk); rv = isempty(outv) ? T[] : reduce(vcat, outv)
        else
            run!(cx, B, G, 1, rk, rv)
            B.s == 0 && return writescalar!(R, B.acc0[], oval, vo), zero(I)
        end
    else
        B.acc0[] = szero(R, T)
        emit!(cx, B, rk, rv)
        B.s == 0 && return writescalar!(R, B.acc0[], oval, vo), zero(I)
    end
    n = reduce!(R, B, rk, rv, okey, oval, ko, vo, typ, srt)
    return n, zero(I)
end


#
# Dense indexes for one-port children with many columns relative to
# the size of their sort: O(1) lookups instead of binary searches.
#
function densemaps!(B::Bag{I}, cx) where {I}
    for j in eachindex(B.clev)
        pstrt = B.xc[j]
        B.xc[j + 1] - pstrt == 1 || continue
        c = cx.vtx[B.cport[pstrt]]
        o = cx.typ.xadj[c] - one(I); n = cx.typ.xadj[c + 1] - cx.typ.xadj[c]
        cx.ccol[j] * 16 >= n || continue
        M = zeros(Int32, n)
        b = cx.ckey[j]
        @inbounds for q in 1:cx.ccol[j]
            M[cx.key[b + q - 1] - o] = q
        end
        B.cmap[j] = M; B.cmoff[j] = o
    end
end

function writescalar!(R, acc::T, oval::Vector{T}, vo) where {T}
    acc == szero(R, T) && return 0
    if !(R isa Boolean)
        length(oval) < vo + 1 && resize!(oval, max(2 * length(oval), vo + 1))
        oval[vo + 1] = acc
    end
    return 1
end

#
# The program: for each depth d, the element order[d] and where its
# candidates come from, given that order[1:d-1] are assigned.
#
struct Prog{I}
    el::Vector{I}
    imgw::Vector{I}; imga::Vector{I}         # generator: image of imgw under imga (0: none)
    xpre::Vector{I}; prea::Vector{I}; prez::Vector{I}   # out-arcs to assigned elements
    pair::Vector{Bool}                       # first two out-arcs form a binary relation
    cj::Vector{I}; ci::Vector{I}             # generator: port ci of child cj (0: none)
    xchk::Vector{I}; chka::Vector{I}; chkz::Vector{I}; chkd::Vector{Bool}   # checks
    prechk::Vector{I}                        # the check implied by each out-arc generator
end

function program(cx::Ctx{I}, B::Bag{I}, order) where {I}
    m = B.m
    pos = zeros(Int, m)
    for (d, k) in enumerate(order); pos[k] = d; end
    imgw = zeros(I, m); imga = zeros(I, m)
    xpre = I[1]; prea = I[]; prez = I[]; pair = falses(m)
    cj = zeros(I, m); ci = zeros(I, m)
    xchk = I[1]; chka = I[]; chkz = I[]; chkd = Bool[]
    for d in 1:m
        k = order[d]
        for p in B.xin[k]:B.xin[k + 1] - 1
            w = B.isrc[p]
            if w != k && pos[w] < d && imgw[d] == 0
                imgw[d] = w; imga[d] = B.iarc[p]
            end
        end
        np = 0
        for p in B.xout[k]:B.xout[k + 1] - 1
            z = B.otgt[p]
            if z != k && pos[z] < d
                push!(prea, B.oarc[p]); push!(prez, z); np += 1
            end
        end
        push!(xpre, length(prea) + 1)
        c = cx.vtx[k]
        pair[d] = np >= 2 && cx.Y.cod.xadj[c + 1] - cx.Y.cod.xadj[c] == 2
        for q in B.xk[k]:B.xk[k + 1] - 1
            j = B.kc[q]
            pstrt = B.xc[j]; w = B.xc[j + 1] - pstrt
            i = w
            while i >= 1 && pos[B.cport[pstrt + i - 1]] < d
                i -= 1
            end
            if i >= 1 && B.cport[pstrt + i - 1] == k && cj[d] == 0
                cj[d] = j; ci[d] = i
            end
        end
        # checks: every arc between k and an earlier element (or itself)
        for p in B.xout[k]:B.xout[k + 1] - 1
            z = B.otgt[p]
            if pos[z] <= d
                push!(chka, B.oarc[p]); push!(chkz, z); push!(chkd, true)
            end
        end
        for p in B.xin[k]:B.xin[k + 1] - 1
            w = B.isrc[p]
            if w != k && pos[w] < d && !(w == imgw[d] && B.iarc[p] == imga[d])
                push!(chka, B.iarc[p]); push!(chkz, w); push!(chkd, false)
            end
        end
        push!(xchk, length(chka) + 1)
    end
    # build the pair indexes now: the search only reads them (threads)
    for d in 1:m
        if pair[d]
            p = xpre[d]
            pairindex(cx.K, cx.Y, cx.pre, cx.base, cx.typ, prea[p], prea[p + 1])
        end
    end
    prechk = zeros(I, length(prea))
    for d in 1:m, p in xpre[d]:xpre[d + 1] - 1, q in xchk[d]:xchk[d + 1] - 1
        if chkd[q] && chka[q] == prea[p] && chkz[q] == prez[p]
            prechk[p] = q
        end
    end
    return Prog{I}(order, imgw, imga, xpre, prea, prez, pair, cj, ci, xchk, chka, chkz, chkd, prechk)
end

@inline function prerange(cx::Ctx{I}, a, y) where {I}
    @inbounds begin
        l = cx.base[a] + y - cx.typ.xadj[cx.Y.cod.adjncy[a]]
        return cx.pre.xadj[l], cx.pre.xadj[l + 1] - one(I)
    end
end

#
# Emit the current complete assignment.
#
@inline function emit!(cx::Ctx{I, T}, B::Bag{I}, okeys::Vector{I}, ovals::Vector{T}) where {I, T}
    R = cx.R
    w = sone(R, T)
    @inbounds if !(R isa Boolean)
        xw = cx.xwgt
        if xw !== nothing
            for x in 1:B.nr
                l = xw[x]
                if l < xw[x + 1]
                    w = sprod(R, w, cx.wgt[l + B.h[x] - cx.typ.xadj[cx.vtx[x]]])
                end
            end
        end
        for j in eachindex(B.clev)
            w = sprod(R, w, cx.val[cx.cval[j] + B.crl[B.clev[j] + 1, j] - 1])
        end
    end
    if B.s == 0
        B.acc0[] = splus(R, B.acc0[], w)
        return
    end
    @inbounds for i in 1:B.s
        push!(okeys, B.h[B.nr + i])
    end
    push!(ovals, w)
    if length(ovals) >= B.cap[]
        n = compact!(R, Int(B.s), okeys, ovals)
        B.cap[] = max(B.cap[], 2n)
    end
    return
end

#
# Sort a row buffer colexicographically and sum equal keys, in place.
# Returns the number of rows left.
#
function compact!(R, s::Int, rk::Vector{I}, rv::Vector{T}) where {I, T}
    nrow = length(rv)
    perm = colexperm(s, rk, nrow)
    nk = similar(rk); nv = similar(rv)
    n = 0
    r = 1
    @inbounds while r <= nrow
        x = perm[r]
        acc = rv[x]
        r2 = r + 1
        while r2 <= nrow && samekey(s, rk, x, perm[r2])
            acc = splus(R, acc, rv[perm[r2]])
            r2 += 1
        end
        if acc != szero(R, T)
            for i in 1:s; nk[n * s + i] = rk[(x - 1) * s + i]; end
            n += 1
            nv[n] = acc
        end
        r = r2
    end
    resize!(nk, n * s); resize!(nv, n)
    copyto!(resize!(rk, n * s), nk); copyto!(resize!(rv, n), nv)
    return n
end

@inline function samekey(s, rk, x, y)
    @inbounds for i in 1:s
        rk[(x - 1) * s + i] == rk[(y - 1) * s + i] || return false
    end
    return true
end

function colexperm(s::Int, rk::Vector{I}, nrow) where {I}
    mx = isempty(rk) ? 0 : maximum(rk)
    bits = 64 - leading_zeros(UInt64(mx))
    if s * bits <= 64
        packed = Vector{UInt64}(undef, nrow)
        @inbounds for r in 1:nrow
            v = zero(UInt64)
            for i in s:-1:1
                v = (v << bits) | UInt64(rk[(r - 1) * s + i])
            end
            packed[r] = v
        end
        return sortperm(packed)
    elseif s * bits <= 128
        packed2 = Vector{UInt128}(undef, nrow)
        @inbounds for r in 1:nrow
            v = zero(UInt128)
            for i in s:-1:1
                v = (v << bits) | UInt128(rk[(r - 1) * s + i])
            end
            packed2[r] = v
        end
        return sortperm(packed2)
    else
        lt = function (x, y)
            @inbounds for i in s:-1:1
                a = rk[(x - 1) * s + i]; b = rk[(y - 1) * s + i]
                a != b && return a < b
            end
            return false
        end
        perm = collect(1:nrow)
        sort!(perm; lt = lt)
        return perm
    end
end

#
# The search, iterative. Depths below d0 are already assigned. At
# depth d0 the generator can be overridden (kind0 = 5: elements lo0 ...
# hi0 of a sort; kind0 = 4: columns lo0 ... hi0 of the child G.cj[d0]).
#
function run!(cx::Ctx{I, T}, B::Bag{I}, G::Prog{I}, d0, okeys, ovals, kind0=0, lo0=zero(I), hi0=-one(I)) where {I, T}
    m = Int(B.m)
    K = cx.K
    Y = cx.Y
    d = Int(d0)
    fresh = true
    @inbounds while true
        if d > m
            emit!(cx, B, okeys, ovals)
            d -= 1; fresh = false
            d < d0 && break
            continue
        end
        if d == B.stop[]
            for i in 1:d - 1
                push!(okeys, B.h[G.el[i]])
            end
            d -= 1; fresh = false
            d < d0 && break
            continue
        end
        k = G.el[d]
        if fresh
            # the generator of depth d
            B.applied[d] = false
            B.sk1[d] = zero(I); B.sk2[d] = zero(I)
            if d == d0 && kind0 != 0
                B.kd[d] = kind0; B.cur[d] = lo0; B.stp[d] = hi0
            elseif G.imgw[d] != 0
                B.kd[d] = 1; B.aux[d] = aimage(K, B.h[G.imgw[d]], G.imga[d]); B.cur[d] = 1; B.stp[d] = 1
            else
                kind = 5
                c = cx.vtx[k]
                lo = cx.typ.xadj[c]; hi = cx.typ.xadj[c + 1] - one(I)
                skip = zero(I)
                for p in G.xpre[d]:G.xpre[d + 1] - 1
                    l2, h2 = prerange(cx, G.prea[p], B.h[G.prez[p]])
                    if h2 - l2 < hi - lo
                        kind = 3; lo = l2; hi = h2; skip = G.prechk[p]
                    end
                end
                if G.pair[d] && hi - lo >= 8
                    p = G.xpre[d]
                    a1 = G.prea[p]; a2 = G.prea[p + 1]
                    q = K.pmap[a1, a2]
                    P = K.plist[q]
                    l2, h2 = prerange(cx, a1, B.h[G.prez[p]])
                    off = cx.pre.xadj[cx.base[a1]] - one(I)
                    v = B.h[G.prez[p + 1]]
                    img = P.img
                    x = l2 - off; z = h2 - off + one(I)
                    while x < z
                        mid = (x + z) >>> 1
                        if img[mid] < v; x = mid + one(I); else; z = mid; end
                    end
                    st = x; z = h2 - off + one(I)
                    while x < z
                        mid = (x + z) >>> 1
                        if img[mid] <= v; x = mid + one(I); else; z = mid; end
                    end
                    kind = 2; lo = st; hi = x - one(I)
                    B.aux[d] = q; B.sk1[d] = G.prechk[p]; B.sk2[d] = G.prechk[p + 1]
                else
                    B.sk1[d] = skip
                    j = G.cj[d]
                    if j != 0
                        lev = B.clev[j]
                        l2 = B.crl[lev + 1, j]; h2 = B.crh[lev + 1, j]
                        if h2 - l2 < hi - lo
                            kind = 4; lo = l2; hi = h2; B.sk1[d] = zero(I)
                        end
                    end
                end
                B.kd[d] = kind; B.cur[d] = lo; B.stp[d] = hi
            end
            B.asg[k] = true
            fresh = false
        elseif B.applied[d]
            uncheck!(B, k)
            B.applied[d] = false
        end
        # the next valid candidate
        found = false
        kind = B.kd[d]
        while B.cur[d] <= B.stp[d]
            c = B.cur[d]
            if kind == 1
                y = B.aux[d]; B.cur[d] = c + one(I)
            elseif kind == 3
                y = cx.pre.adjncy[c]; B.cur[d] = c + one(I)
            elseif kind == 5
                y = c; B.cur[d] = c + one(I)
            elseif kind == 2
                y = K.plist[B.aux[d]].adj[c]; B.cur[d] = c + one(I)
            else
                j = G.cj[d]; i = G.ci[d]
                w = B.xc[j + 1] - B.xc[j]
                b = cx.ckey[j]
                key = cx.key
                y = key[b + (c - 1) * w + i - 1]
                c2 = c + one(I)
                while c2 <= B.stp[d] && key[b + (c2 - 1) * w + i - 1] == y
                    c2 += one(I)
                end
                B.cur[d] = c2
            end
            wzero(cx, k, y) && continue
            B.h[k] = y
            ok = true
            s1 = B.sk1[d]; s2 = B.sk2[d]
            for p in G.xchk[d]:G.xchk[d + 1] - 1
                (p == s1 || p == s2) && continue
                z = G.chkz[p]
                if G.chkd[p]
                    aimage(K, y, G.chka[p]) == B.h[z] || (ok = false; break)
                else
                    aimage(K, B.h[z], G.chka[p]) == y || (ok = false; break)
                end
            end
            ok || continue
            if children!(cx, B, k)
                B.applied[d] = true
                found = true
                break
            else
                uncheck!(B, k)
            end
        end
        if found
            d += 1; fresh = true
        else
            B.asg[k] = false
            d -= 1; fresh = false
            d < d0 && break
        end
    end
    return
end

#
# Parallel search. The frontier is the set of valid assignments of the
# first D - 1 elements of the order, for the smallest D that gives
# enough work items; each item is replayed and searched on a thread.
#
function psearch(cx::Ctx{I, T}, B0::Bag{I}, G::Prog{I}, nt) where {I, T}
    m = Int(B0.m)
    if m == 1
        rk = I[]; rv = T[]
        run!(cx, B0, G, 1, rk, rv)
        return [rk], [rv], [B0.acc0[]]
    end
    want = 64 * nt
    # depth 1: split the generator of the first element directly
    k1 = G.el[1]
    c = cx.vtx[k1]
    lo = cx.typ.xadj[c]; hi = cx.typ.xadj[c + 1] - one(I)
    kind = 5
    j = G.cj[1]
    if j != 0 && B0.crh[1, j] - B0.crl[1, j] < hi - lo
        kind = 4; lo = B0.crl[1, j]; hi = B0.crh[1, j]
    end
    if hi - lo + 1 >= want
        n = hi - lo + 1
        nch = min(n, 64 * nt)
        bounds = [lo + (n * (q - 1)) ÷ nch for q in 1:nch + 1]
        if kind == 4
            # move the bounds to the starts of runs of equal values
            i = G.ci[1]; w = B0.xc[j + 1] - B0.xc[j]; b = cx.ckey[j]
            keyat(q) = cx.key[b + (q - 1) * w + i - 1]
            for q in 2:nch
                x = bounds[q]
                while x > bounds[q - 1] && x <= hi && keyat(x) == keyat(x - 1)
                    x -= 1
                end
                bounds[q] = max(x, bounds[q - 1])
            end
        end
        outk = [I[] for _ in 1:nch]; outv = [T[] for _ in 1:nch]
        acc = [szero(cx.R, T) for _ in 1:nch]
        Threads.@threads :dynamic for q in 1:nch
            B = fresh(B0)
            B.acc0[] = szero(cx.R, T)
            run!(cx, B, G, 1, outk[q], outv[q], kind, I(bounds[q]), I(bounds[q + 1] - 1))
            acc[q] = B.acc0[]
        end
        return outk, outv, acc
    end
    D = 2
    fr = I[]
    while true
        empty!(fr)
        B0.stop[] = D
        run!(cx, B0, G, 1, fr, T[])
        B0.stop[] = m + 1
        nitem = length(fr) ÷ (D - 1)
        (nitem >= want || D == m) && break
        D += 1
    end
    w = D - 1
    nitem = length(fr) ÷ w
    chunks = collect(Iterators.partition(1:nitem, max(1, cld(nitem, 64nt))))
    outk = [I[] for _ in chunks]; outv = [T[] for _ in chunks]
    acc = [szero(cx.R, T) for _ in chunks]
    Threads.@threads :dynamic for q in eachindex(chunks)
        B = fresh(B0)
        B.acc0[] = szero(cx.R, T)
        for it in chunks[q]
            # replay the frontier assignment
            for i in 1:w
                k = G.el[i]
                B.h[k] = fr[(it - 1) * w + i]
                B.asg[k] = true
                children!(cx, B, k)
            end
            run!(cx, B, G, D, outk[q], outv[q])
            for i in w:-1:1
                k = G.el[i]
                uncheck!(B, k)
                B.asg[k] = false
            end
        end
        acc[q] = B.acc0[]
    end
    return outk, outv, acc
end

#
# Parallel reduction of per-chunk rows by buckets of the most
# significant key (the last separator element): rows are scattered to
# buckets, and each bucket is sorted and summed by one thread.
#
function pbucketreduce!(R, s::Int, outk::Vector{Vector{I}}, outv::Vector{Vector{T}}, okey::Vector{I}, oval::Vector{T}, ko::I, vo::I, typ, srt, nt) where {I, T}
    nc = length(outv)
    tot = sum(length, outv; init=0)
    tot == 0 && return zero(I)
    ol = typ.xadj[srt[s]]
    nl = typ.xadj[srt[s] + 1] - ol
    np = 8 * nt
    span = cld(nl, np)
    cnt = zeros(Int, np, nc)
    Threads.@threads :dynamic for c in 1:nc
        rk = outk[c]
        @inbounds for r in eachindex(outv[c])
            cnt[(rk[r * s] - ol) ÷ span + 1, c] += 1
        end
    end
    off = zeros(Int, np, nc)
    acc = 0
    for p in 1:np, c in 1:nc
        off[p, c] = acc; acc += cnt[p, c]
    end
    gk = Vector{I}(undef, tot * s); gv = Vector{T}(undef, tot)
    Threads.@threads :dynamic for c in 1:nc
        pos = off[:, c]
        rk = outk[c]; rv = outv[c]
        @inbounds for r in eachindex(rv)
            p = (rk[r * s] - ol) ÷ span + 1
            pos[p] += 1
            q = pos[p]
            for i in 1:s; gk[(q - 1) * s + i] = rk[(r - 1) * s + i]; end
            gv[q] = rv[r]
        end
    end
    pst = [off[p, 1] for p in 1:np]; push!(pst, tot)
    bk = Vector{Vector{I}}(undef, np); bv = Vector{Vector{T}}(undef, np)
    Threads.@threads :dynamic for p in 1:np
        k = gk[pst[p] * s + 1:pst[p + 1] * s]; v = gv[pst[p] + 1:pst[p + 1]]
        isempty(v) || compact!(R, s, k, v)
        bk[p] = k; bv[p] = v
    end
    need = sum(length, bv)
    length(okey) < ko + need * s && resize!(okey, max(2 * length(okey), ko + need * s))
    !(R isa Boolean) && length(oval) < vo + need && resize!(oval, max(2 * length(oval), vo + need))
    start = cumsum([0; length.(bv)])
    Threads.@threads :dynamic for p in 1:np
        n = start[p]
        @inbounds for r in eachindex(bv[p])
            for i in 1:s; okey[ko + (n + r - 1) * s + i] = bk[p][(r - 1) * s + i]; end
            !(R isa Boolean) && (oval[vo + n + r] = bv[p][r])
        end
    end
    return I(need)
end

#
# Parallel dense reduction of per-chunk rows (separators of one or two
# elements): rows are scattered by key range, and each range is summed
# by one thread.
#
function pdensereduce!(R, s, outk::Vector{Vector{I}}, outv::Vector{Vector{T}}, okey::Vector{I}, oval::Vector{T}, ko::I, vo::I, typ, srt, nd, nt) where {I, T}
    o1 = typ.xadj[srt[1]]
    n1 = typ.xadj[srt[1] + 1] - o1
    o2 = s == 2 ? typ.xadj[srt[2]] : zero(I)
    np = 4 * nt
    span = cld(nd, np)
    nc = length(outv)
    sum(length, outv; init=0) == 0 && return zero(I)
    cnt = zeros(Int, np, nc)
    qidx = [Vector{Int}(undef, length(outv[c])) for c in 1:nc]
    Threads.@threads :dynamic for c in 1:nc
        rk = outk[c]
        @inbounds for r in eachindex(outv[c])
            q = s == 1 ? rk[r] - o1 + 1 : (rk[2r] - o2) * n1 + (rk[2r - 1] - o1) + 1
            qidx[c][r] = q
            cnt[(q - 1) ÷ span + 1, c] += 1
        end
    end
    # offsets: partition-major, then chunk
    off = zeros(Int, np, nc)
    tot = 0
    for p in 1:np, c in 1:nc
        off[p, c] = tot; tot += cnt[p, c]
    end
    gq = Vector{Int}(undef, tot); gv = Vector{T}(undef, tot)
    Threads.@threads :dynamic for c in 1:nc
        pos = off[:, c]
        @inbounds for r in eachindex(outv[c])
            q = qidx[c][r]; p = (q - 1) ÷ span + 1
            pos[p] += 1
            gq[pos[p]] = q; gv[pos[p]] = outv[c][r]
        end
    end
    pst = [off[p, 1] for p in 1:np]; push!(pst, tot)
    # per partition: dense accumulate its key range, count distinct keys
    accs = Vector{Vector{T}}(undef, np); hits = Vector{BitVector}(undef, np); nk = zeros(Int, np)
    Threads.@threads :dynamic for p in 1:np
        lo = (p - 1) * span + 1; hi = min(p * span, nd)
        len = max(hi - lo + 1, 0)
        acc = fill(szero(R, T), len); hit = falses(len)
        @inbounds for r in pst[p] + 1:pst[p + 1]
            q = gq[r] - lo + 1
            acc[q] = splus(R, acc[q], gv[r]); hit[q] = true
        end
        n = 0
        @inbounds for q in 1:len
            (hit[q] && acc[q] != szero(R, T)) || (hit[q] = false; continue)
            n += 1
        end
        accs[p] = acc; hits[p] = hit; nk[p] = n
    end
    need = sum(nk)
    length(okey) < ko + need * s && resize!(okey, max(2 * length(okey), ko + need * s))
    !(R isa Boolean) && length(oval) < vo + need && resize!(oval, max(2 * length(oval), vo + need))
    start = cumsum([0; nk])
    Threads.@threads :dynamic for p in 1:np
        lo = (p - 1) * span + 1
        n = start[p]
        acc = accs[p]; hit = hits[p]
        @inbounds for q in eachindex(hit)
            hit[q] || continue
            g = lo + q - 1
            if s == 1
                okey[ko + n + 1] = o1 + g - 1
            else
                okey[ko + 2n + 1] = o1 + (g - 1) % n1
                okey[ko + 2n + 2] = o2 + (g - 1) ÷ n1
            end
            if !(R isa Boolean)
                oval[vo + n + 1] = acc[q]
            end
            n += 1
        end
    end
    return I(need)
end

end # module
