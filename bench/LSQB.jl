# Copied from csets-showcase/lsqb/LSQB.jl; run() uses the public score API with a kernel.
#
# LSQB (LDBC Labelled Subgraph Query Benchmark) as C-set homomorphism
# counting.
#
# The database is a C-set over a "typed graph" schema: one sort per
# node table (Person, Tag, ...) with no arcs, and one sort per edge
# table (Person_knows_Person, ...) with two arcs, src and tgt, into the
# node sorts. A LSQB query is SELECT count(*) over a join of these
# tables; the join pattern is itself a small C-set X over the same
# schema (one element per table alias), and count(*) is the number of
# homomorphisms X -> Y. Inequalities and anti-joins (NOT EXISTS / LEFT
# JOIN ... IS NULL) become signed sums of homomorphism counts
# (inclusion-exclusion over quotients and extensions of X). LEFT JOINs
# that count (q7) become a derived "optional" edge sort with a null
# row for each unmatched key.
#
module LSQB

using CSets
using CSets: FVector, Span, Instance, saturate
using CliqueTrees: MF

const I = Int64

#
# Node sorts and edge sorts used by the nine queries.
#
const NODES = [:Country, :City, :Person, :Forum, :Message, :Tag, :TagClass]
const EDGES = [
    (:CityCountry, :City, :Country),          # City_isPartOf_Country
    (:PersonCity, :Person, :City),            # Person_isLocatedIn_City
    (:Member, :Forum, :Person),               # Forum_hasMember_Person
    (:Container, :Forum, :Message),           # Forum_containerOf_Post
    (:ReplyPost, :Message, :Message),         # Comment_replyOf_Post
    (:Reply, :Message, :Message),             # Comment_replyOf_Message (posts and comments)
    (:HasTag, :Message, :Tag),                # Message_hasTag_Tag
    (:TagType, :Tag, :TagClass),              # Tag_hasType_TagClass
    (:Knows, :Person, :Person),               # Person_knows_Person, both directions
    (:Creator, :Message, :Person),            # Message_hasCreator_Person
    (:Interest, :Person, :Tag),               # Person_hasInterest_Tag
    (:Likes, :Person, :Message),              # Person_likes_Message
    (:OptReply, :Message, :Message),          # q7: Reply, plus (m, m) for each message with no reply (src is a dummy)
    (:OptLikes, :Message, :Message),          # q7: Likes keyed by message, plus a null row per unliked message
]

struct DB
    nnode::Dict{Symbol, Int}
    src::Dict{Symbol, Vector{Int32}}
    tgt::Dict{Symbol, Vector{Int32}}
    fun::Dict{Symbol, Vector{Int32}}
end

# a functional relation as a vector: the image of each source element
function tofunc(n, s, t)
    v = zeros(Int32, n)
    for i in eachindex(s)
        @assert v[s[i]] == 0 "not functional"
        v[s[i]] = t[i]
    end
    @assert all(>(0), v) "not total"
    return v
end

# ---------------------------------------------------------------- loading

using Mmap

# parse a '|'-separated file of one or two integer columns, after a header line, on all threads
function readcols(path, ncol)
    buf = Mmap.mmap(path)
    n = length(buf)
    p = findfirst(==(UInt8('\n')), buf)
    p === nothing && return [Int64[] for _ in 1:ncol]
    st = p + 1
    nt = Threads.nthreads()
    # chunk boundaries at line starts
    cuts = [st + ((n - st + 1) * k) ÷ nt for k in 0:nt]
    for k in 2:nt
        q = cuts[k]
        while q <= n && buf[q - 1] != UInt8('\n'); q += 1; end
        cuts[k] = q
    end
    cuts[end] = n + 1
    parts = Vector{Vector{Vector{Int64}}}(undef, nt)
    Threads.@threads :static for k in 1:nt
        cols = [Int64[] for _ in 1:ncol]
        q = cuts[k]; stop = cuts[k + 1]
        @inbounds while q < stop
            for c in 1:ncol
                v = 0; neg = false
                if buf[q] == UInt8('-'); neg = true; q += 1; end
                while q < stop && UInt8('0') <= buf[q] <= UInt8('9')
                    v = 10v + (buf[q] - UInt8('0')); q += 1
                end
                push!(cols[c], neg ? -v : v)
                q += 1     # skip '|' or newline
            end
            # skip anything else to the end of the line (e.g. '\r')
            while q < stop && q > 1 && buf[q - 1] != UInt8('\n'); q += 1; end
        end
        parts[k] = cols
    end
    return [reduce(vcat, [parts[k][c] for k in 1:nt]) for c in 1:ncol]
end

readpairs(path) = Tuple(readcols(path, 2))
readids(path) = readcols(path, 1)[1]

# map raw ids to 1..n by sorting
struct IdMap
    ids::Vector{Int64}
end
idmap(raw::Vector{Int64}) = IdMap(sort(raw))
function (m::IdMap)(x::Vector{Int64})
    out = Vector{Int32}(undef, length(x))
    Threads.@threads for i in eachindex(x)
        j = searchsortedfirst(m.ids, x[i])
        @assert j <= length(m.ids) && m.ids[j] == x[i]
        out[i] = j
    end
    return out
end

function load(dir)
    f(n) = joinpath(dir, n * ".csv")
    country = idmap(readids(f("Country"))); city = idmap(readids(f("City")))
    person = idmap(readids(f("Person"))); forum = idmap(readids(f("Forum")))
    tag = idmap(readids(f("Tag"))); tagclass = idmap(readids(f("TagClass")))
    rawc = readids(f("Comment")); rawp = readids(f("Post"))
    message = idmap([rawc; rawp])
    @assert allunique(message.ids)
    nn = Dict(:Country => length(country.ids), :City => length(city.ids), :Person => length(person.ids),
        :Forum => length(forum.ids), :Message => length(message.ids), :Tag => length(tag.ids),
        :TagClass => length(tagclass.ids))
    S = Dict{Symbol, Vector{Int32}}(); T = Dict{Symbol, Vector{Int32}}()
    function edge!(name, file, ms, mt; append=false)
        a, b = readpairs(f(file))
        s = ms(a); t = mt(b)
        if append
            append!(S[name], s); append!(T[name], t)
        else
            S[name] = s; T[name] = t
        end
    end
    edge!(:CityCountry, "City_isPartOf_Country", city, country)
    edge!(:PersonCity, "Person_isLocatedIn_City", person, city)
    edge!(:Member, "Forum_hasMember_Person", forum, person)
    edge!(:Container, "Forum_containerOf_Post", forum, message)
    edge!(:ReplyPost, "Comment_replyOf_Post", message, message)
    S[:Reply] = copy(S[:ReplyPost]); T[:Reply] = copy(T[:ReplyPost])
    edge!(:Reply, "Comment_replyOf_Comment", message, message; append=true)
    edge!(:HasTag, "Comment_hasTag_Tag", message, tag)
    edge!(:HasTag, "Post_hasTag_Tag", message, tag; append=true)
    edge!(:TagType, "Tag_hasType_TagClass", tag, tagclass)
    edge!(:Knows, "Person_knows_Person", person, person)
    S[:Knows], T[:Knows] = [S[:Knows]; T[:Knows]], [T[:Knows]; S[:Knows]]
    edge!(:Creator, "Comment_hasCreator_Person", message, person)
    edge!(:Creator, "Post_hasCreator_Person", message, person; append=true)
    edge!(:Interest, "Person_hasInterest_Tag", person, tag)
    edge!(:Likes, "Person_likes_Comment", person, message)
    edge!(:Likes, "Person_likes_Post", person, message; append=true)
    F = Dict{Symbol, Vector{Int32}}()
    for (f, rel) in ((:creator, :Creator), (:city, :PersonCity), (:country, :CityCountry), (:type, :TagType))
        _, srt, _ = funcinfo(f)
        F[f] = tofunc(nn[srt], S[rel], T[rel])
    end
    db = DB(nn, S, T, F)
    optional!(db)
    return db
end

#
# Outer joins as data: OptReply(m', m) and OptLikes(m', m) have target
# m (the parent / liked message) and a source that is never inspected
# by q7; a message with no reply (like) gets one null row (m, m).
#
function optional!(db::DB)
    nm = db.nnode[:Message]
    for (opt, base, key) in ((:OptReply, :Reply, db.tgt), (:OptLikes, :Likes, db.tgt))
        k = key[base]
        hit = falses(nm)
        hit[k] .= true
        miss = Int32.(findall(!, hit))
        db.src[opt] = [k; miss]       # source: unused by q7, any message
        db.tgt[opt] = [k; miss]
    end
end

# ---------------------------------------------------------------- C-sets

#
# Functional relations (every element has exactly one image) are arcs
# of the node sorts, not edge sorts: a C-set knows its foreign keys.
#
const FUNCS = [
    (:creator, :Message, :Person),     # Message_hasCreator_Person
    (:city, :Person, :City),           # Person_isLocatedIn_City
    (:country, :City, :Country),       # City_isPartOf_Country
    (:type, :Tag, :TagClass),          # Tag_hasType_TagClass
]

#
# The schema restricted to the edge sorts and functions used: node
# sorts (each with its functions), then edge sorts (arcs src, tgt).
#
struct Sub
    nodes::Vector{Symbol}
    edges::Vector{Symbol}
    funcs::Vector{Symbol}
    sort::Dict{Symbol, Int}
    arc::Dict{Symbol, Int}        # arc number of each function
    schema::Span{I}
end

edgeinfo(e) = EDGES[findfirst(x -> x[1] == e, EDGES)]
funcinfo(f) = FUNCS[findfirst(x -> x[1] == f, FUNCS)]

function Sub(edges::Vector{Symbol}, funcs::Vector{Symbol}=Symbol[])
    edges = unique(edges); funcs = unique(funcs)
    nodes = Symbol[]
    for e in edges
        _, s, t = edgeinfo(e)
        s in nodes || push!(nodes, s); t in nodes || push!(nodes, t)
    end
    for f in funcs
        _, s, t = funcinfo(f)
        s in nodes || push!(nodes, s); t in nodes || push!(nodes, t)
    end
    sort = Dict{Symbol, Int}()
    for (i, n) in enumerate(nodes); sort[n] = i; end
    for (i, e) in enumerate(edges); sort[e] = length(nodes) + i; end
    xadj = [1]; adjncy = Int[]; arc = Dict{Symbol, Int}()
    for n in nodes
        for f in funcs
            _, s, t = funcinfo(f)
            if s == n
                push!(adjncy, sort[t]); arc[f] = length(adjncy)
            end
        end
        push!(xadj, length(adjncy) + 1)
    end
    for e in edges
        _, s, t = edgeinfo(e)
        push!(adjncy, sort[s], sort[t])
        push!(xadj, length(adjncy) + 1)
    end
    ns = length(nodes) + length(edges)
    return Sub(nodes, edges, funcs, sort, arc, Span{I}(ns, ns, length(adjncy), FVector{I}(xadj), FVector{I}(adjncy)))
end

function arcs(sub::Sub, e::Symbol)
    c = sub.sort[e]
    a = sub.schema.xadj[c]
    return (a, a + 1)
end

#
# A C-set over the sub-schema: node elements first (grouped by sort,
# each with its function values), then edge elements.
#   fval[f]: the image of each element of f's source sort (local ids)
#
function typedgraph(sub::Sub, nnode::Vector{Int}, fval::Vector, src::Vector, tgt::Vector)
    off = cumsum([0; nnode])
    nN = off[end]
    nE = sum(length, src; init=0)
    n = nN + nE
    nfa = [Base.count(f -> funcinfo(f)[2] == nd, sub.funcs) for nd in sub.nodes]   # arcs per node sort
    xadj = Vector{I}(undef, n + 1)
    vtx = Vector{I}(undef, n)
    xadj[1] = 1
    for (i, k) in enumerate(nnode)
        vtx[off[i] + 1:off[i + 1]] .= i
        for x in off[i] + 1:off[i + 1]
            xadj[x + 1] = xadj[x] + nfa[i]
        end
    end
    nA = xadj[nN + 1] - 1
    adjncy = Vector{I}(undef, nA + 2nE); arc = Vector{I}(undef, nA + 2nE)
    for (j, f) in enumerate(sub.funcs)
        _, s, t = funcinfo(f)
        i = sub.sort[s]; ot = off[sub.sort[t]]
        # position of f among the functions of its sort
        r = Base.count(g -> funcinfo(g)[2] == s, sub.funcs[1:j])
        V = fval[j]
        @assert length(V) == nnode[i]
        Threads.@threads for x in 1:nnode[i]
            pp = xadj[off[i] + x] + r - 1
            adjncy[pp] = ot + V[x]; arc[pp] = sub.arc[f]
        end
    end
    p = 0
    for (j, e) in enumerate(sub.edges)
        _, s, t = edgeinfo(e)
        os = off[sub.sort[s]]; ot = off[sub.sort[t]]
        a1, a2 = arcs(sub, e)
        S = src[j]; T = tgt[j]; m = length(S)
        Threads.@threads for i in 1:m
            q = p + i
            adjncy[nA + 2q - 1] = os + S[i]; arc[nA + 2q - 1] = a1
            adjncy[nA + 2q] = ot + T[i]; arc[nA + 2q] = a2
            vtx[nN + q] = sub.sort[e]
            xadj[nN + q + 1] = nA + 2q + 1
        end
        p += m
    end
    dom = Span{I}(n, n, nA + 2nE, FVector{I}(xadj), FVector{I}(adjncy))
    return Instance{I}(dom, sub.schema, FVector{I}(vtx), FVector{I}(arc))
end

function target(db::DB, sub::Sub)
    return typedgraph(sub, [db.nnode[s] for s in sub.nodes], [db.fun[f] for f in sub.funcs],
        [db.src[e] for e in sub.edges], [db.tgt[e] for e in sub.edges])
end

#
# A pattern: named variables with node sorts, edges (sort, u, v), and
# function values (f, u, v) meaning f(u) = v.
#
struct Pattern
    vars::Vector{Pair{Symbol, Symbol}}
    edges::Vector{Tuple{Symbol, Symbol, Symbol}}
    funcs::Vector{Tuple{Symbol, Symbol, Symbol}}
end
Pattern(vars, edges) = Pattern(vars, edges, Tuple{Symbol, Symbol, Symbol}[])

# identify variable b with a
function quotient(P::Pattern, a::Symbol, b::Symbol)
    vars = filter(p -> p.first != b, P.vars)
    ren(x) = x == b ? a : x
    return Pattern(vars, [(e, ren(u), ren(v)) for (e, u, v) in P.edges], [(f, ren(u), ren(v)) for (f, u, v) in P.funcs])
end

extend(P::Pattern, e) = Pattern(P.vars, [P.edges; e], P.funcs)

function pattern(sub::Sub, P::Pattern)
    idx = Dict{Symbol, Int}()
    nnode = zeros(Int, length(sub.nodes))
    for (v, s) in P.vars
        nnode[sub.sort[s]] += 1
        idx[v] = nnode[sub.sort[s]]
    end
    fval = [zeros(Int32, nnode[sub.sort[funcinfo(f)[2]]]) for f in sub.funcs]
    for (f, u, v) in P.funcs
        fval[findfirst(==(f), sub.funcs)][idx[u]] = idx[v]
    end
    for V in fval
        @assert all(>(0), V) "every variable needs every function of its sort"
    end
    src = [Int32[] for _ in sub.edges]; tgt = [Int32[] for _ in sub.edges]
    for (e, u, v) in P.edges
        j = findfirst(==(e), sub.edges)
        push!(src[j], idx[u]); push!(tgt[j], idx[v])
    end
    return typedgraph(sub, nnode, fval, src, tgt)
end

#
# The support. A variable that is the image of a function of other
# variables is supported where they are (it is determined by them);
# the other variables support themselves; edges are saturated onto
# their endpoints. With no functions this is the Gaifman graph of the
# pattern; with functions, the decomposition is aware of the
# functional dependencies.
#
function support(sub::Sub, P::Pattern, X::Instance; fd=true)
    # element number of each variable in X (node elements are grouped by sort)
    nnode = zeros(Int, length(sub.nodes)); idx = Dict{Symbol, Int}()
    for (v, s) in P.vars
        nnode[sub.sort[s]] += 1; idx[v] = nnode[sub.sort[s]]
    end
    off = cumsum([0; nnode])
    el(v) = off[sub.sort[Dict(P.vars)[v]]] + idx[v]
    pre = Dict{Symbol, Vector{Symbol}}()
    for (f, u, v) in P.funcs
        push!(get!(pre, v, Symbol[]), u)
    end
    roots = [v for (v, _) in P.vars if !fd || !haskey(pre, v)]
    rid = Dict(v => i for (i, v) in enumerate(roots))
    memo = Dict{Symbol, Vector{Int}}()
    function seed(v)
        get!(memo, v) do
            haskey(rid, v) ? [rid[v]] : unique(reduce(vcat, [seed(u) for u in pre[v]]))
        end
    end
    n = X.dom.nu
    lists = [Int[] for _ in 1:n]
    for (v, _) in P.vars
        lists[el(v)] = sort(seed(v))
    end
    xadj = cumsum([1; length.(lists)])
    return saturate(X, Span{I}(n, length(roots), xadj[end] - 1, FVector{I}(xadj), FVector{I}(reduce(vcat, lists; init=Int[]))))
end

#
# Give every variable every function of its sort (in the sub-schema):
# a missing image becomes a fresh variable. It is determined by its
# source, so the number of homomorphisms does not change.
#
function close(sub::Sub, P::Pattern)
    vars = copy(P.vars); funcs = copy(P.funcs)
    i = 1
    while i <= length(vars)
        v, srt = vars[i]
        for f in sub.funcs
            _, s, t = funcinfo(f)
            if s == srt && !any(x -> x[1] == f && x[2] == v, funcs)
                w = Symbol(v, "_", f)
                push!(vars, w => t); push!(funcs, (f, v, w))
            end
        end
        i += 1
    end
    return Pattern(vars, P.edges, funcs)
end

function decompose(sub::Sub, P::Pattern; alg=MF(), fd=true)
    P = close(sub, P)
    X = pattern(sub, P)
    S = support(sub, P, X; fd=fd)
    return CSets.decomposition(X, S; alg=alg)
end

count(D, Y) = CSets.score(CSets.Probability(), nothing, D, Y)
count(K, D, Y) = CSets.score(K, CSets.Probability(), nothing, D, Y)

# ---------------------------------------------------------------- queries

#
# Each query: the edge sorts it uses and a list of (sign, pattern).
#
function query(q::Int)
    if q == 1
        P = Pattern([:co => :Country, :ci => :City, :pe => :Person, :fo => :Forum, :po => :Message,
                :cm => :Message, :tg => :Tag, :tc => :TagClass],
            [(:Member, :fo, :pe), (:Container, :fo, :po), (:ReplyPost, :cm, :po), (:HasTag, :cm, :tg)],
            [(:country, :ci, :co), (:city, :pe, :ci), (:type, :tg, :tc)])
        return [(1, P)]
    elseif q == 2
        P = Pattern([:p1 => :Person, :p2 => :Person, :c => :Message, :p => :Message],
            [(:Knows, :p1, :p2), (:ReplyPost, :c, :p)],
            [(:creator, :c, :p1), (:creator, :p, :p2)])
        return [(1, P)]
    elseif q == 3
        P = Pattern([:k => :Country, :ca => :City, :cb => :City, :cc => :City, :pa => :Person, :pb => :Person, :pc => :Person],
            [(:Knows, :pa, :pb), (:Knows, :pb, :pc), (:Knows, :pc, :pa)],
            [(:country, :ca, :k), (:country, :cb, :k), (:country, :cc, :k),
             (:city, :pa, :ca), (:city, :pb, :cb), (:city, :pc, :cc)])
        return [(1, P)]
    elseif q == 4
        P = Pattern([:m => :Message, :t => :Tag, :p => :Person, :c => :Message, :l => :Person],
            [(:HasTag, :m, :t), (:Reply, :c, :m), (:Likes, :l, :m)],
            [(:creator, :m, :p)])
        return [(1, P)]
    elseif q == 5 || q == 8
        P = Pattern([:m => :Message, :t1 => :Tag, :c => :Message, :t2 => :Tag],
            [(:HasTag, :m, :t1), (:Reply, :c, :m), (:HasTag, :c, :t2)])
        terms = [(1, P), (-1, quotient(P, :t1, :t2))]
        if q == 8   # minus the rows where c also has tag t1
            P2 = extend(P, (:HasTag, :c, :t1))
            append!(terms, [(-1, P2), (1, quotient(P2, :t1, :t2))])
        end
        return terms
    elseif q == 6 || q == 9
        P = Pattern([:p1 => :Person, :p2 => :Person, :p3 => :Person, :t => :Tag],
            [(:Knows, :p1, :p2), (:Knows, :p2, :p3), (:Interest, :p3, :t)])
        terms = [(1, P), (-1, quotient(P, :p1, :p3))]
        if q == 9   # minus the rows where p1 knows p3
            P2 = extend(P, (:Knows, :p1, :p3))
            append!(terms, [(-1, P2), (1, quotient(P2, :p1, :p3))])
        end
        return terms
    elseif q == 7
        P = Pattern([:m => :Message, :t => :Tag, :p => :Person, :c => :Message, :l => :Message],
            [(:HasTag, :m, :t), (:OptReply, :c, :m), (:OptLikes, :l, :m)],
            [(:creator, :m, :p)])
        return [(1, P)]
    end
    error("q$q")
end

funcsorts(terms) = unique([f for (_, P) in terms for (f, _, _) in P.funcs])

function edgesorts(terms)
    es = Symbol[]
    for (_, P) in terms, (e, _, _) in P.edges
        e in es || push!(es, e)
    end
    return es
end

#
# Prepare (target C-set for the sub-schema, decompositions), then count.
#
function prepare(db::DB, q::Int; alg=MF(), fd=true)
    terms = query(q)
    sub = Sub(edgesorts(terms), funcsorts(terms))
    Y = target(db, sub)
    Ds = [(s, decompose(sub, P; alg=alg, fd=fd)) for (s, P) in terms]
    return Y, Ds
end

#
# Count with the public API (the target is relabeled and indexed on
# every call, for every kernel alike).
#
function run(Y, Ds; kernel=:default, nthreads=1)
    total = 0.0
    for (s, D) in Ds
        total += s * CSets.score(CSets.Probability(), nothing, D, Y; kernel, nthreads, strategy=CSets.Eager)
    end
    return total
end

end # module
