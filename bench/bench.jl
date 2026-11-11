# usage: julia --project=bench bench/bench.jl SUITE CASE KERNEL OUT.tsv
#   SUITE lsqb : CASE = sf/q, e.g. 1/6   (LSQB_DATA, LSQB_EXPECTED)
#         hom  : CASE = dataset/pattern, e.g. grqc/cycle4 (HOM_DATA)
#         miter: CASE = kind-n, e.g. bug-10000  (Lawvere and Boolean, MMD ordering)
#         csp  : CASE = workload name, see below
#   KERNEL default | relk | auto
# One process per case and kernel, so the driver can time out the slow ones.
# Prints: suite case kernel semiring decomp_s search_s value [check]
const SUITE, CASE, KERNEL, OUT = ARGS[1], ARGS[2], Symbol(ARGS[3]), ARGS[4]
using CSets: CSets, score, Boolean, Lawvere, Probability, Eager
using CliqueTrees: MF, MMD
using Random
r4(x) = round(x, digits=4)
emit(xs...) = (line = join(xs, '\t'); println(line); flush(stdout); open(io -> println(io, line), OUT, "a"))
const REPS = parse(Int, get(ENV, "REPS", "2"))
function timed(f)
    t = @elapsed v = f()          # includes compilation; the minimum of REPS
    for _ in 2:REPS
        t > 30 && break
        t = min(t, @elapsed f())
    end
    return t, v
end
st(R) = R isa Boolean ? Eager : nothing     # eager for every semiring, so the kernel is used
if SUITE == "lsqb"
    include(joinpath(@__DIR__, "LSQB.jl")); using .LSQB
    sf, q = split(CASE, "/"); q = parse(Int, q)
    exp = Dict{Tuple{String,Int},Int}()
    for ln in eachline(ENV["LSQB_EXPECTED"])
        f = split(ln, '\t'); exp[(f[3], parse(Int, f[4]))] = parse(Int, f[6])
    end
    db = LSQB.load(joinpath(ENV["LSQB_DATA"], "social-network-sf$sf-projected-fk"))
    td = @elapsed (Y, Ds) = LSQB.prepare(db, q)
    ts, v = timed(() -> LSQB.run(Y, Ds; kernel=KERNEL))
    e = get(exp, (String(sf), q), -1)
    emit("lsqb", CASE, KERNEL, "Probability", r4(td), r4(ts), Int(v), Int(v) == e ? "OK" : "MISMATCH expected $e")
elseif SUITE == "hom"
    include(joinpath(@__DIR__, "..", "test", "graphs.jl"))
    function read_edges(path)
        io = open(path); h = split(readline(io)); n = parse(Int, h[1])
        E = Tuple{Int,Int}[]
        for ln in eachline(io); a, b = split(ln); push!(E, (parse(Int, a), parse(Int, b))); end
        close(io); return n, E
    end
    cycle(k) = (k, [[(i, i + 1) for i in 1:k-1]; (k, 1)])
    path(k) = (k + 1, [(i, i + 1) for i in 1:k])
    ladder(k) = (2k, [[(i, i + 1) for i in 1:k-1]; [(k + i, k + i + 1) for i in 1:k-1]; [(i, k + i) for i in 1:k]])
    pats = Dict("path4" => path(4), "cycle3" => cycle(3), "cycle4" => cycle(4), "cycle5" => cycle(5), "cycle6" => cycle(6), "ladder4" => ladder(4))
    ds, pn = split(CASE, "/")
    n, E = read_edges(joinpath(ENV["HOM_DATA"], "$ds.edges")); Y = graph_instance(n, E)
    k, QE = pats[pn]; X = graph_instance(k, QE)
    td = @elapsed D = CSets.decomposition(X, graph_support(X); alg=MF())
    ts, v = timed(() -> score(Probability(), nothing, D, Y; kernel=KERNEL, strategy=Eager))
    emit("hom", CASE, KERNEL, "Probability", r4(td), r4(ts), v)
else
    include(joinpath(@__DIR__, "workloads.jl"))
    if SUITE == "miter"
        kind, n = split(CASE, "-"); n = parse(Int, n)
        c, = miter(n, adder!, kind == "eq" ? adder2! : (c, a, b) -> buggy!(c, a, b, n ÷ 2)); X = instance(c); Y = instance(boolean())
        td = @elapsed D = CSets.decomposition(X, support(X); alg=MMD())
        # satisfiability (a counterexample) and Lawvere with no weights (0 if one exists): both eager
        for (R, W) in ((Lawvere(), nothing), (Boolean(), nothing))
            ts, v = timed(() -> score(R, W, D, Y; kernel=KERNEL, strategy=st(R)))
            emit("miter", CASE, KERNEL, nameof(typeof(R)), r4(td), r4(ts), v)
        end
    else
        name = CASE
        X, S, Y = cspcase(name)
        td = @elapsed D = CSets.decomposition(X, S; alg=MF())
        for R in (Boolean(), Probability(), Lawvere())
            W = R isa Lawvere ? lawvereweights(Y, X) : nothing
            ts, v = timed(() -> score(R, W, D, Y; kernel=KERNEL, strategy=st(R)))
            emit("csp", CASE, KERNEL, nameof(typeof(R)), r4(td), r4(ts), v)
        end
    end
end
