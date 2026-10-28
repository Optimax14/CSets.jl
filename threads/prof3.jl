include("workloads.jl")
using CSets: CSets, Boolean, Threaded, normalize, invvtx, preimages, weights
B = instance(boolean())
c, = miter(10000, adder!, (c, a, b) -> buggy!(c, a, b, 5000)); X = instance(c); D = decomposition(X, support(X); alg=MF())
P = Threaded(1; deep=1)
Y, yord = normalize(B); typ = invvtx(Y); base, pre = preimages(Y, typ); xwgt, wgt = weights(nothing, D, typ, Bool)
L = CSets.localtables(P, Boolean(), D, Y, pre, base, typ, xwgt, wgt)
@time L = CSets.localtables(P, Boolean(), D, Y, pre, base, typ, xwgt, wgt)
nt = D.tree.nu
@time for t in 1:nt; CSets.runs!(D, L, t); end
pos = zeros(Int, X.dom.nu)
@time for t in 1:nt; CSets.match!(D, L, pos, t); end
using Profile
Profile.Allocs.clear(); Profile.Allocs.@profile sample_rate=0.01 CSets.localtables(P, Boolean(), D, Y, pre, base, typ, xwgt, wgt)
r = Profile.Allocs.fetch()
d = Dict{String,Int}()
for a in r.allocs
    fr = [string(f.func, ":", f.line) for f in a.stacktrace if occursin("CSets", string(f.file))]
    k = isempty(fr) ? "?" : fr[1]
    d[k] = get(d, k, 0) + a.size
end
for (k, v) in sort(collect(d), by=-last)[1:min(8, end)]; println(v, " ", k); end
