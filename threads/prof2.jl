include("workloads.jl")
using CSets: CSets, score, Boolean, Threaded, normalize, invvtx, preimages, weights
B = instance(boolean())
c, = miter(10000, adder!, (c, a, b) -> buggy!(c, a, b, 5000)); X = instance(c); D = decomposition(X, support(X); alg=MF())
P = Threaded(Threads.nthreads(); deep=1)
Y, yord = normalize(B); typ = invvtx(Y); base, pre = preimages(Y, typ); xwgt, wgt = weights(nothing, D, typ, Bool)
for _ in 1:3
    t1 = @elapsed L = CSets.localtables(P, Boolean(), D, Y, pre, base, typ, xwgt, wgt)
    t2 = @elapsed m = CSets.contracttree(P, D, L)
    t3 = @elapsed CSets.contractsolve(P, Boolean(), nothing, D, B, true)
    println("local=$t1 tree=$t2 all(with witness)=$t3")
end
@time CSets.contracttree(P, D, CSets.localtables(P, Boolean(), D, Y, pre, base, typ, xwgt, wgt))
