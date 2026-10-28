include("workloads.jl")
using CSets: CSets, score, Boolean, Threaded
rng = MersenneTwister(3)
nv, E = partialktree(rng, 40, 2, 0.9; bushy=true)
X = graphinstance(nv, E); D = decomposition(X, graphsupport(X); alg=MF()); Y = complete(3)
up, down = CSets.levels(D)
println("nt=", D.tree.nu, " levels=", up.nu)
for l in 1:up.nu
    println(l, ": ", up.adjncy[up.xadj[l]:up.xadj[l+1]-1])
end
for t in 1:D.tree.nu
    println(t, " children ", D.tree.adjncy[D.tree.xadj[t]:D.tree.xadj[t+1]-1])
end
using CSets: normalize, invvtx, preimages, weights, restrict!, gather!, contract!, TaskWorkspace, Piece, solve!, FVector
Yn, yord = normalize(Y); typ = invvtx(Yn); base, pre = preimages(Yn, typ)
ws = TaskWorkspace(D, Bool, false, false)
nt = D.tree.nu
xtab = FVector{Int}(undef, nt); vtab = FVector{Int}(undef, nt); ncol = FVector{Int}(undef, nt)
tab = Vector{Int}(undef, 1024); tval = Vector{Bool}(undef, 1024); ntab = 0
for l in 1:up.nu, t in up.adjncy[up.xadj[l]:up.xadj[l+1]-1]
    pc = Piece{Int}(t, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0)
    try
        solve!(Boolean(), D, Yn, pre, base, typ, nothing, nothing, xtab, vtab, ncol, tab, tval, ws, pc, 0, 0, 0)
    catch e
        println("fail at t=$t level $l"); rethrow()
    end
    s = D.sep.xadj[t+1] - D.sep.xadj[t]
    global tab; if length(tab) < ntab + pc.n * s; resize!(tab, 2(ntab + pc.n * s)); end
    copyto!(tab, ntab + 1, ws.okey, 1, pc.n * s)
    xtab[t] = ntab + 1; vtab[t] = 1; ncol[t] = pc.n; global ntab += pc.n * s
    println("t=$t n=", pc.n)
end
