include("workloads.jl"); include("stats.jl"); include("CSetsGPU.jl"); using .CSetsGPU
rng = MersenneTwister(9)
# brute-force number of graph homomorphisms G → H (H symmetric, simple)
function brute(nv, E, ny, adj)
    c = 0
    for code in 0:ny^nv - 1
        h = digits(code; base=ny, pad=nv) .+ 1
        c += all(adj[h[u], h[v]] for (u, v) in E)
    end
    return c
end
nok = 0; N = 0
for trial in 1:120
    n = rand(rng, 3:8); k = rand(rng, 1:3)
    nv, E = partialktree(rng, n, k, rand(rng, 0.6:0.1:1.0); bushy=rand(rng, Bool))
    isempty(E) && continue
    ny = rand(rng, 2:4)
    EY = rand(rng, Bool) ? [(u, v) for u in 1:ny for v in u+1:ny] : randomgraph(rng, ny, 0.6)
    adj = falses(ny, ny); for (u, v) in EY; adj[u, v] = adj[v, u] = true; end
    X = graphinstance(nv, E); Y = symmetric(ny, EY)
    D = decomposition(X, graphsupport(X); alg=MF())
    cnt = CSetsGPU.contract(D, Y; alg=CSetsGPU.counting(Float64), witness=false)
    hb = CSetsGPU.contract(D, Y; alg=CSetsGPU.boolean(), witness=true)
    hc = color(D, Y)
    b = brute(nv, E, ny, adj)
    ok = cnt == b && isnothing(hb) == isnothing(hc) && (isnothing(hb) || ishom(X, Y, hb))
    ok || println("MISMATCH trial $trial count=$cnt brute=$b dense-bool=", isnothing(hb), " cpu=", isnothing(hc))
    global nok += ok; global N += 1
end
println("semiring ok $nok / $N")
