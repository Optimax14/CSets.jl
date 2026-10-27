include("workloads.jl"); include("stats.jl"); include("CSetsGPU.jl"); using .CSetsGPU
B = instance(boolean())
c, = miter(4, adder!, (c, a, b) -> buggy!(c, a, b, 2)); X = instance(c)
D = decomposition(X, support(X); alg=MF())
ph = phases(D, B)
for (bud, ms) in ((4, 0), (4, 1), (4, 6), (1<<14, 6))
st = Dict{Symbol,Any}(:wanttab => true)
try
CSetsGPU.color(D, B; nitems=1, minwork=0, budget=bud, maxsplit=ms, stats=st)
catch e; println("err ", sprint(showerror, e)[1:min(300,end)]); continue; end
bad = [t for t in eachindex(ph.ncol) if ph.ncol[t] != st[:ncol][t]]
println("bud=$bud maxsplit=$ms bad bags: ", bad[1:min(end, 10)], " cpu ncol ", ph.ncol[bad[1:min(end,5)]], " gpu ", st[:ncol][bad[1:min(end,5)]])
end
