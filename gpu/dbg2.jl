include("workloads.jl"); include("stats.jl"); include("CSetsGPU.jl"); using .CSetsGPU
B = instance(boolean())
for n in (100, 1000, 3000)
c, = miter(n, adder!, (c, a, b) -> buggy!(c, a, b, n ÷ 2)); X = instance(c)
D = decomposition(X, support(X); alg=MF())
t = @elapsed C = CSetsGPU.prepare_contract(D); println("prep ", t); flush(stdout)
st = Dict{Symbol,Any}()
t = @elapsed h = CSetsGPU.contract(C, B; stats=st); println("n=$n contract ", t, " ", st); flush(stdout)
t = @elapsed h = CSetsGPU.contract(C, B; stats=st); println("n=$n contract ", t, " ", st); flush(stdout)
end
