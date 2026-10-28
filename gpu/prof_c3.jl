include("workloads.jl"); include("stats.jl"); include("CSetsGPU.jl"); using .CSetsGPU, CUDA
B = instance(boolean())
c, = miter(10000, adder!, (c, a, b) -> buggy!(c, a, b, 5000)); X = instance(c)
D = decomposition(X, support(X); alg=MF())
C = CSetsGPU.prepare_contract(D)
for i in 1:3
st = Dict{Symbol,Any}(); t = @elapsed CSetsGPU.contract(C, B; stats=st); println("t=$t local=", st[:local], " contract=", st[:contract], " jumptime=", st[:jumptime], " witness=", st[:witness])
end
@time CSetsGPU.contract(C, B)
using Profile
Profile.clear(); @profile for _ in 1:3; CSetsGPU.contract(C, B); end
Profile.print(format=:flat, sortedby=:count, mincount=30, maxdepth=200, noisefloor=0, C=false, combine=true)
