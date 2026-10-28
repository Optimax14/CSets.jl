include("workloads.jl"); include("stats.jl"); include("CSetsGPU.jl"); using .CSetsGPU, CUDA
B = instance(boolean())
c, = miter(10000, adder!, (c, a, b) -> buggy!(c, a, b, 5000)); X = instance(c)
D = decomposition(X, support(X); alg=MF())
C = CSetsGPU.prepare_contract(D)
CSetsGPU.contract(C, B); 
st = Dict{Symbol,Any}(); t = @elapsed CSetsGPU.contract(C, B; stats=st); println("t=$t ", st)
p = CUDA.@profile CSetsGPU.contract(C, B)
show(IOContext(stdout, :limit => true, :displaysize => (60, 140)), p)
