include("workloads.jl"); include("stats.jl"); include("CSetsGPU.jl"); using .CSetsGPU, CUDA
B = instance(boolean())
c, = miter(30000, adder!, (c, a, b) -> buggy!(c, a, b, 15000)); X = instance(c)
D = decomposition(X, support(X); alg=MF())
CSetsGPU.contract(D, B); CSetsGPU.contract(D, B)
display(CUDA.@profile CSetsGPU.contract(D, B))
