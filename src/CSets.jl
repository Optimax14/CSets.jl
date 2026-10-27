module CSets

using FixedSizeArrays

const FArray{T} = FixedSizeArrayDefault{T}
const FMatrix{T} = FArray{T, 2}
const FVector{T} = FArray{T, 1}
const FScalar{T} = FArray{T, 0}

include("span.jl")
include("schema.jl")
include("instance.jl")

greet() = print("Hello World!")

end # module CSets
