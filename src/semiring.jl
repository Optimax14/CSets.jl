#
# A commutative semiring (S, ⊕, ⊗, 0, 1). Each element of X carries a
# weight for each of its possible images, and a homomorphism h has
# weight
#
#    ⊗ₓ w(x, h(x)),
#
# so that color computes the sum (⊕) of these weights over all
# homomorphisms X -> Y.
#
abstract type AbstractSemiring end

#
# The Boolean semiring ({false, true}, ∨, ∧, false, true): is there a
# homomorphism?
#
struct Boolean <: AbstractSemiring end

#
# The Lawvere semiring ([0, ∞], min, +, ∞, 0): what is the cheapest
# homomorphism?
#
struct Lawvere <: AbstractSemiring end

#
# The probability semiring ([0, ∞), +, ×, 0, 1): what is the total
# weight of all homomorphisms?
#
struct Probability <: AbstractSemiring end

#
# The zero of the semiring: the weight of a forbidden image.
#
function szero(::Boolean, ::Type{T}) where {T}
    return zero(T)
end

function szero(::Lawvere, ::Type{T}) where {T}
    return typemax(T)
end

function szero(::Probability, ::Type{T}) where {T}
    return zero(T)
end

#
# The unit of the semiring: the weight of an unconstrained image.
#
function sone(::Boolean, ::Type{T}) where {T}
    return one(T)
end

function sone(::Lawvere, ::Type{T}) where {T}
    return zero(T)
end

function sone(::Probability, ::Type{T}) where {T}
    return one(T)
end

#
# Addition: combine alternatives.
#
function splus(::Boolean, a, b)
    return a | b
end

function splus(::Lawvere, a, b)
    return min(a, b)
end

function splus(::Probability, a, b)
    return a + b
end

#
# Multiplication: combine independent parts.
#
function sprod(::Boolean, a, b)
    return a & b
end

function sprod(::Lawvere, a, b)
    return a + b
end

function sprod(::Probability, a, b)
    return a * b
end

#
# Whether the unit is the top element, so that 1 ⊕ a = 1 for all a.
# Then a sum stops changing once it reaches 1.
#
function isintegral(::Boolean)
    return true
end

function isintegral(::Lawvere)
    return true
end

function isintegral(::Probability)
    return false
end

#
# Whether addition is selective, so that a ⊕ b is always a or b. Then
# every sum is attained by one of its terms, and a homomorphism of
# that weight can be reconstructed.
#
function isselective(::Boolean)
    return true
end

function isselective(::Lawvere)
    return true
end

function isselective(::Probability)
    return false
end

#
# The default type of weights.
#
function valtype(::Boolean)
    return Bool
end

function valtype(::Lawvere)
    return Float64
end

function valtype(::Probability)
    return Float64
end
