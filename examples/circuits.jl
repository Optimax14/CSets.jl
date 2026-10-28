#
# Circuit equivalence checking as homomorphism search.
#
# A circuit is a C-set with one sort of wires and one sort per gate
# type. Each gate has maps in1, in2, out to its wires. The target Bool
# is the circuit whose wires are 0 and 1 and whose gates are the rows
# of the truth tables, so a homomorphism circuit -> Bool is a
# consistent evaluation of the circuit.
#
# To compare two circuits, build their miter: share the inputs, XOR
# corresponding outputs, OR the results together, and pin that wire to
# 1 with a list (its only allowed image is 1). A homomorphism is then
# an input on which the circuits differ; if there is none, they are
# equivalent.
#
using CSets: FVector, Span, Instance, saturate, color
using CliqueTrees: MF

const WIRE = 1
const AND  = 2
const OR   = 3
const XOR  = 4

#
# The schema: AND, OR, and XOR each have arcs in1, in2, out to Wire.
#
function schema()
    xadj = FVector{Int}([1, 1, 4, 7, 10])
    adjncy = FVector{Int}(fill(WIRE, 9))
    return Span{Int}(4, 4, 9, xadj, adjncy)
end

mutable struct Circuit
    nwire::Int
    kind::Vector{Int}    # sort of each gate
    wires::Vector{Int}   # in1, in2, out of each gate
    pins::Vector{Int}    # wires pinned to 1
end

Circuit() = Circuit(0, Int[], Int[], Int[])

function wire!(c::Circuit)
    c.nwire += 1
    return c.nwire
end

function gate!(c::Circuit, kind::Int, a::Int, b::Int, out::Int=wire!(c))
    push!(c.kind, kind)
    append!(c.wires, (a, b, out))
    return out
end

function pin!(c::Circuit, w::Int)
    push!(c.pins, w)
    return w
end

function instance(c::Circuit)
    ngate = length(c.kind)
    n = c.nwire + ngate
    xadj = Vector{Int}(undef, n + 1)
    adjncy = Int[]
    arc = Int[]
    vtx = Vector{Int}(undef, n)
    xadj[1:c.nwire+1] .= 1
    vtx[1:c.nwire] .= WIRE
    p = 1

    for g in 1:ngate
        k = c.kind[g]
        x = c.nwire + g
        vtx[x] = k

        for i in 1:3
            push!(adjncy, c.wires[p]); push!(arc, 3 * (k - AND) + i)
            p += 1
        end

        xadj[x + 1] = length(adjncy) + 1
    end

    ne = length(adjncy)
    dom = Span{Int}(n, n, ne, FVector{Int}(xadj), FVector{Int}(adjncy))
    return Instance{Int}(dom, schema(), FVector{Int}(vtx), FVector{Int}(arc))
end

#
# The target Bool: wire 1 is false, wire 2 is true, and the gates are the
# rows of the truth tables.
#
function boolean()
    c = Circuit()
    f = wire!(c); t = wire!(c)
    v = (f, t)

    for a in 0:1, b in 0:1
        gate!(c, AND, v[a+1], v[b+1], v[(a & b) + 1])
        gate!(c, OR,  v[a+1], v[b+1], v[(a | b) + 1])
        gate!(c, XOR, v[a+1], v[b+1], v[(a ⊻ b) + 1])
    end

    return c
end

#
# Ripple-carry adders on input wires a and b, least significant bit
# first. Each returns the output wires: n sum bits, then the carry.
#
function adder!(c::Circuit, a, b)
    s = similar(a)
    s[1] = gate!(c, XOR, a[1], b[1])
    carry = gate!(c, AND, a[1], b[1])

    for i in 2:length(a)
        t = gate!(c, XOR, a[i], b[i])
        s[i] = gate!(c, XOR, t, carry)
        carry = gate!(c, OR, gate!(c, AND, a[i], b[i]), gate!(c, AND, t, carry))
    end

    return [s; carry]
end

# the same function, written differently: majority carry, reassociated sum
function adder2!(c::Circuit, a, b)
    s = similar(a)
    s[1] = gate!(c, XOR, b[1], a[1])
    carry = gate!(c, AND, b[1], a[1])

    for i in 2:length(a)
        s[i] = gate!(c, XOR, a[i], gate!(c, XOR, b[i], carry))
        carry = gate!(c, OR, gate!(c, AND, a[i], b[i]), gate!(c, AND, carry, gate!(c, OR, a[i], b[i])))
    end

    return [s; carry]
end

# an adder with a bug: at bit j, the sum uses OR where it should use XOR
function buggy!(c::Circuit, a, b, j)
    s = similar(a)
    s[1] = gate!(c, XOR, a[1], b[1])
    carry = gate!(c, AND, a[1], b[1])

    for i in 2:length(a)
        t = gate!(c, XOR, a[i], b[i])
        s[i] = gate!(c, i == j ? OR : XOR, t, carry)
        carry = gate!(c, OR, gate!(c, AND, a[i], b[i]), gate!(c, AND, t, carry))
    end

    return [s; carry]
end

#
# The miter of two n-bit circuits. Returns the circuit and its input
# and output wires.
#
function miter(n, left!, right!)
    c = Circuit()
    a = [wire!(c) for _ in 1:n]
    b = [wire!(c) for _ in 1:n]
    x = left!(c, a, b)
    y = right!(c, a, b)
    d = gate!(c, XOR, x[1], y[1])

    for i in 2:length(x)
        d = gate!(c, OR, d, gate!(c, XOR, x[i], y[i]))
    end

    pin!(c, d)
    return c, a, b, x, y
end

#
# The support: each wire is supported on itself.
#
function support(X::Instance)
    n = X.dom.nu
    nwire = count(==(WIRE), X.vtx)
    xadj = Vector{Int}(undef, n + 1)

    for x in 1:n + 1
        xadj[x] = min(x, nwire + 1)
    end

    return saturate(X, Span{Int}(n, nwire, nwire, FVector{Int}(xadj), FVector{Int}(collect(1:nwire))))
end

#
# Search for an input on which the two circuits differ. Returns
# nothing if they are equivalent, and otherwise the values of every
# wire of the miter.
#
function counterexample(c::Circuit; kw...)
    X = instance(c)
    #
    # every wire may be 0 or 1, except pinned wires, which must be 1
    #
    L = trues(2, c.nwire)

    for w in c.pins
        L[1, w] = false
    end

    W = [L, nothing, nothing, nothing]
    h = color(W, X, support(X), instance(boolean()); alg=MF(), kw...)
    return isnothing(h) ? nothing : [h[w] - 1 for w in 1:c.nwire]
end

bits(v, w) = sum(v[w[i]] << (i - 1) for i in eachindex(w))
