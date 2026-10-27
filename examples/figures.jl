#
# Figures for the README: a miter of an adder and a buggy adder, with
# each wire drawn by its value in the counterexample (solid = 1,
# dotted = 0), all in the same ink. Red marks only the bug and the outputs that disagree.
#
include("circuits.jl")

const INK  = "#333333"
const GREY = "#9a9a9a"
const RULE = "#c8c8c8"
const RED  = "#d62728"
const SYMBOL = Dict(AND => "∧", OR => "∨", XOR => "⊕")

#
# The miter of adder! and buggy!, recording which gates belong to each
# part. Returns the circuit, inputs, outputs, gate ranges, and the
# index of the bugged gate.
#
function parts(n, j)
    c = Circuit()
    a = [wire!(c) for _ in 1:n]
    b = [wire!(c) for _ in 1:n]
    g0 = length(c.kind)
    x = adder!(c, a, b)
    g1 = length(c.kind)
    y = buggy!(c, a, b, j)
    g2 = length(c.kind)
    d = gate!(c, XOR, x[1], y[1])

    for i in 2:length(x)
        d = gate!(c, OR, d, gate!(c, XOR, x[i], y[i]))
    end

    pin!(c, d)
    bug = g1 + 2 + 5 * (j - 2) + 2
    return c, a, b, x, y, (g0+1:g1, g1+1:g2, g2+1:length(c.kind)), bug
end

function dot(n, j; big=false)
    c, a, b, x, y, (left, right, comp), bug = parts(n, j)
    v = counterexample(c)
    #
    # offsets of each gate's wires, the producer of each wire, and the
    # wires that disagree
    #
    off = Int[]
    p = 1

    for k in c.kind
        push!(off, p)
        p += k == ONE ? 1 : 3
    end

    #
    # each adder gets its own copy of the input labels, so the shared
    # inputs do not cross the picture
    #
    src = Dict{Int, String}()
    inp = Dict{Int, String}()

    for i in 1:n
        inp[a[i]] = "a$i"
        inp[b[i]] = "b$i"
    end

    for (g, k) in enumerate(c.kind)
        k == ONE || (src[c.wires[off[g] + 2]] = "g$g")
    end

    hot = Set(w for i in eachindex(x) for w in (x[i], y[i]) if v[x[i]] != v[y[i]])
    hotgate = Set(g for g in comp if c.kind[g] == XOR && c.wires[off[g]] in hot)

    io = IOBuffer()
    println(io, "digraph miter {")
    println(io, "  graph [rankdir=LR, newrank=true, bgcolor=white, pad=0.15, fontname=\"Helvetica\", fontsize=11, fontcolor=\"$GREY\", nodesep=0.12, ranksep=$(big ? 0.2 : 0.32), outputorder=edgesfirst];")
    println(io, "  node [fontname=\"Helvetica\", fontsize=12, color=\"$INK\", fontcolor=\"$INK\", penwidth=0.9];")
    println(io, "  edge [arrowhead=none, color=\"$INK\", penwidth=$(big ? 0.35 : 1.1)];")

    function gates(io, range, name, label, side, loc="t")
        println(io, "  subgraph cluster_$name {")
        println(io, "    label=\"$label\"; labelloc=$loc; style=\"dashed,rounded\"; color=\"$RULE\"; penwidth=1;")

        if !isempty(side)
            names = String[]

            for i in 1:n, s in ("a", "b")
                label = big ? "" : "$s$(join(Char(0x2080 + d) for d in reverse(digits(i))))"
                shape = big ? "point" : "plaintext"
                push!(names, "$side$s$i")
                println(io, "    $side$s$i [shape=$shape, label=\"$label\", width=0.04, height=0.2, color=\"$INK\"];")
            end

            println(io, "    { rank=same; $(join(names, "; ")); }")
            println(io, "    $(join(names, " -> ")) [style=invis];")
        end

        for g in range
            k = c.kind[g]
            k == ONE && continue
            color = g == bug || g in hotgate ? RED : INK
            width = g == bug || g in hotgate ? 2.0 : 0.9

            if big
                println(io, "    g$g [shape=circle, label=\"\", width=0.06, style=filled, fillcolor=\"$color\", color=\"$color\"];")
            else
                println(io, "    g$g [shape=box, style=rounded, label=\"$(SYMBOL[k])\", width=0.3, height=0.26, margin=0.02, color=\"$color\", fontcolor=\"$color\", penwidth=$width];")
            end
        end

        println(io, "  }")
    end

    gates(io, left, "left", "adder", "L")
    gates(io, right, "right", "buggy adder", "R", "b")
    gates(io, comp, "comp", "differ?", "")

    for (g, k) in enumerate(c.kind)
        ins = k == ONE ? (c.wires[off[g]],) : (c.wires[off[g]], c.wires[off[g] + 1])
        dst = k == ONE ? "one" : "g$g"

        for w in ins
            style = w in hot ? "color=\"$RED\", penwidth=$(big ? 3.0 : 2.0)" :
                    isone(v[w]) ? "" : "style=dotted"
            if haskey(inp, w)
                from = (g in left ? "L" : "R") * inp[w]
            else
                from = src[w]
            end

            println(io, "  $from -> $dst [$style];")
        end
    end

    println(io, "  one [shape=plaintext, label=\"$(big ? "" : "= 1")\", width=0.04];")
    println(io, "}")
    return String(take!(io)), c, a, b, x, y, v
end

function render(path, n, j; big=false, engine=big ? "sfdp" : "dot")
    src, c, a, b, x, y, v = dot(n, j; big)
    write(path * ".dot", src)
    run(`$engine -Tsvg $(path).dot -o $(path).svg`)
    run(`$engine -Tpng -Gdpi=110 $(path).dot -o $(path).png`)
    return c, a, b, x, y, v
end

#
# The big figure: the miter drawn as a strip, one column per bit. The
# adder's gates sit above the middle, the buggy adder's are mirrored
# below, and the comparison runs along the middle.
#
function strip(n, j; dx=5.0, dy=9.0)
    c, a, b, x, y, (left, right, comp), bug = parts(n, j)
    v = counterexample(c)
    off = Int[]
    p = 1

    for k in c.kind
        push!(off, p)
        p += k == ONE ? 1 : 3
    end
    #
    # position of each gate: an adder builds 2 gates for bit 1 and 5 for
    # each later bit; the comparison builds an XOR and an OR per output
    #
    pos = Dict{Int, Tuple{Float64, Float64}}()

    for (range, sign) in ((left, 1.0), (right, -1.0))
        g = first(range)
        pos[g] = (1.0, sign * 2.0); pos[g + 1] = (1.0, sign * 4.0)
        g += 2

        for i in 2:n, r in 1:5
            pos[g] = (i + 0.15 * (r - 3), sign * (1.0 + r))
            g += 1
        end
    end

    g = first(comp)
    pos[g] = (1.0, 0.0)
    g += 1

    for i in 2:n + 1
        pos[g] = (i - 0.3, 0.4); pos[g + 1] = (i, 0.0)
        g += 2
    end

    src = Dict{Int, Int}()

    for (h, k) in enumerate(c.kind)
        k == ONE || (src[c.wires[off[h] + 2]] = h)
    end

    for i in 1:n
        pos[-a[i]] = (i - 0.2, 7.2); pos[-b[i]] = (i + 0.2, 7.2)
    end

    hot = Set(w for i in eachindex(x) for w in (x[i], y[i]) if v[x[i]] != v[y[i]])
    hotgate = Set(h for h in comp if c.kind[h] == XOR && c.wires[off[h]] in hot)
    xy(q) = "\"$(round(dx * q[1], digits=1)),$(round(dy * q[2], digits=1))!\""

    io = IOBuffer()
    println(io, "graph strip {")
    println(io, "  graph [bgcolor=white, splines=false, outputorder=edgesfirst, pad=0.15];")
    println(io, "  node [shape=point, width=0.012, color=\"$INK\", label=\"\"];")
    println(io, "  edge [color=\"$INK\", penwidth=0.18];")

    for (h, k) in enumerate(c.kind)
        k == ONE && continue
        red = h == bug || h in hotgate
        attrs = red ? ", width=0.07, color=\"$RED\"" : ""
        println(io, "  g$h [pos=$(xy(pos[h]))$attrs];")
    end


    for (h, k) in enumerate(c.kind)
        k == ONE && continue

        for w in (c.wires[off[h]], c.wires[off[h] + 1])
            haskey(src, w) || continue
            from = "g$(src[w])"

            style = w in hot ? "color=\"$RED\", penwidth=1.4" :
                    isone(v[w]) ? "" : "style=dotted"
            println(io, "  $from -- g$h [$style];")
        end
    end

    println(io, "}")
    return String(take!(io)), c, a, b, x, y, v
end

function renderstrip(path, n, j)
    src, c, a, b, x, y, v = strip(n, j)
    write(path * ".dot", src)
    run(`neato -n2 -Tsvg $(path).dot -o $(path).svg`)
    run(`neato -n2 -Tpng -Gdpi=150 $(path).dot -o $(path).png`)
    return c, a, b, x, y, v
end
