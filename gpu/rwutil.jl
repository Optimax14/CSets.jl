function hweight(R, W, X, Y, h)
    T = CSets.valtype(R, W)
    w = sone(R, T)
    for c in 1:X.cod.nu
        (isnothing(W) || isnothing(W[c])) && continue
        xs = findall(==(c), collect(X.vtx)); ys = findall(==(c), collect(Y.vtx))
        for (j, x) in enumerate(xs)
            w = sprod(R, w, W[c][findfirst(==(h[x]), ys), j])
        end
    end
    return w
end
function randweights(rng, R, Y, nv)
    nyv = count(==(1), collect(Y.vtx))
    if R isa Boolean
        return rand(rng, Bool) ? nothing : Any[rand(rng, nyv, nv) .< 0.85, nothing]
    elseif R isa Lawvere
        return Any[[rand(rng) < 0.1 ? Inf : round(10rand(rng), digits=2) for i in 1:nyv, j in 1:nv], nothing]
    else
        return Any[rand(rng, nyv, nv), nothing]
    end
end
