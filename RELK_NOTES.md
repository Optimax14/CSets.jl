# relk: the output-sensitive local solver as a CSets feature

Branch `itay/relk`, from origin/main c9b038b (Richard's eager/lazy rewrite). The kernel was written against 0d2c2a7 (`csets-showcase/relk/RelKernel.jl`, a plug-in for the old `score(f!, ...)`). `contract!`'s signature did not change in c9b038b, so it ports without changes to the search. I based the branch on origin/main rather than `itay/gpu`: `itay/gpu` is still on the pre-lazy API, and a feature branch on it would conflict with upstream's `color.jl`/`eager.jl` rewrite. `itay/gpu` only adds `gpu/`, so either base gives the same `src/` conflicts against `itay/gpu`: none.

## API

```julia
score(R, W, D, Y; kernel = :default)            # unchanged: contract! everywhere
score(R, W, D, Y; kernel = :relk, nthreads = 8)  # relk at every bag (eager search)
score(R, W, D, Y; kernel = :auto, strategy = Eager)  # relk where contract! is estimated to blow up
```

- `kernel ∈ (:default, :relk, :auto)`; anything else is an `ArgumentError`. The default is `:default`, so existing calls are bit-for-bit unchanged.
- `strategy` now defaults to `nothing`, which resolves to:
  - `Eager` for `kernel = :relk`;
  - upstream's `default_strategy(R)` otherwise (`Lazy` for Boolean).
  The lazy search always uses `contract!`. So Boolean CSP/miter calls with `:auto` keep the lazy default, and `:auto` acts inside the eager search.
- `nthreads` is relk's own thread count (it splits the first generator range or a frontier, with a parallel reduction).
- `color` (witnesses) always uses `contract!`, because relk computes scores only.

Code:
- `src/relkernel.jl`: module `CSets.RelKernel`, the showcase kernel plus `ordercost` and `prefer`. I removed a dead duplicate `emit!` (the second definition, the one that ran in the showcase validation, is kept).
- `src/eager.jl`: one branch at the local-problem call.
- `src/color.jl`: keywords, `resolve`, kernel construction in `solve`.

## The `:auto` rule (`RelKernel.prefer`, per bag, at run time)

`contract!` assigns a bag from the end: the separator first (k = m, ..., 1), then the residual. relk plans a whole-bag order with a DP over subsets, using a fanout model `fan(k, S)`. The model estimates the candidates of element k given the assigned set S from four sources: the image of an assigned element (1), preimages (|Y(k)| / |Y(z)|), the pair index, and the child-table ranges. A filter on the default order's cost decides between the two:

1. **Cheap bound, no allocation.** Cost of the default order m..1 counting only arcs: 1 candidate when an assigned element points to k, |Y(k)| / |Y(z)| when k points to an assigned z, |Y(k)| otherwise. Child tables only lower this. If it is ≤ τ = 1e6 visited partial assignments, use `contract!`. Truth tables and K_q targets (sorts of 2–64 elements) stop here.
2. **Model comparison.** Build the bag and compute `cdef = ordercost(m:-1:1)` and `crel = plan(...)`'s DP cost, both under the same fanout model including children. Use relk iff `cdef > τ` and `cdef > ρ · crel`, with ρ = 8.

τ and ρ are `RelKernel.TAU[]` / `RATIO[]`. Bags with more than 62 elements, or with an empty bag, keep `contract!`. The model is not fitted to a machine: it counts visited assignments.

## Validation (`test/relk.jl`; `julia -t N --project=test test/relk.jl`)

This is the showcase's test plus `:auto`. Every score is compared with the default kernel by `==`, with the same type (`Bool` for Boolean); each check also includes the eager Boolean default.

| set | instances | variants per instance | result |
|---|---|---|---|
| small random graph homs, 300 trials, kept when (mY+1)^mX · nY^k ≤ 1e8 | 163 × 4 semirings (Probability, Boolean, Probability with integer weights 0–2, Lawvere with Inf/integer weights on vertices and edges) = 652 | relk serial, relk N threads, auto serial, auto N threads | 652 / 652 |
| large random (nY 50–400, mY 100–2000, k 2–5), 40 trials | 40 × 3 semirings = 120 | same | 120 / 120 (exact `==`; the showcase used `≈` here) |
| `:auto` forced to relk at every bag (τ = ρ = 0) and forced off (τ = ρ = ∞) | 92 × 2 semirings each | same | 184 / 184 each |
| witness through `color`, kernel-name check | | | ok |

The suite passes on the laptop at 4 threads and on HPG (hpg-turin, job 50859190) at 1 and 8 threads: `ALL OK 1140` each.

## Benchmarks (HPG hpg-turin, EPYC 9655P, one core per process; `bench/`)

`bench/bench.jl SUITE CASE KERNEL` runs one case with one kernel. `bench/run_cases.sh` runs each case with `default`, `relk` and `auto` in separate processes and kills each after a timeout. Every kernel uses the eager search, nthreads = 1. Times are warm (min of 2), search only (`score`, including relabeling Y). Decomposition is listed separately.

Jobs (submitted Nov 11): tests 50859190 (done, pass); LSQB SF0.1 50859191, SF1 50859192; hom 50859193–97 (grqc, wikivote, hepth, enron, epinions × path4, cycle3–6, ladder4); miters 50859198 (eq/bug 10k, 30k; Lawvere and Boolean, MMD); CSP/K_q 50859199 (pk3-n500-K4, pk4-n20000-K5, pk5-n20000-K6, pk6-n5000-K7, pk3-n5000-G30, pk4-n1000-G40, grid6x1000-K3, grid5x1000-K4, grid4x20000-K3; Boolean, Probability, Lawvere). Results land in `/blue/fairbanksj/itaykadosh/gpu/csets/res/relk/`.

Results: pending (tables to be filled from those files).

Laptop smoke (REPS=1, so the times include compilation; they show only that every path runs): LSQB SF0.1 q2, the default kernel timed out at 120 s while relk and auto gave the expected 82990. q6 matched for all three kernels. grqc/cycle4 matched for all three, and so did the miter and pk3-n500-K4 Boolean / Probability / Lawvere values.

## GPU feasibility (design note; no prototype)

relk's per-bag work maps onto the existing level-synchronous path (`gpu/CSetsGPU.jl`, `itay/gpu`), with these differences:

- **Work items.** Partial assignments of the first d elements of relk's planned order, instead of separator prefixes. The existing expansion rounds (BFS until 8 items per thread) and the step-budget resplit apply unchanged. The plan is computed on the host per bag; it is target-dependent only through the sort sizes and child column counts, which the host already has after each level.
- **Candidate sources.** All four are CSR arrays already uploaded by `target()`, or cheap to add: the arc image columns (`aimg`), the preimage spans (`pre`), the pair index (one more CSR per arc pair, built on the host on first use, as relk does), and child-table ranges (the existing `narrow!` binary searches). The thread-local DFS is the existing `searchkernel!` with a different element order and source table.
- **Emission.** Unlike `contract!`, relk's rows come out in plan order, not colex order, so the scan-ranked, bit-identical emission does not apply. Rows (separator key, value) go to a staged buffer: the existing two-pass count-then-write.
- **Aggregation, the new part:**
  - (a) Dense histogram for 1–2 separator elements: an atomic ⊕ into |Y(c1)|·|Y(c2)| slots. This is the existing `atomic_or!` / `atomic_add!` / `atomicmin!` set, then a compaction scan that yields colex order directly.
  - (b) Otherwise, pack the key into UInt64 (s · ⌈log2 |Y|⌉ ≤ 64 bits covers LSQB's separators) or two words, then radix-sort (`CUDA.sort!` on keys with a permutation), then a segmented ⊕ reduce-by-key. CUDA.jl has no reduce-by-key, so it is one more kernel, a segmented scan.
- **Estimate:**
  - About 600–900 lines: a new search kernel variant (~400), pair-index upload (~100), dense/sort aggregation (~250).
  - About 1.5–2 weeks including validation against the CPU relk.
- **Expected value:**
  - The LSQB bags that need relk are output-heavy, with a few huge leaf bags (q2/q6/q9 leaves over Message × Person). That is the same heavy-tailed shape where the level-synchronous path got 15–70× over one core on big-Y homs. On the GPU it would be bounded by the radix sort of the emitted rows, about 1–2 G keys/s on the RTX PRO 6000.
  - But CPU relk at 64 threads already answers SF10 in 0.1–5 s per query (SHOWCASE.md), and LSQB loading and indexing dominate end to end.
  - Verdict: feasible, but not cheap, and not the bottleneck for the showcase. I did not prototype it. Worth doing only if the database direction becomes a paper target, starting with (a), which reuses the existing atomics and covers the 1–2-element separators.
