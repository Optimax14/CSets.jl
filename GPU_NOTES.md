# CSets.jl on the GPU: feasibility study and prototype

Branch `itay/gpu`. The CPU package is untouched. The GPU code is under `gpu/`:

| file | what |
|---|---|
| `gpu/CSetsGPU.jl` | (a)+(b) level-synchronous search. `CSetsGPU.color(D, Y)` gives the same answer as `CSets.color`, with bit-identical tables. Also `CSetsGPU.auto` |
| `gpu/contractgpu.jl` | (d) tree contraction on the GPU. `CSetsGPU.contract(D, Y)`; returns `:wide` when a separator domain has more than 64 assignments |
| `gpu/contract.jl` | sequential reference for (d), with round and work counters |
| `gpu/workloads.jl` | generators: miters (from `examples/circuits.jl`), partial k-trees, grids, random targets, `ishom` |
| `gpu/stats.jl` | instrumented copy of `color(D, Y)` with phase timers and tree shape |
| `gpu/test*.jl` | validation against the CPU code |
| `gpu/bench*.jl`, `gpu/hpg_*` | laptop and HPG benchmarks |

Hardware:
- laptop: RTX 5060 Laptop GPU (26 SMs, 8 GB).
- HPG: RTX PRO 6000 Blackwell (GPU job), AMD EPYC 9655P (single-threaded CPU job, the same as the CPU code).

All times are warm, the minimum of 2–3 runs. "Search" means `color(D, Y)` given the decomposition `D`.

## 1. CPU profile (where the time goes)

`decomposition` (CliqueTrees MF ordering plus the CSR work) is separate from the search. The search is the forward pass, and the backward pass is always under 10% of it.

| workload | decomp | forward | backward | tree nodes | depth (levels) | max nodes / level | max bag / sep | table cols max / mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| adder≡adder2 n=10k | 0.24 | 0.35 | – (unsat) | 120k | 12,320 | 18 | 8 / 7 | 24 / 9 |
| adder vs buggy n=10k | 0.25 | 0.23 | 0.03 | 100k | 19,934 | 15 | 10 / 7 | 56 / 9 |
| partial 4-tree n=20k → K5 (bushy) | 0.08 | 0.92 | 0.01 | 18.5k | 26 | 2,513 | 21 / 15 | 2,880 / 113 |
| partial 6-tree n=2k → K7 | 0.007 | 5.8 | 0.004 | 1.9k | 24 | 220 | 29 / 22 | 88k / 5.4k |
| partial 4-tree n=1k → G(40, .3) | 0.001 | 24 | 0.002 | 925 | 17 | 196 | 19 / 13 | 155k / 3.8k |
| grid 4×2000 → K3 | 0.004 | 0.045 | 0.003 | 8k | 7,977 | 2 | 9 / 7 | 54 / 33 |
| grid 8×500 → K3 | 0.003 | 0.82 | 0.003 | 3.5k | 939 | 14 | 13 / 10 | 10k / 1.4k |

(laptop, `gpu/profile.out`)

- **Circuits:** the work is tiny (tables of at most 64 columns, separators of at most 7) but the tree is a few long chains: depth ≈ 1.2–2× the number of bits, with about 10 nodes per level. The CPU spends about 2 µs per bag. Decomposition costs as much as the search.
- **Bushy trees with big targets:** the depth is about 25, there are thousands of bags per level, and the table sizes and per-bag search costs are heavy-tailed. A single bag can hold 155k columns.
- **Grids:** these are path-like, and the work per level grows like 3^width.

In `decomposition`, the MF ordering is 60–80% of the time. Measured on HPG at n = 10k / 30k / 100k bits:

| part | 10k | 30k | 100k |
|---|---:|---:|---:|
| `decomposition` total | 0.17 | 0.77 | 2.85 |
| ordering only | 0.07 | 0.37 | 1.70 |
| `decomposition` with the ordering given | 0.07 | 0.40 | 1.08 |

`sccs`, `gaifman` and `permute` are each only a few ms. The remainder is clique-tree construction plus `separators`.

**Cheap win for Richard:** `alg=MMD()` gives a decomposition 3–5× faster than `MF()` on the miters, with the same search time (laptop, 10k: 0.039 s vs 0.20 s, max bag 9 vs 10). `RCMMD` / `AMF` are also cheaper.

## 2. Designs

**(a) Level-synchronous (one thread per work item, all bags of a level in parallel).**
- Fine for bushy trees.
- Hopeless for the miters: 12k–48k levels. Each level costs one kernel launch, one small device-to-host copy, a scan and a gather, about 85–100 µs per level even with no host overhead.
- The real killer is that one GPU thread running one bag's backtracking search is about 30–50× slower than a CPU core: it is latency-bound, with scratch space in global memory. So even a persistent kernel with no launch overhead would lose to the CPU on deep trees.

**(b) Intra-bag splitting (cuTS / EGSM-style frontier).**
- A bag is split by fixing a prefix of its separator: the work items of a bag are its partial assignments of positions 1..d, in the order the CPU search visits them.
- Item i's columns get rank `scan(count)[i-1] + seq`, so the table is written in the CPU's order with no sort and no dedup. Tables come out bit-identical to the CPU's.
- Expansion runs BFS rounds until the level has 8 items per GPU thread.
- This is what makes (a) work on heavy-tailed bags. It was implemented together with (a), and the 8-items-per-thread default alone was worth 2.5×.

**(c) Relational (Yannakakis joins).**
- Not prototyped.
- The table of a bag is the projection of a join of edge relations, with semijoins against the child tables. The existing search is already a worst-case-optimal-style join, interleaving the variables of the bag and probing child tables by binary search.
- A GPU hash or sort-merge join would materialize intermediate relations that the DFS never builds; for big Y they would be blow-ups of |Y|^k. I don't think it beats (b), which is the same computation without intermediates.

**(d) Tree contraction (rake + pointer jumping on Boolean matrices).**
- `Full_t` is the local table of bag t: the assignments consistent inside the bag, ignoring the children. `Loc_t` is its projection onto the separator.
- The message of a bag is a bit vector over `Loc_t`. With all children but one (c) known, the map from v_c to v_t is a Boolean matrix `M_t : Loc_c × Loc_t`, so chains compose by Boolean matrix products. This is the relational composition of spans.
- Each phase:
  1. Build M_t for chain bags (exactly one undone child).
  2. Pointer-jump: Wyllie, O(log length) rounds.
  3. Rake bags whose children are all done.
  4. Finish chains whose bottom is done: v_t = v_bottom · M_t.
- The witness: x_t(b) for every b, f_t : Loc_parent → Loc_t, pointer jumping on the parent pointers, then all bags fill in h in parallel.
- Everything is O(log n) dependent rounds: 26–61 rounds instead of 12k–231k levels.
- The cost is work: composition is L × L × L instead of the sequential vector × matrix L × L. It wins only when L is small:

| workload | max L | contraction work (bit-word ops) |
|---|---:|---:|
| circuits | 64 | 0.05–0.09 G for 10k bits |
| grid 4×N → K3 | 54 | small |
| grid 6×1000 → K3 | 1,458 | 48 G (the sequential Julia reference took 31 s; **the CPU code takes only 0.15 s**) |
| grid 8×500 → K3 | 19,683 | 5.7 T |
| deep partial 4-tree → K5 | 6,400 | 30 G |

  So (d) is for small separator domains, which are exactly the circuits. The GPU version uses one UInt64 per matrix row (L ≤ 64); `:wide` otherwise.

## 3. Prototype results (HPG: RTX PRO 6000 vs one EPYC 9655P core)

All answers agree with the CPU code: same satisfiable/unsatisfiable verdict, and every returned map is verified to be a homomorphism.

Validation, on random partial k-trees with K_q or random targets plus miters of 4–100 bits:
- `test.jl`: 174/174, with tables bit-identical per bag.
- `test_cgpu.jl`: 310/310.
- `test_contract.jl` (reference): 210/210.

### (a)+(b) level-synchronous: big tables, shallow trees

| workload | CPU search (s) | GPU search (s) | speedup |
|---|---:|---:|---:|
| partial 4-tree n=3000 → G(40, 0.3) | 120.5 | 1.70 | **71×** |
| partial 3-tree n=5000 → G(30, 0.3) | 13.2 | 0.21 | **63×** |
| partial 4-tree n=1000 → G(40, 0.3) | 47.2 | 1.18 | **40×** |
| partial 5-tree n=20000 → K6 | 7.77 | 0.21 | **37×** |
| partial 6-tree n=5000 → K7 | 22.1 | 0.68 | **33×** |
| partial 4-tree n=20000 → K5 | 1.21 | 0.080 | 15× |
| grid 10×200 → K3 (113 levels) | 3.48 | 0.23 | 15× |
| grid 7×500 → K4 (596 levels) | 3.18 | 0.57 | 5.6× |
| grid 8×500 → K3 (939 levels) | 1.20 | 0.77 | 1.6× |
| grid 6×1000 → K4 (2003 levels) | 1.55 | 1.27 | 1.2× |
| partial 5-tree n=1000 → G(60, .25), unsat | 11.7 | 7.39 | 1.6× (one huge leaf-level bag: tail-bound) |

On the laptop (RTX 5060) the same kind of workloads give 10–19×.

On deep trees the level-synchronous code loses: miters 0.05–0.1×, grid 4×2000 0.02×.

### (d) contraction: the circuits and other path-like trees with small domains

| workload | levels | CPU search | GPU contract | search speedup | rounds |
|---|---:|---:|---:|---:|---:|
| adder vs buggy, 10k bits | 19,934 | 0.335 | 0.031 | 11× | 4 phases / 46 jumps |
| adder vs buggy, 30k bits | 48,490 | 1.17 | 0.102 | 11× | 4 / 49 |
| adder vs buggy, 100k bits | 231,019 | 3.57 | 0.331 | 11× | 4 / 58 |
| adder≡adder2, 10k (unsat) | 12,320 | 0.329 | 0.042 | 8× | 5 / 29 |
| adder≡adder2, 30k | 45,546 | 1.05 | 0.099 | 11× | 5 / 32 |
| adder≡adder2, 100k | 144,007 | 3.22 | 0.494 | 6.5× | 5 / 35 |
| grid 4×20000 → K3 | 79,975 | 0.64 | 0.032 | 20× | 2 / 52 |

Note: grid 4×100000 → K3 gives `:wide` (one MF bag with L > 64).

**End to end with MF on the CPU** (n = 10k miter): 0.22 + 0.335 = 0.56 s on the CPU vs 0.22 + 0.031 = 0.25 s with the GPU, about 2.2×. Decomposition now dominates (Amdahl).

#### Where the contraction time goes

Laptop, 30k-bit buggy miter, 300k bags, device total about 150 ms:

| step | time | note |
|---|---:|---|
| local tables `Full_t` (one level-synchronous kernel over all bags) | 56 ms | |
| `cproj` binary searches | 27 ms | since parallelized per column |
| jump kernel | 21 ms | 32 rounds × 0.66 ms |
| witness pointer jumping | 12.5 ms | 17 rounds × 0.74 ms |
| host-to-device plan upload | 10 ms | |
| everything else | < 5 ms | |

- Host round-trips are about 10–30 µs per round; the round loop is **not** launch-bound. The cost is Wyllie's O(n log n) work: every chain bag does 64 rows of work each round.
- Host `plan` is 10–25 ms; it is per decomposition, so it can be cached when the same X is used with many Y.

## 4. Where the GPU can and cannot win

**Wins:**
- Wide bags, large Y, big tables, shallow or bushy trees: 15–71× on search with design (a)+(b). This is where the work is.
- Deep path-like trees with small separator domains (circuits, narrow grids): 6–20× on search with design (d).

**Loses or ties:**
- Deep trees with big domains (grid 6×1000 → K4, deep partial k-trees → K5): (a) is depth-bound and (d) is work-bound (cubic in L). The CPU or a hybrid is better.
- Single giant bags: one item's DFS tail dominates. This needs finer splitting inside the residual, or warp-cooperative candidate checks.

**About "100×":**
- On search, ~70× is measured on the largest big-Y instance, and the trend grows with instance size: 40× at n=1k, 71× at n=3k for G(40). 100× on search is plausible for bigger big-Y instances or a B200.
- On the circuits, end to end is capped by the decomposition: about 2× today.
  - A GPU ordering (MF is 60–80% of decomposition) and MMD instead of MF lift the cap to roughly 4–5×.
  - Beyond that, clique-tree construction and `separators` (≈ 0.4× of the CPU search time) also have to move off the CPU, or be amortized over many targets for the same X.
  - 100× end to end on the miters is not realistic. 10–20× might be, with all of decomposition on the GPU.

## 5. Recommended design

1. `auto`: run contraction (d) when every separator domain has at most 64 assignments (multi-word rows would extend this to about 256). Otherwise run level-synchronous (a)+(b) when the tree has fewer than ~256 levels, else the CPU.
2. Next steps, by expected value:
   - **(d)** Replace Wyllie with random-mate compress, O(n) work instead of O(n log n): about 2–3× on the jump rounds.
   - **(d)** Multi-word bitsets for L up to 256–1024.
   - **(d)** Cache `plan`, `Full_t` and `cproj` per decomposition.
   - **(a)+(b)** Split heavy items inside the residual (existence search: first thread to find one wins), and put scratch in shared memory or registers for small bags.
   - **Decomposition:** a GPU ordering (our GND/MF work) plus a GPU clique-tree/`separators` pass. Until then, use `MMD()` for circuits.
3. The heterogeneous split: symbolic work (decomposition, plan) stays on the CPU and is cached per X; Y-dependent numeric work (tables, contraction) runs on the GPU.

## What to tell Richard

- The search ports cleanly. The GPU kernel is `bagsearch!` almost line for line, one thread per (bag, separator prefix). The tables are bit-identical because items are ordered and ranked by a scan, not sorted.
- Big targets and wide bags are where the GPU shines: 15–71× on the search.
- Deep decompositions (the miters) need the categorical trick: separator tables are relations, chains of bags compose as Boolean matrices (composition of spans), and pointer jumping turns 20k–230k sequential levels into about 50 rounds. That gives 6–20× on the miter search. It is only cheap when the separator domains are small (≤ 64 here), because composition is cubic.
- On the miters the bottleneck is now `decomposition`, mostly the MF ordering. `alg=MMD()` is 3–5× faster with the same search time.

## Round 2 (multi-word contraction, semirings, random mate, budgets, caching)

HPG was not reachable for this round (the SSH control master died: run `! ssh -fN hpg`). The numbers below are from the laptop RTX 5060 while other jobs were running on it, so they are noisy. The HPG suite is ready but has not been run: `gpu/hpg_gpu2.sbatch` (suites `big wide e2e multi`, script `gpu/hpg_bench2.jl`) and `gpu/hpg_decomp.sbatch` (`prof_nd.jl`, `prof_decomp.jl`). Raw laptop TSVs are in `gpu/results/laptop_round2/`.

### 1. Separator domains of any size, and an operator-parameterized algebra

`contractgpu.jl` is rewritten around an algebra parameter:

- **`BoolPacked()`** (the default): Boolean messages and matrices, 64 entries per `UInt64` word. Bag t has `cld(L_t, 64)` words per row.
  - The product M_u ⋅ M_t uses one thread per output word: it ORs the rows of M_t selected by the set bits of row a of M_u, walking them with `trailing_zeros`.
  - The vector–matrix product works the same way.
- **`Semiring(zero, one, add, mul, atomic)`**: dense entries for any commutative semiring.
  - Column values are ⊗ over the done children's messages, and the projections are ⊕-accumulated with `atomic`: `CUDA.atomic_or!` for Booleans, `atomic_add!` for counting, `atomic_min!` for min-plus.
  - `contract(C, Y; alg=CSetsGPU.counting(Float64), witness=false)` returns the number of homomorphisms. It matches brute force on 120/120 random instances (`gpu/test_semiring.jl`).
  - `CSetsGPU.boolean()` is the dense Boolean semiring with a witness. It agrees with the CPU code.
  - The semiring branch (`itay/semiring`) can use the same rake / chain / jump / fin / witness driver and swap in its own `Semiring`.
- Matrices use CSR-like offsets: bag t stores `maxL` rows of width `width(alg, L_t)`. There is no cap at 64 any more; the only limit is a memory check, `maxbytes` (default: a quarter of the device memory), and `:wide` now means "does not fit".

Validation: `test_cgpu.jl` passes 310/310 with 0 `:wide` cases (11 were `:wide` before).

### 3. Linear-work contraction (random mate) and caching

- **Random mate (`jump=:mate`, the default):** a chain bag with heads absorbs its chain child with tails (M_t = M_u ⋅ M_t). The absorbed bag is frozen with its matrix and pointer.
  - After the chain's bottom is done, frozen bags are resolved in reverse round order, as v_u = v_rptr ⋅ M_u.
  - Total work is O(n) matrix products instead of Wyllie's O(n log n), at the cost of about 2× more rounds. Each round is one small kernel, one `findall` over the absorbers, and three kernels over the absorbers only.
  - `jump=:wyllie` keeps the old scheme.
- **Caching:** `CSetsGPU.prepare(D)` (level-synchronous) and `CSetsGPU.prepare_contract(D)` (contraction) hold everything that depends only on the decomposition: the plan of every bag, the levels, the tree arrays and the local-table plan of D'. The target-dependent work runs from them for each Y.

Contraction, laptop, CPU time vs GPU time (s):

| workload | CPU | GPU, mate | GPU, Wyllie | jump time: mate / Wyllie | max L |
|---|---:|---:|---:|---:|---:|
| grid 4×100000 → K3 | 3.07 | **0.227 (13.5×)** | 1.00 | 0.030 / 0.59 | 81 |
| grid 5×1000 → K4 | 0.45 | **0.11 (4.1×)** | 1.36 | 0.06 / 1.31 | 1024 |
| grid 6×1000 → K3 | 0.18 | 0.54 (0.33×) | 0.70 | 0.34 / 0.64 | 1458 |
| grid 5×2000 → K3 | 0.18 | 0.36 | 0.14 (1.3×) | 0.28 / 0.066 | 243 |
| grid 6×4000 → K3, grid 7×1000 → K3 | 0.77, 0.54 | – | – | – | `:wide` (> 25% of 8 GB) |

- Random mate fixes the deep, small-L cases: the jump rounds are 20× cheaper on grid 4×100000.
- For L in the thousands, the cubic composition cost dominates and the CPU wins, as predicted in section 2.
- The grid 5×2000 regression comes from per-round overhead: 75 rounds, each with a `findall` sync. A fixed number of mate rounds without a sync would fix it.

### 2. Decomposition

Breakdown of `decomposition` on the laptop, min of 5 runs, adder vs buggy miter (s):

| part | 10k, MF | 10k, MMD | 30k, MF | 30k, MMD |
|---|---:|---:|---:|---:|
| total | 0.228 | 0.055 | 0.441 | 0.259 |
| ordering only | 0.084 | 0.010 | 0.298 | 0.055 |
| `cliquetree` with the ordering given | 0.017 | 0.011 | 0.057 | 0.041 |
| CSets plumbing | | ≈ 0.025 | | ≈ 0.07 |

The CSets plumbing is sccs, permutes, gaifman, invspan, invmap and separators. Each of these is a few ms (`prof_rest.jl`).

- **MMD is the big lever:** the ordering is 8× cheaper than MF and the search time is the same.
- With MMD, what remains is mostly CliqueTrees' clique-tree construction plus about 0.07 s (at 30k) of O(n) CSR passes. A GPU ordering (GND) would save at most the 0.01–0.055 s that MMD's ordering costs.
- **ND quality on miters (CPU `ND(MMD(), METISND())`, 3k bits):** max bag 10 (same as MF/AMD) and depth 3,390 instead of 12k, and the CPU search time drops 6×. So the GPU ND orderings (tn-work GND) would give bags just as good. They matter more for level-synchronous search than for contraction, which does not care about depth.
- I did not port the plumbing (gaifman / separators) to the GPU: it is ≤ 0.07 s at 30k, smaller than the remaining CliqueTrees symbolic work.

End to end (decomposition + search), CPU vs GPU contraction with the same ordering on both, laptop:

| miter | ordering | decomp | CPU search | GPU contract | search speedup | end to end, prep uncached | end to end, prep cached |
|---|---|---:|---:|---:|---:|---:|---:|
| bug 10k | MF | 0.19 | 0.31 | 0.041 | 7.5× | 1.6× | 2.2× |
| bug 10k | MMD | 0.054 | 0.34 | 0.036 | 9.3× | 3.5× | **4.3×** |
| eq 10k | MMD | 0.066 | 0.31 | 0.024 | 12.8× | 3.3× | 4.2× |
| bug 30k | MMD | 0.22 | 0.78 | 0.111 | 7.0× | 2.3× | 3.0× |
| eq 30k | MMD | 0.25 | 0.77 | 0.069 | 11.2× | 2.5× | 3.2× |

### Same X, many targets

| X | targets | CPU | level-sync, cached prep | level-sync, uncached | speedup |
|---|---|---:|---:|---:|---:|
| partial 3-tree, n=20000 | 24 random targets on 4–9 vertices (11 satisfiable) | 6.84 s | **0.79 s** | 1.09 s | 8.6× |

- Contraction on the same 24 targets took 1.1–1.5 s; this run was noisy.
- True one-launch batching (a target index on every work item and table) is not implemented: it would need a target dimension in `Plan`/`Target` and in the tables. Caching already removes the per-decomposition setup.

### 4. Splitting giant nodes (step budget)

- In a final round (mode 0), `color(...; budget=1<<14, maxsplit=6)` stops any item whose DFS takes more than `budget` steps while its separator is not yet fully fixed, and sets `abort[i]`.
- The host then splits only the aborted items, by one more separator position, and reruns the round. Completed items pass through unchanged, and the budget is multiplied by 4 on each resplit.
- Order is preserved because aborted items are replaced in place by their children, so the tables are still bit-identical. `test.jl` passes 174/174 with budgets of 4, 64 and 16k.
- Not split yet: items whose whole separator is fixed but whose residual existence search is huge. That needs first-to-succeed sub-items with a per-parent flag.
- The tail-bound unsatisfiable case (pk5-n1000-G60, 1.6× on HPG) is in the HPG `big` suite with `budget=0` and `budget=16k`, still to be run.

## Round 3: Richard's (R, W) API, batching, HPG results

### Upstream API (origin/main, through 0d2c2a7)

The branch is rebased on origin/main. `src/` is untouched; the GPU code is in `gpu/`.

| GPU call | what it does |
|---|---|
| `CSetsGPU.score(R, W, D, Y)` | the score, for `R ∈ {Boolean(), Lawvere(), Probability()}` and W in upstream's per-sort weight-matrix convention (`nothing`, or `W[c]` of size \|Y(c)\| × \|X(c)\|, caller numbering) |
| `CSetsGPU.color(R, W, D, Y)` | the score and a witness, for selective R |
| `CSetsGPU.color(W, D, Y)` | a Boolean witness |

These run on the contraction (`contractgpu.jl`):

- The semiring maps to an algebra:
  - Boolean: bit-packed.
  - Lawvere: `Semiring(Inf, 0, min, +, atomicmin!)`, with min done by compare-and-swap.
  - Probability: `Semiring(0, 1, +, *, atomic_add!)`.
- Weights become one column weight per entry of a bag's local table: the product of the weights of the bag's residual elements. Each element is residual in exactly one bag, so every weight is counted once.
- The witness for a selective semiring picks, in each bag, the first column whose value equals the best value over its separator assignment.
- `:wide` (out of memory) falls back to the CPU `CSets.score` / `CSets.color`.

Validation against upstream's CPU `score`/`color`:

| test | what it checks | result |
|---|---|---|
| `test_rw.jl` / `test_rw_rep.jl` | Boolean / Lawvere / Probability with random weights: score, witness value, witness is a homomorphism, witness weight equals the CPU value | 150/150 in each of 15 runs (12 seeds) |
| `test.jl` | GPU tables vs the d2927d3 search, kept as `gpu/refsearch.jl` (upstream's new decompositions reorder elements; the tables still match bit for bit) | 174/174 |
| `test_cgpu.jl` | contraction answers vs CPU | 310/310 |
| `test_many.jl` | batched answers vs CPU, per target | 434/434 |
| `test_semiring.jl` | counting vs brute force | 120/120 |
| `test_contract.jl` | contraction reference vs CPU | 210/210 |

One `test_rw` run before the rebase reported 149/150 and could not be reproduced in 14 later runs. GPU float atomics can reorder sums, so a near-tie is the likely cause.

### Batching many targets in one pass (level-synchronous)

`CSetsGPU.color(prep, Ys::Vector)` carries a target index on every work item. Bag t gets a table per target (index `(q-1)·nt + t`), and the targets are concatenated into one instance with per-target blocks of `typx`/`base`. A target whose table becomes empty is marked dead; the other targets continue, and the backward pass runs only for the live ones.

HPG (RTX PRO 6000 vs one EPYC 9655P core), partial 3-tree n=20000, 24 random targets (11 satisfiable):

| method | time (s) | vs CPU |
|---|---:|---:|
| CPU | 7.72 | 1× |
| one target at a time, cached prep | 0.67 | 11.5× |
| **one batched pass** | **0.142** | **54×** |

### Random-mate rounds without a host sync per round

The rounds now run in batches of 8, with one host check per batch, and no `findall`. On the laptop this fixed grid 5×2000 (0.36 s → 0.022 s, 8× over the CPU).

### HPG v3 results (after the batched-round and type-stability fixes; job 48021453)

RTX PRO 6000 vs one core of the GPU node's CPU, upstream CPU code (0d2c2a7). Raw: `gpu/results/hpg/v2_wide_48021453.tsv`, `v2_e2e_48021453.tsv`. Job 48020048 ran the same suites before the type-stability fix (`gpu/results/hpg/v3b/`); its contraction times are 5–10× slower and are superseded, except for `multi`.

Contraction (search only, s):

| workload | max L | CPU | GPU, mate | GPU, Wyllie | best speedup |
|---|---:|---:|---:|---:|---:|
| grid 4×100000 → K3 | 81 | 3.81 | 0.166 | 0.128 | **30×** (mate 23×) |
| grid 5×2000 → K3 | 243 | 0.223 | 0.0107 | 0.0118 | **21×** |
| grid 5×1000 → K4 | 1024 | 0.503 | 0.069 | 0.206 | 7.3× |
| grid 6×4000 → K3 (2.7 GB) | 1458 | 0.782 | 0.151 | 0.532 | 5.2× |
| grid 6×1000 → K3 | 1458 | 0.188 | 0.075 | 0.117 | 2.5× |
| grid 7×1000 → K3 (5.3 GB) | 4374 | 0.793 | 0.573 | 1.36 | 1.4× |

- Random mate is the right default: it wins everywhere except grid 4×100000, where Wyllie's fewer rounds (59 vs 140) win by 1.3×.
- The batched rounds fixed grid 5×2000 on HPG too: 14× → 21×.
- Above L ≈ 1000 the cubic composition dominates (`contract` is 90–97% of the GPU time); at L = 4374 the GPU barely wins.

Miters, end to end (decomposition on the CPU + search), same ordering on both sides:

| miter | MF: search | MF: end to end | MMD: search | MMD: end to end | MMD: end to end, prep cached |
|---|---:|---:|---:|---:|---:|
| eq 10k | 24× | 2.3× | 26× | 4.8× | 5.5× |
| bug 10k | 19× | 3.9× | 19× | 6.1× | 7.5× |
| eq 30k | 21× | 2.9× | 38× | 6.5× | 7.7× |
| bug 30k | 23× | 3.8× | 27× | 6.1× | 7.3× |
| eq 100k | 22× | 2.9× | 37× | 5.5× | 6.6× |
| bug 100k | 24× | 3.1× | 26× | 5.4× | 6.5× |

- Search is 19–38× now (was 11–27×). Caveat (Round 4): the CPU search times in this job were 1.7–1.9× slower than in later runs and in the separate CPU job, so these miter speedups are overstated by about that factor. End to end with MMD is 4.8–6.5×, or 5.5–7.7× with the per-decomposition prep cached. With MF it is 2.3–3.9×: MF's ordering is the Amdahl term (0.3–1.9 s vs 0.03–0.68 s for MMD).
- At 100k the GPU search is 0.13–0.22 s against 0.6–1.9 s of decomposition, so the decomposition is now 75–90% of the GPU end-to-end time.

Many targets (job 48020048, before the type-stability fix; the contraction column there is the slow version):

| X | targets | CPU | level-sync, one at a time | batched one pass | contraction, one at a time | best |
|---|---|---:|---:|---:|---:|---:|
| partial 3-tree n=20000 | 24 (11 sat) | 7.72 | 0.672 | **0.142** | 1.98 | **54×** |
| grid 4×2000 | 24 (24 sat) | 35.4 | 170 | **5.97** | 45.9 (1 `:wide`) | 5.9× |

grid 4×2000 is the bad case for the level-synchronous path: 8k levels, so one target at a time is 4.8× slower than the CPU, and only batching (24 targets per level) recovers 5.9×. Per target, `:auto` would pick the contraction here (height ≥ 256).

### HPG v2 results (before the batched-round and type-stability fixes)

Contraction:

| workload | CPU (s) | GPU (s) | speedup |
|---|---:|---:|---:|
| grid 4×100000 → K3 | 3.29 | 0.093 | **35.5×** |
| grid 5×2000 → K3 | 0.18 | 0.013 | 14× |
| grid 5×1000 → K4 | 0.45 | 0.070 | 6.5× |
| grid 6×4000 → K3 (2.7 GB of matrices; L = 1458) | 0.96 | 0.139 | 6.9× |
| grid 6×1000 → K3 | 0.18 | 0.076 | 2.3× |
| grid 7×1000 → K3 (L = 4374) | 0.77 | 0.59 | 1.3× |

Level-synchronous search, best of `budget=0` / `budget=16k`:

| workload | speedup |
|---|---:|
| pk4-n3000-G40 | 71× |
| pk4-n1000-G40 | 69× (the budget split helps: 43× without it) |
| pk3-n5000-G30 | 67× |
| pk5-n20000-K6 | 40× |
| pk6-n5000-K7 | 36× |
| pk4-n20000-K5 | 17× |
| pk5-n1000-G60 (unsat; tail-bound in the residual search) | 1.7–2× |

Miters, end to end (decomposition + search), CPU vs GPU contraction with the same ordering on both:

| miter | ordering | search speedup | end-to-end speedup |
|---|---|---:|---:|
| 10k | MMD | 18–23× | 5.1–6.4× |
| 10k | MF | 11.5× | 1.7–3.3× |
| 100k | MMD | 20–27× | 3.4–4.0× |

- **Decomposition (HPG CPU):** MMD and AMD build the decomposition 2.5–4× faster than MF with the same bags. ND(METIS) gives the same width but is 2–5× slower than MF.
- **Upstream CPU speed-ups:** the new upstream CPU search is 1.2–2.9× faster than d2927d3 on the laptop (miter 10k: 0.27 vs 0.78 s). That lowers the miter speedups proportionally; the large-target speedups barely change.

Raw TSVs are copied to `gpu/results/hpg/` (`v2/`, `v3a/`, `v3b/`, plus the top-level `v2_*_480*.tsv`).

### Semiring values and weights in the level-synchronous search

The level-synchronous search (`CSetsGPU.color(prep, Ys; S, Ws, witness)`) now carries semiring values, so Richard's API runs on the GPU for large targets and shallow trees as well as for deep trees.

- **Tables:** every column (separator assignment) has a value: the ⊕ over its residual extensions of the ⊗ of the weights of the bag's residual elements and of the values of the matching child columns.
  - When the separator is complete, the narrowed child range is a single column.
  - A work item emits the separator assignment, with its sum, when its search backtracks out of the residual. Items split only on separator prefixes, so no separator assignment is shared by two items.
- **Weights:** upstream's per-sort matrices, one block per target. A weight equal to the semiring's zero prunes the candidate immediately.
- **Witness (selective semirings):** in the backward pass, each bag keeps the best residual extension given its parent's choice.
- **Dispatch:** `CSetsGPU.score(R, W, D, Y; method=:auto | :levels | :contract)` and `CSetsGPU.color(R, W, D, Y; ...)`. `:auto` uses the level-synchronous search below 256 levels and the contraction above (falling back to the level-synchronous search if the contraction does not fit).
- **Batched forms:** `CSetsGPU.score(R, Ws, D, Ys)` and `CSetsGPU.color(R, Ws, D, Ys)` run many targets with their weights in one pass.

Validation against upstream's CPU `score`/`color`:

| test | what it checks | result |
|---|---|---|
| `test_lsr.jl` | level-synchronous search; Boolean / Lawvere / Probability; 1–5 targets per pass with random weights; random split and budget settings | 231/231 and 233/233 (two seeds) |
| `test_rw.jl` | random `method` | 150/150 × 4 |

The whole suite also passes with `--check-bounds=yes`. That run caught a real bug: expansion rounds were computing values of partial prefixes. It is fixed.

Laptop RTX 5060 vs the new upstream CPU (which is up to 3× faster than d2927d3):

| instance | semiring | speedup |
|---|---|---:|
| pk3-n2000 → G(30) | Boolean | 4.1× |
| pk3-n2000 → G(30) | Lawvere | 3.4× |
| pk3-n2000 → G(30) | Probability | 5.8× |
| small instances (CPU < 0.1 s) | all three | 0.4–0.6× |

On the small instances the GPU loses: the GPU time is all kernel time, spent in one thread's search of a large leaf bag.

HPG numbers for this version: see Round 4 below.

## Round 4: HPG on the level-search semiring version (Nov 10–11; jobs 50848717 RTX, 50848718 B200, 50755228 CPU)

Code: `itay/gpu` at c753ee7 plus kernel stats only (3df63c6). Script `gpu/hpg_cmp.jl` (the `threads/hpgcmp.jl` workloads, same names and seeds, plus a size sweep); jobs `gpu/hpg_gpu3.sbatch` (one GPU per job), `gpu/hpg_cpu3.sbatch` (CPU baselines in their own jobs). All jobs source `env.sh` (`ulimit -c 0`). Raw: `gpu/results/hpg/v4/`. Correctness suites on the RTX PRO 6000 (job 50753980): test_rw 150/150, test_lsr 231/231, test 174/174, test_cgpu 310/310, test_many 434/434, test_semiring 120/120, test_contract 210/210.

- serial = upstream CPU code, one EPYC 9655P core (new job 50755228).
- best threads = min over 1–84 threads and the four itay/threads runs (`threads/results/cmp_threads_*.tsv`). The new 84-thread run of the sweep (job 50755229) is still pending, so sweep rows have serial only; their "best CPU" is an upper bound on the CPU time.
- GPU auto = `CSetsGPU.score(R, W, D, Y)` with the default `method=:auto` (levels below 256 levels, else contraction); best = min over `:auto`, `:levels`, `:contract`. Search only, decomposition excluded; min of 2–3 warm runs.

| workload | serial | best threads (n) | best CPU | RTX auto | RTX best (method) | B200 auto | B200 best | best CPU / RTX auto | best CPU / B200 auto |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pk4-n20000-K5 | 0.8165 | 0.1468 (32) | 0.1468 | 0.0763 | 0.0758 (levels) | 0.1003 | 0.1003 (auto) | 1.9× | 1.5× |
| pk5-n20000-K6 | 5.776 | 0.4986 (84) | 0.4986 | 0.2026 | 0.2026 (auto) | 0.239 | 0.2308 (levels) | 2.5× | 2.1× |
| pk6-n5000-K7 | 17.12 | 1.171 (84) | 1.171 | 0.8084 | 0.8084 (auto) | 0.8984 | 0.8984 (auto) | 1.4× | 1.3× |
| pk3-n5000-G30 | 5.183 | 0.1874 (64) | 0.1874 | 0.1428 | 0.1422 (levels) | 0.1523 | 0.1523 (auto) | 1.3× | 1.2× |
| pk4-n1000-G40 | 20.32 | 1.062 (84) | 1.062 | 0.575 | 0.575 (auto) | 0.5766 | 0.5766 (auto) | 1.8× | 1.8× |
| pk4-n3000-G40 | 46.13 | 3.258 (32) | 3.258 | 1.328 | 1.328 (auto) | 1.231 | 1.231 (auto) | 2.5× | 2.6× |
| pk3-n3000-G30-Lawvere | 4.604 | 0.2187 (84) | 0.2187 | 0.3459 | 0.3454 (levels) | 0.3707 | 0.3707 (auto) | 0.63× | 0.59× |
| pk3-n3000-G30-Probability | 2.794 | 0.2204 (84) | 0.2204 | 0.1421 | 0.1421 (auto) | 0.1581 | 0.1581 (auto) | 1.6× | 1.4× |
| grid4x100000-K3 | 2.508 | 0.8802 (32) | 0.8802 | 0.4243 | 0.1972 (contract) | 0.5506 | 0.262 (contract) | 2.1× | 1.6× |
| grid6x1000-K3 | 0.1766 | 0.137 (32) | 0.137 | 0.0739 | 0.0739 (auto) | 0.0739 | 0.0735 (contract) | 1.9× | 1.9× |
| grid5x1000-K4 | 0.4373 | 0.2847 (32) | 0.2847 | 0.0678 | 0.0678 (auto) | 0.0656 | 0.0656 (auto) | 4.2× | 4.3× |
| grid6x4000-K3 | 0.7072 | 0.5434 (1) | 0.5434 | 0.1494 | 0.1494 (auto) | 0.1445 | 0.1445 (auto) | 3.6× | 3.8× |
| miter-bug-10000 | 0.2478 | 0.1776 (32) | 0.1776 | 0.0301 | 0.0296 (contract) | 0.0438 | 0.0438 (auto) | 5.9× | 4.1× |
| miter-bug-30000 | 0.7808 | 0.51 (32) | 0.51 | 0.0937 | 0.0937 (auto) | 0.1324 | 0.1324 (auto) | 5.4× | 3.9× |
| miter-bug-100000 | 2.494 | 1.641 (32) | 1.641 | 0.343 | 0.343 (auto) | 0.5256 | 0.5256 (auto) | 4.8× | 3.1× |
| miter-eq-30000 | 0.7887 | 0.5403 (32) | 0.5403 | 0.0975 | 0.0975 (auto) | 0.1368 | 0.1347 (contract) | 5.5× | 3.9× |
| pk3-n500-K4 | 0.0031 | 0.002 (8) | 0.002 | 0.013 | 0.0129 (levels) | 0.0163 | 0.0163 (auto) | 0.15× | 0.12× |
| grid4x200-K3 | 0.0049 | 0.0042 (64) | 0.0042 | 0.0031 | 0.0031 (auto) | 0.0043 | 0.0042 (contract) | 1.4× | 0.98× |
| miter-bug-1000 | 0.0262 | 0.0202 (32) | 0.0202 | 0.0073 | 0.0072 (contract) | 0.0099 | 0.0099 (auto) | 2.8× | 2× |
| pk3-n200-K4 | 0.0012 | – (pending) | 0.0012 | 0.0068 | 0.0031 (contract) | 0.0089 | 0.0042 (contract) | 0.18× | 0.13× |
| pk3-n2000-K4 | 0.0125 | – (pending) | 0.0125 | 0.0169 | 0.0169 (auto) | 0.0222 | 0.0222 (auto) | 0.74× | 0.56× |
| pk3-n5000-K4 | 0.0318 | – (pending) | 0.0318 | 0.0217 | 0.0217 (auto) | 0.0313 | 0.0297 (levels) | 1.5× | 1× |
| pk3-n300-G30-Boolean | 0.258 | – (pending) | 0.258 | 0.0235 | 0.0235 (auto) | 0.0312 | 0.0312 (auto) | 11× | 8.3× |
| pk3-n300-G30-Lawvere | 0.4361 | – (pending) | 0.4361 | 0.1083 | 0.1082 (levels) | 0.12 | 0.12 (auto) | 4× | 3.6× |
| pk3-n300-G30-Probability | 0.4302 | – (pending) | 0.4302 | 0.1086 | 0.1083 (levels) | 0.1222 | 0.1216 (levels) | 4× | 3.5× |
| pk3-n1000-G30-Boolean | 1.121 | – (pending) | 1.121 | 0.0599 | 0.0597 (levels) | 0.0758 | 0.069 (levels) | 19× | 15× |
| pk3-n1000-G30-Lawvere | 2.367 | – (pending) | 2.367 | 0.7186 | 0.7177 (levels) | 0.7359 | 0.7357 (levels) | 3.3× | 3.2× |
| pk3-n1000-G30-Probability | 2.302 | – (pending) | 2.302 | 0.1863 | 0.1863 (auto) | 0.2055 | 0.2055 (auto) | 12× | 11× |
| pk3-n3000-G30-Boolean | 2.807 | – (pending) | 2.807 | 0.0986 | 0.0986 (auto) | 0.1165 | 0.1165 (auto) | 28× | 24× |
| pk3-n10000-G30-Boolean | 9.14 | – (pending) | 9.14 | 0.2301 | 0.2301 (auto) | 0.2428 | 0.2423 (levels) | 40× | 38× |
| pk3-n10000-G30-Lawvere | 16.53 | – (pending) | 16.53 | 1.091 | 1.085 (levels) | 1.097 | 1.097 (auto) | 15× | 15× |
| pk3-n10000-G30-Probability | 1.328 | – (pending) | 1.328 | 0.4396 | 0.4373 (levels) | 0.4408 | 0.439 (levels) | 3× | 3× |
| pk4-n1000-G40-Lawvere | 34.68 | – (pending) | 34.68 | 0.9597 | 0.9597 (auto) | 0.9914 | 0.9909 (levels) | 36× | 35× |
| pk4-n1000-G40-Probability | 31.84 | – (pending) | 31.84 | 0.7804 | 0.7804 (auto) | 0.8521 | 0.8521 (auto) | 41× | 37× |
| grid4x50-K3 | 0.0013 | – (pending) | 0.0013 | 0.0949 | 0.003 (contract) | 0.1215 | 0.0045 (contract) | 0.014× | 0.011× |
| grid4x1000-K3 | 0.0256 | – (pending) | 0.0256 | 0.0043 | 0.0043 (auto) | 0.0066 | 0.0061 (contract) | 6× | 3.9× |
| grid4x5000-K3 | 0.1235 | – (pending) | 0.1235 | 0.0094 | 0.0094 (auto) | 0.013 | 0.013 (auto) | 13× | 9.5× |
| grid4x20000-K3 | 0.4864 | – (pending) | 0.4864 | 0.0282 | 0.0282 (auto) | 0.041 | 0.0392 (contract) | 17× | 12× |
| grid4x20000-K3-Lawvere | 0.7592 | – (pending) | 0.7592 | 0.1445 | 0.1444 (contract) | 0.1448 | 0.1194 (contract) | 5.3× | 5.2× |
| grid4x20000-K3-Probability | 0.0182 | – (pending) | 0.0182 | 0.1166 | 0.1108 (contract) | 0.1692 | 0.1189 (contract) | 0.16× | 0.11× |
| grid6x1000-K3-Lawvere | 0.3406 | – (pending) | 0.3406 | 1.129 | 1.115 (levels) | 15.28 | 1.524 (levels) | 0.3× | 0.022× |
| grid6x1000-K3-Probability | 0.1174 | – (pending) | 0.1174 | 0.2605 | 0.2358 (levels) | 15.62 | 0.2992 (levels) | 0.45× | 0.0075× |
| miter-bug-100 | 0.0026 | – (pending) | 0.0026 | 0.0562 | 0.0038 (contract) | 0.0724 | 0.0058 (contract) | 0.046× | 0.036× |
| miter-bug-300 | 0.0077 | – (pending) | 0.0077 | 0.0045 | 0.0045 (auto) | 0.0062 | 0.0061 (contract) | 1.7× | 1.2× |
| miter-bug-3000 | 0.0752 | – (pending) | 0.0752 | 0.0128 | 0.0127 (contract) | 0.017 | 0.017 (auto) | 5.9× | 4.4× |
| many24-pk3-n20000 | 4.766 | 0.405 (84) | 0.405 | 0.1269 | 0.1269 (batched) | 0.1354 | 0.1354 (batched) | 3.2× | 3× |
| many24-grid4x2000 | 28.37 | – (pending) | 28.37 | 6.206 | 6.206 (batched) | 7.764 | 7.764 (batched) | 4.6× | 3.7× |

**Where the GPU wins (vs the full CPU node):** big shallow searches 1.2–2.6×; deep trees 1.9–5.9× on the RTX, 1.6–4.3× on the B200; 24 targets in one batched pass 3.0–3.2× vs 84 threads (pk3-n20000), 4.6× vs serial on grid4x2000. Against one core the semiring sweep is 3–41× (pk4-n1000-G40 Lawvere 36×, Probability 41×).

**Where it still loses:**

| case | best CPU | GPU (RTX auto) | why |
|---|---:|---:|---|
| pk3-n3000 → G30, Lawvere | 0.219 (84 thr) | 0.346 (0.63×; B200 0.59×) | semiring search enumerates every residual extension (no early exit as in Boolean); Lawvere is 2.4× slower than Probability on the same instance, because Probability underflows to 0 and zero values prune |
| pk3-n200/500/2000 → K4 | 1.2 / 2.0 / 12.5 ms | 6.8 / 13 / 17 ms | ~20 levels × 5–7 rounds of ~0.1 ms launches + host syncs; the GPU floor is ~3 ms (contraction) / ~7 ms (levels) |
| grid4x50 → K3, miter-bug-100 | 1.3 / 2.6 ms | 95 / 56 ms (contract: 3.0 / 3.8 ms) | `:auto` picks levels (height 179 / 143 < 256) at ~0.4–0.5 ms per level |
| grid4x20000 → K3, Probability | 18 ms | 117 ms | CPU result underflows to 0 early and prunes; the GPU dense contraction does the full L×L×L work |
| grid6x1000 → K3, Lawvere / Probability (L = 1458) | 0.34 / 0.12 s | 1.1 / 0.26 s; **B200 auto 15.3 / 15.6 s** | dense (non-bit-packed) composition at L = 1458 is 64× the Boolean work; on the B200 the dense contraction is 13× slower than on the RTX (to investigate: likely the CAS-based float atomics or FP64 path) |

Other findings:
- **`:auto` misses the better method** on small trees: levels vs contraction at height 9–180 (pk3-n200-K4 levels 6.8 ms vs contract 3.1 ms; grid4x50 95 vs 3.0 ms). grid4x100000 picks contraction with mate (0.42 s) while Wyllie/`:contract` best was 0.20 s.
- **Level-synchronous per-level cost** (deep trees, RTX): 0.25–0.55 ms per level (grid4x1000: 2.01 s / 3978 levels; miter-bug-3000: 1.88 s / 6010). Contraction on the same trees: 4–13 ms.
- **Giant bags (shallow big-Y):** the slowest single kernel is at the leaf level with 1.6–4.6M items: pk4-n3000-G40 kmax 0.355 s of 1.47 s kernel time (24%); pk3-n10000-G30 Probability 0.131 of 0.454 s (29%). Raising items further does not help: these kernels are tail-bound on a few items with fully fixed separators whose residual search is long. (Lawvere rows did not record stats: that path returns before the stats block; to fix.)
- **B200 vs RTX:** on shallow searches the B200 is within 0.8–1.1× of the RTX; deep-tree contraction is 1.3–1.5× slower on the B200, and the dense contraction above is 13× slower (lower clocks, launch-bound rounds).
- **Rerun of the old suites (`v4_*`):** contraction (wide) is unchanged (grid4x100000 23×, grid5x2000 14×, grid5x1000-K4 6.9×, grid7x1000 1.03×). Miter `e2e`: the GPU times are unchanged, but the CPU search measured in this job is 1.7–1.9× faster than in 48021453 (miter 10k: 0.22–0.24 s vs 0.37–0.40 s; it matches the separate CPU job 50755228), so the honest numbers are search 9–20×, end to end with MMD 2.7–4.1× (3.7–5.2× with prep cached) on the RTX. The 48021453 speedups above overstate by that factor. multi: pk3-n20000 24 targets batched 50× (RTX) / 39× (B200) vs one core; grid4x2000 4.5×.

### Proposed fix (not implemented): CPU fallback and method choice in `:auto`

All thresholds are cost-model terms whose constants come from a per-device measurement at first use (launch + sync latency, dense/packed product throughput), per the generic-GPU rule; the numbers below are the RTX / B200 calibration from this run.

1. **Method choice (levels vs contraction), replace `height < 256`:**
   - t_levels ≈ height × c_lev + kernel work, with c_lev = measured cost of one level (0.25–0.55 ms here; measure as ~5 launches + 1 sync).
   - t_contract ≈ c_c + Σ_t work(L_t), c_c ≈ 3–5 ms here, work = L_t² per bag for the local tables plus L³ per absorbed chain bag, /64 for bit-packed Boolean. L_t comes from the local-table count, which `prepare_contract` + the local pass already compute (cheap: 1–40 ms here).
   - Concretely: prefer contraction whenever it fits and maxL ≤ 4096 (Boolean, bit-packed) or maxL ≤ 256 (dense semiring); this flips grid4x50 (95 → 3 ms), miter-bug-100 (56 → 3.8 ms), pk3-n200-K4 (6.8 → 3.1 ms) and keeps every current win. Dense semiring with maxL > 256 goes to levels (grid6x1000 Lawvere: 15.3 → 1.5 s on the B200), and to the CPU when height × c_lev is also large (point 2).
   - Wyllie instead of mate when the tree is a few long chains with small L (grid4x100000: 0.42 → 0.20 s); pick by the number of mate rounds, which is known after the first batch.
2. **CPU fallback:**
   - **Tiny instances:** run the CPU serial search with a deadline equal to the device floor t_floor (≈ 2× min(c_c, c_lev × height); 6–10 ms here); if it finishes, return its answer, else start the GPU. Losses on the GPU side (pk3-n200/500/2000-K4, grid4x50, miter-bug-100: CPU 1.2–12.5 ms) would then run on the CPU, and a case where the GPU wins pays at most t_floor (pk3-n5000-K4: 22 ms → ≤ 32 ms, still ≈ CPU). A static pre-filter that skips the GPU when nt × |Y| < ~2·10⁵ element pairs covers the same rows without spending CPU time (pk3-n2000-K4: 1831 × 16 = 2.9·10⁴, loses; pk3-n300-G30: 276 × 274 = 7.6·10⁴ but wins 11×), so the deadline race is the safer rule.
   - **Deep tree + dense semiring with large L:** when maxL > 256 (dense) and height × c_lev > the serial estimate, use the CPU (grid6x1000 Lawvere/Probability: CPU 0.34 / 0.12 s vs best GPU 1.1 / 0.24 s).
   - **Lawvere on big shallow trees vs a threaded CPU:** the GPU still beats one core 3.3–36× on Lawvere, so the fallback should only trigger when `Threaded` CPU is available and the instance is mid-size (estimated serial time < ~5 s); the real fix is point 3.
3. **Residual-search split for giant bags** (the Lawvere and kmax losses): when an item's separator is fully fixed and its step budget runs out, split it on the first residual position into one sub-item per candidate, each accumulating a partial ⊕ into the parent's column slot with the semiring's atomic (Boolean: first-to-succeed flag; Lawvere: `atomicmin!`; Probability: `atomic_add!`). Trigger: kmax / kernel time > 20% at a level and that level has < 4 × (SMs × 8 × 128) items, i.e. the last kernels of pk4-n3000-G40 (24%) and pk3-n10000-G30 (29%). Expected: removes most of the 0.1–0.35 s tails, enough to move Lawvere pk3-n3000-G30 from 0.35 s to ≈ 0.2 s (parity with 84 threads); needs Lawvere stats first to confirm the share.


The branch is pushed to https://github.com/Optimax14/CSets.jl/tree/itay/gpu (rebased on origin/main 0d2c2a7; no PR).
