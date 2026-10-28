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

The rounds now run in batches of 8, with one host check per batch, and no `findall`. On the laptop this fixed grid 5×2000 (0.36 s → 0.022 s, 8× over the CPU). The HPG `wide` and `e2e` reruns with this version did not finish: SSH to HPG failed again.

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

Raw TSVs are on HPG under `/blue/fairbanksj/itaykadosh/gpu/csets/res/` (`v3a`, `v3b`); they were not copied here because SSH to HPG failed.
