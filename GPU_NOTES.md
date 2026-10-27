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
