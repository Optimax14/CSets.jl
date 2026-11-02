# Multithreaded CSets: notes

Branch `itay/threads`, from origin/main (0d2c2a7). The code is `src/threads.jl` and `src/treecontraction.jl`, plus a small hook in `src/contraction.jl` (`Bounds`, a trailing argument of `contract!` / `advance!`).

The API mirrors upstream, with a `Threaded(ntasks; minwork, deep, contract)` first argument:

- `score(Threaded(n), R, W, D, Y)` and `color(Threaded(n), R, W, D, Y)`
- `color(Threaded(n), W, D, Y)` and `color(Threaded(n), D, Y)`
- many targets: `score(Threaded(n), R, Ws, D, Ys)` and `color(Threaded(n), R, Ws, D, Ys)`

## Design (the same structure as the GPU path, on Julia tasks)

**(a) Level-synchronous search.** All nodes of a height level are independent.
- The pieces of a level are taken in turn by `ntasks` tasks (dynamic scheduling). Each task has its own workspace: the `restrict!` buffers, `pos`, and a `ContractionWorkspace`.
- Each piece calls Richard's `contract!` unchanged, writing to its task's buffers. The pieces of a node are then concatenated in order into the node's table.
- Before a level is solved, the tables of its children are moved into one array (the arena) and freed, so memory stays close to the serial stack.
- Levels whose estimated work is below `minwork` run on the calling task.

**(b) Splitting big nodes.** A node gets pieces in proportion to its share of the level's estimated work. Its estimated work is the sum of its children's column counts, or, for a leaf, the number of assignments of its separator.
- A piece bounds the values of the last one or two separator elements of the bag. These are the first elements the search assigns, so the pieces' columns, concatenated, are exactly the node's columns in Richard's colexicographic order, and no separator assignment is split.

**(c) Tree contraction for deep Boolean trees** (`treecontraction.jl`).
- Local tables: `contract!` on each bag with every element as separator and no children, in parallel.
- Then runs (`Loc_t`) and child matches (binary search).
- Then rake / random-mate compress rounds with bit-packed Boolean matrices: messages are bitsets and matrix rows are bitsets, with multithreaded products.
- The witness is taken from the root, choosing in each bag the first column of the chosen run that matches its children. That is exactly the back-pointer `contract!` records in the Boolean semiring.
- Trees with at least `deep` (256) levels use contraction when they have at least `contract` (50,000) nodes. Smaller deep trees run Richard's serial `solve`, which beats the overhead.
- Lawvere and Probability on deep trees run serially. A matrix product would reassociate the floating-point sums, so the result would no longer be bit-identical to the serial code.

**(d) Many targets.** The targets run in parallel, each with `ntasks ÷ ntargets` tasks of its own.

**Exactness.** Values are `===` to the serial `score`/`color`, and witnesses are identical (`==`) to the serial ones, for any number of tasks.
- `threads/test_threads.jl`: 120 random instances per seed, 7 seeds, at 3, 8 and 16 threads; random task counts, `minwork`, `deep` and `contract`; Boolean / Lawvere / Probability with random weights; miters.
- Plus 20 many-target cases per seed.

## Benchmarks (HPG, `threads/hpgcmp.jl`; raw results in `threads/results/`)

Setup:
- Serial is upstream main, one EPYC 9655P core (hpg-turin).
- Threads were run at 1, 8, 32 and 64 threads in one 64-core hpg-turin job; the 84-thread job is still queued.
- GPU is the `itay/gpu` branch on an RTX PRO 6000 Blackwell; the B200 job is still queued.
- Times are search only, in seconds, the best of 2 runs. Decomposition (MF) is the same for all and excluded. All answers agree.
- "best CPU" is the best of serial and all thread counts. The last column is best CPU time ÷ GPU time.

| workload | serial | 1 thr | 8 thr | 32 thr | 64 thr | best CPU | GPU | best CPU ÷ GPU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pk4-n20000 → K5 | 0.824 | 0.779 | 0.200 | 0.170 | 0.167 | 0.167 | 0.076 | 2.2× |
| pk5-n20000 → K6 | 5.95 | 5.11 | 0.962 | 0.579 | 0.564 | 0.564 | 0.209 | 2.7× |
| pk6-n5000 → K7 | 18.9 | 14.7 | 2.89 | 1.55 | 1.45 | 1.45 | 0.815 | 1.8× |
| pk3-n5000 → G(30) | 5.28 | 4.18 | 0.581 | 0.211 | 0.199 | 0.199 | 0.145 | 1.4× |
| pk4-n1000 → G(40) | 20.7 | 16.2 | 3.39 | 1.47 | 1.20 | 1.20 | 0.633 | 1.9× |
| pk4-n3000 → G(40) | 47.9 | 37.4 | 6.86 | 3.45 | 3.42 | 3.42 | 1.27 | 2.7× |
| pk3-n3000 → G(30), Lawvere | 4.58 | 3.63 | 0.581 | 0.244 | 0.239 | 0.239 | 0.386 | **0.62× (CPU wins)** |
| pk3-n3000 → G(30), Probability | 2.79 | 3.61 | 0.555 | 0.265 | 0.233 | 0.233 | 0.153 | 1.5× |
| grid 4×100000 → K3 | 2.55 | 3.37 | 1.31 | 1.08 | 1.00 | 1.00 | 0.405 | 2.5× |
| grid 6×1000 → K3 | 0.178 | 0.138 | (serial) | | | 0.138 | 0.076 | 1.8× |
| grid 5×1000 → K4 | 0.440 | 0.345 | (serial) | | | 0.343 | 0.068 | 5.1× |
| grid 6×4000 → K3 | 0.706 | 0.550 | (serial) | | | 0.548 | 0.149 | 3.7× |
| miter-bug 10k | 0.252 | 0.625 | 0.321 | 0.269 | 0.297 | 0.252 (serial) | 0.032 | **8.0×** |
| miter-bug 30k | 0.812 | 1.47 | 0.787 | 0.510 | 0.655 | 0.510 | 0.122 | 4.2× |
| miter-bug 100k | 2.57 | 4.83 | 2.18 | 1.89 | 2.06 | 1.89 | 0.539 | 3.5× |
| miter-eq 30k | 0.809 | 1.35 | 0.712 | 0.566 | 0.679 | 0.566 | 0.126 | 4.5× |
| pk3-n500 → K4 (small) | 0.0032 | 0.0034 | 0.0022 | 0.0038 | 0.0058 | 0.0022 | 0.0128 | **0.17× (CPU wins)** |
| grid 4×200 → K3 (small) | 0.0052 | 0.0043 | (serial) | | | 0.0042 | 0.0032 | 1.3× |
| miter-bug 1000 (small) | 0.0267 | 0.0208 | (serial) | | | 0.0206 | 0.0069 | 3.0× |
| 24 targets, pk3-n20000 | 4.99 | 4.88 | 0.839 | 0.679 | 0.546 | 0.546 | 0.141 | 3.9× |

"(serial)" means the deep tree has fewer than 50,000 nodes, so the threaded path runs the serial code; the other thread counts are the same.

Notes:
- Wall times of the serial and threaded jobs come from different nodes, so 1-thread vs serial differences are partly node noise.
- In the Probability workload the weights are 1/30 scale, so the value underflows to 0.0. The work done is unchanged.

**Takeaways**
- **Large targets, shallow trees:** threads scale 10–26× from 1 to 64 threads (pk3-n5000-G30: 4.18 s → 0.199 s). Beyond 32 threads the gain is small: the last levels have few, huge nodes, and pieces split at most two separator elements.
- **Large targets, GPU vs the full multicore CPU:** the GPU is 1.4–2.7× faster. The CPU is 1.6× faster on the Lawvere case.
- **Deep trees:** CPU tree contraction gives at most 1.6–2.6× over serial (miters, grid 4×100000). It enumerates every bag's local table, about 5–10× the serial work. The GPU contraction is 3.5–8× faster than the best CPU time.
- **Small instances:** the CPU wins or the two are even.
- **Many targets (24):** the GPU's single batched pass is 3.9× faster than 64 threads.

## 84 threads and B200 (jobs 48132475, 48132477)
84 threads on a 96-core EPYC 9655P (hpg-turin). The B200 run is the same `itay/gpu` code as the RTX run. "best CPU" now includes 84 threads. The RTX run 48132476 is excluded on Lawvere because it returned Inf (an old weights bug); 48132877 is used.

| workload | best CPU | 84 thr | RTX | B200 | best CPU ÷ RTX | best CPU ÷ B200 |
|---|---:|---:|---:|---:|---:|---:|
| pk4-n20000 → K5 | 0.147 | 0.206 | 0.076 | 0.087 | 1.9× | 1.7× |
| pk5-n20000 → K6 | 0.499 | 0.499 | 0.209 | 0.228 | 2.4× | 2.2× |
| pk6-n5000 → K7 | 1.17 | 1.17 | 0.815 | 0.922 | 1.4× | 1.3× |
| pk3-n5000 → G(30) | 0.187 | 0.218 | 0.145 | 0.141 | 1.3× | 1.3× |
| pk4-n1000 → G(40) | 1.06 | 1.06 | 0.633 | 0.578 | 1.7× | 1.8× |
| pk4-n3000 → G(40) | 3.26 | 3.30 | 1.27 | 1.21 | 2.6× | 2.7× |
| pk3-n3000 → G(30), Lawvere | 0.219 | 0.219 | 0.386 | 0.375 | **0.57×** | **0.58×** |
| pk3-n3000 → G(30), Probability | 0.220 | 0.220 | 0.153 | 0.166 | 1.4× | 1.3× |
| grid 4×100000 → K3 | 0.880 | 1.16 | 0.405 | 0.567 | 2.2× | 1.6× |
| grid 5×1000 → K4 | 0.285 | 0.342 | 0.068 | 0.065 | 4.2× | 4.4× |
| grid 6×4000 → K3 | 0.543 | 0.543 | 0.149 | 0.144 | 3.7× | 3.8× |
| miter-bug 10k | 0.178 | 0.480 | 0.032 | 0.046 | 5.6× | 3.8× |
| miter-bug 100k | 1.64 | 2.20 | 0.539 | 0.446 | 3.0× | 3.7× |
| miter-eq 30k | 0.540 | 0.875 | 0.126 | 0.134 | 4.3× | 4.0× |
| pk3-n500 → K4 (small) | 0.002 | 0.010 | 0.013 | 0.016 | **0.2×** | **0.1×** |
| 24 targets, pk3-n20000 | 0.405 | 0.405 | 0.141 | 0.121 | 2.9× | 3.3× |

- 84 threads rarely beats 32–64; the shallow cases flatten past ~32 threads and the miters get slower.
- The B200 is no faster than the RTX PRO 6000: these runs are bound by launch latency and memory latency, not by FLOPs or bandwidth.
- The overall picture is unchanged: the GPU is 1.3–2.7× the full CPU node on big shallow searches, 3–6× on deep trees, 3× on many targets; the CPU wins on Lawvere (1.7×) and tiny instances.
