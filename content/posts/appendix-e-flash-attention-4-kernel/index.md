---
title: "Appendix E: Anatomy of the FlashAttention-4 Blackwell Kernel"
date: 2026-06-06
draft: true
math: true
toc: true
tocDepth: 3
tags: ["flash-attention", "blackwell", "cutedsl", "cutlass", "tmem", "mbarrier", "warp-specialization", "tcgen05", "tma", "mma"]
description: "A ground-up reading guide to the FA4 CuTeDSL forward kernel on Blackwell (SM100): the host/device split, the memory hierarchy (incl. TMEM), the CuTe naming grammar, online softmax, warp specialization, mbarrier pipelines, fragments vs descriptors, atoms, and the loop hierarchy."
weight: 1095
---

## TL;DR

FlashAttention-4's forward kernel (in `flash_attn/cute/flash_fwd_sm100.py`) is written in **CuTeDSL** and JIT-compiled for **Blackwell (SM100)**. It looks impenetrable at first because it combines several unfamiliar ideas at once: a four-level memory hierarchy including Blackwell's **tensor memory (TMEM)**, a terse **layout-algebra naming convention**, **warp specialization** (different warps do different jobs), and **mbarrier pipelines** for synchronization. This appendix builds those fundamentals from the ground up so the kernel reads as ordinary code. It is a *reading guide*, not a line-by-line API reference.

Prerequisites: the [GPU Architecture Primer]({{< ref "appendix-a-gpu-architecture" >}}) (SMs, memory, tensor cores, TMA) and [Flash Attention v1 & v2]({{< ref "09-flash-attention-v1-v2" >}}) (tiling, online softmax). For the algorithmic v3/v4 story, see [Flash Attention v3 & v4]({{< ref "10-flash-attention-v3-v4" >}}).

---

## 1. Two worlds: host code vs. device code

The single most important framing: there are two distinct layers, and only one of them does math.

| | **Host (CPU) code** | **Device (GPU) code** |
|---|---|---|
| Where | `interface.py`; the `__init__` / `__call__` methods of the kernel class | the `@cute.kernel` method and its helpers |
| Runs | once per call, in Python | massively parallel, on the GPU |
| Job | validate inputs, choose config, JIT-compile/cache, launch | perform the attention computation |

The host entry point `_flash_attn_fwd` does **no math**. It normalizes inputs, resolves options (`causal`, tile sizes, `num_splits`, `pack_gqa`, …), builds a **compile key**, and — because kernels are JIT-compiled and specialized to compile-time constants — looks that key up in a cache. Each distinct combination of (dtype, head_dim, causal, tile sizes, arch, …) is a *different compiled kernel*. It then constructs the right class (`FlashAttentionForwardSm100` for Blackwell) and calls it.

The kernel class itself has three parts:

- **`__init__`** — pure Python on the host. Computes compile-time constants: padded head dims, MMA tiler shapes, the TMEM column layout, and the warp-role assignment.
- **`__call__`** (`@cute.jit`) — still host-side, but the CuTeDSL part: builds shared-memory layouts, TMA descriptors, tiled-MMA objects, the tile-scheduler params, and **launches the grid**.
- **`kernel`** (`@cute.kernel`) — the actual GPU code that every thread runs.

Everything else in this appendix is about reading that third part.

---

## 2. The memory hierarchy

Data must travel through progressively smaller, faster memories before the math happens, and back out afterward. Four levels matter on Blackwell:

| Memory | What it is | Scope | Size (order of) | Speed |
|---|---|---|---|---|
| **Global / HBM** | the Q/K/V/O tensors as passed in | whole GPU | 10s–100s of GB | slow |
| **Shared memory (SMEM)** | staging buffers (tiles pulled in to work on) | per thread block | ~228 KB / block | fast |
| **Registers** | per-thread working values | per thread | ~1 KB / thread | fastest |
| **TMEM** (tensor memory) | Blackwell's dedicated MMA accumulator memory | per thread block | 512 columns × 128 rows | fast, MMA-coupled |

The whole point of FlashAttention is to **never materialize the full $n \times n$ score matrix**. The kernel pulls one 128-wide block of K and V into SMEM at a time, computes against it, updates a running result, discards it, and moves on. SMEM only ever holds a few small tiles.

**TMEM** is the Blackwell-specific piece. The 5th-generation tensor cores (`tcgen05`) write their results into a dedicated on-chip memory rather than into registers. The kernel hand-allocates this 512-column space in `__init__`:

```
TMEM columns:  0 ──── 128 ──── 256 ──── 384 ──── 512
               │  S0  │  S1   │   O0   │   O1   │
               └──┬───┴───┬───┴────────┴────────┘
                  P0      P1     (P overlaps the right half of each S region)
```

- `tmem_s_offset = [0, 128]` — score accumulators `S0`, `S1`.
- `tmem_o_offset = [256, 384]` — output accumulators `O0`, `O1`.
- `tmem_p_offset = [64, 192]` — the probabilities `P` are written **on top of** the score region (P is bf16, half the width of fp32 scores, so it packs in).

---

## 3. The naming grammar

CuTe code uses a systematic naming convention. Once you see the grammar, dense names like `tSrKi` read like English. Two pieces: a **memory-space prefix** and the **MMA-partition grammar**.

### 3.1 Memory-space prefix

| Prefix | Meaning | Example |
|---|---|---|
| `m` | the full tensor in global memory | `mK` = the entire K tensor |
| `g` | a *tiled view* of a global tensor (carries a block index) | `gK` = K, indexable by block |
| `s` | shared memory | `sK` = a block of K in SMEM |
| `t…` | a thread-partitioned tensor (a fragment) | `tSrK` (see below) |

A `_cur` suffix means "the current work-tile (or current ring-slot) slice": `mK_cur` is K with head/batch fixed; `sK_cur` is one ring slot of `sK`.

So the lifecycle of K is: `mK` → `mK_cur` (fix head+batch) → `gK` (tile into blocks) → TMA-copied into an `sK` ring slot → read by the MMA as `tSrKi`. **Only the TMA copy moves data**; the rest are layout views.

`m_cur` vs `g` is purely layout, no copy:
- `mK_cur` = `mK` with head and batch indexed away → the 2D `(seqlen_k, head_dim)` matrix for one (head, batch).
- `gK` = that matrix re-expressed as a stack of tiles via `local_tile`, shape `(n_block_size, head_dim, num_n_blocks)`, so `gK[:, :, n]` is key block `n`.

### 3.2 The `t…` grammar

Any name starting with `t` is a **per-thread partition**, and the letters after it form a 3-part code:

```
  t  S  r  K
  │  │  │  └── logical data:  Q, K, V, S, P, O
  │  │  └───── memory:  r = register/fragment, s = smem, g = gmem, t = tmem
  │  └──────── which MMA:  S = the Q·Kᵀ matmul (produces S)
  │                        O = the P·V matmul (produces O)
  └─────────── "thread-partitioned tensor"
```

There are **two** matmuls in attention, so operands are tagged by which one consumes them:

| Name | Decode | Meaning |
|---|---|---|
| `tSrQ` | tS·r·Q | Q·Kᵀ matmul, registers, **Q** (operand A) |
| `tSrK` | tS·r·K | Q·Kᵀ matmul, registers, **K** (operand B) |
| `tSrKi` | `tSrK[…, Ki_index]` | the K fragment for ring slot `Ki_index` (trailing `i` = "current iteration") |
| `tStS` | tS·t·S | Q·Kᵀ matmul, **TMEM**, **S** (the accumulator) |
| `tOrP` | tO·r·P | P·V matmul, registers, **P** (operand A) |
| `tOrV` | tO·r·V | P·V matmul, registers, **V** (operand B) |
| `tOtO` | tO·t·O | P·V matmul, **TMEM**, **O** (the accumulator) |

The generic CuTe helpers use `tCrA` / `tCrB` / `tCrC` instead, where `C` names the MMA by its accumulator. So `tCrB` = "operand-B fragment," which in our two matmuls is K (for Q·Kᵀ) or V (for P·V). A `_t2r` suffix marks a tensor participating in a TMEM→register copy (`tStS_t2r` source, `tSrS_t2r` register destination).

### 3.3 Logical vs. physical indices

A recurring source of confusion is that "index" can mean two unrelated things:

- **Logical indices** point at real data: `m_block` (which query-row block), `n_block` (which key block of K), `stage` ∈ {0,1} (which Q subtile).
- **Physical indices** point at scratch buffers: the **ring slot** (`Ki_index` = `count mod kv_stage`) is just *where in the SMEM ring a block currently sits*. It's recycled; the mapping logical→physical is many-to-one over time.

`tSrKi` (`Ki_index`, physical slot) and `sK_cur` (same slot, raw SMEM tensor) refer to the same K block — different views of one ring slot.

---

## 4. Online softmax

This is the algorithm from [FA v1/v2]({{< ref "09-flash-attention-v1-v2" >}}); recapped here because it drives the kernel's structure.

We want, per query row:

$$O = \mathrm{softmax}(Q K^\top \cdot \text{scale}) \, V$$

We stream K/V in blocks and maintain three running quantities per row: $m$ (running max, for stability), $\ell$ (running sum of $e^{s-m}$, the softmax denominator), and $O$ (running weighted sum of value rows). For each new block with scores $s$:

$$
\begin{aligned}
m_{\text{new}} &= \max(m,\ \mathrm{rowmax}(s)) \\
c &= e^{\,m_{\text{old}} - m_{\text{new}}} \quad (\le 1, \text{ the *correction* factor}) \\
\ell &\leftarrow \ell \cdot c + \mathrm{rowsum}\!\left(e^{\,s - m_{\text{new}}}\right) \\
O &\leftarrow O \cdot c + e^{\,s - m_{\text{new}}} V
\end{aligned}
$$

After the last block: $O \leftarrow O / \ell$ and $\mathrm{LSE} = m + \log \ell$.

The kernel works in **base-2** (the hardware has a fast `exp2`), folding $\log_2 e$ into the scale (`softmax_scale_log2`). The math is identical; only `exp`→`exp2` and `log`→`log2` (with a final `·ln2`).

### Worked example

Scores `[1, 3, 2, 5]`, values `[10, 20, 30, 40]`, streamed in two blocks of two.

Block 1, scores `[1,3]`, $m=3$:
$$\ell = e^{-2}+e^{0} = 1.135, \qquad O = 0.135\cdot 10 + 1\cdot 20 = 21.35$$

Block 2, scores `[2,5]`, $m_{\text{new}}=5$, $c=e^{3-5}=0.135$:
$$\ell = 1.135\cdot 0.135 + (e^{-3}+e^0) = 1.203, \qquad O = 21.35\cdot 0.135 + (0.05\cdot 30 + 1\cdot 40) = 44.38$$

Normalize: $O/\ell = 44.38 / 1.203 = 36.88$ — exactly the all-at-once answer. And $\mathrm{LSE} = 5 + \log 1.203 = 5.185 = \log(e^1+e^3+e^2+e^5)$. ✓

The correction factor $c$ is the protagonist of the kernel: every time the max rises, previously accumulated $\ell$ and $O$ must be rescaled down. Crucially, **rescaling $O$ (128×128 values in TMEM) is independent work from computing `exp`** — which is why those two jobs land on different warps (§6).

---

## 5. LSE and SplitKV

`LSE = m + log(ℓ)` recovers the true log-denominator $\log \sum_j e^{s_j}$ from the two running quantities. It's needed for two reasons:

1. **The backward pass** recomputes softmax weights as $w_j = e^{s_j - \mathrm{LSE}}$ — one number per row replaces the whole score matrix. (`interface.py` forces `lse` to exist when `requires_grad`.)
2. **Combining partial attentions** — which is what SplitKV does.

### SplitKV

During decode (batch≈1, one query token, huge KV cache) there is ~1 work tile but ~132 SMs — terrible utilization. **SplitKV** chops the *key* dimension into `num_splits` pieces, each handled by a different SM. Each split $s$ writes a partial output $O_s$ (normalized over its own keys) and a partial $\mathrm{LSE}_s$. A small combine kernel (`flash_fwd_combine.py`) merges them.

The merge is a **weighted average with weights $e^{\mathrm{LSE}_s}$** (each split's total softmax mass), stabilized by the max LSE $M$:

$$
O = \frac{\sum_s e^{\mathrm{LSE}_s - M}\, O_s}{\sum_s e^{\mathrm{LSE}_s - M}}, \qquad \mathrm{LSE} = M + \log\!\sum_s e^{\mathrm{LSE}_s - M}
$$

This is literally a softmax *over the per-split LSEs* used as blend weights — the same online-softmax merge as §4, one level up. Taking the §4 example and splitting it into two splits reproduces $O = 36.88$ exactly, with the high-scoring split getting weight ≈ 1.

> **Throughline:** the LSE/running-max machinery is the same idea at three scales — merging K blocks inside one warp's loop, merging splits across SMs, and recomputing weights from one saved number in the backward pass.

---

## 6. Warp specialization

A normal CUDA kernel runs the same code on all threads. This kernel does the opposite: it's a **warp-specialized persistent kernel** — the block's 16 warps (512 threads) are divided into roles, like an assembly line:

| Warps | Role | Job |
|---|---|---|
| 0–3 | `softmax0` | softmax for Q-subtile 0 |
| 4–7 | `softmax1` | softmax for Q-subtile 1 |
| 8–11 | `correction` | rescale `O` when the running max moves; final `O/ℓ` |
| 12 | `mma` | drive the tensor cores (both matmuls) |
| 13 | `epilogue` | write `O` to global memory |
| 14 | `load` | drive TMA loads of Q/K/V |
| 15 | `empty` | idle / repurposed (e.g. CLC scheduler) |

The kernel body branches on `warp_idx` and dispatches each warp to its loop (`load`, `mma`, `softmax_loop`, `correction_loop`, `epilogue_s2g`). These run **concurrently** and hand work to each other through pipelines (§7). The online-softmax recurrence is split across three compute warps:

- **MMA** does the two matmuls (`Q·Kᵀ→S`, `P·V→O`).
- **softmax** does max/exp/sum and produces the correction factor.
- **correction** applies that factor to the `O` accumulator and does the final normalization.

While `correction` rescales `O` for block $i$, `softmax` is exp-ing block $i{+}1$, `mma` is matmul-ing block $i{+}2$, and `load` is fetching block $i{+}3$. That overlap is *why it's fast*.

---

## 7. mbarriers and pipelines

Because warps run independently at different speeds, they need to signal "this buffer is ready" / "this buffer is free." That's an **mbarrier**: a 64-bit synchronization object in shared memory holding an arrival/transaction count and a **phase bit**. Threads (or async hardware) *arrive* on it; waiters *wait* on `(barrier, expected_phase)`; completion flips the phase.

Unlike `__syncthreads()` (which syncs the whole block), an mbarrier lets a specific set of producers signal a specific set of consumers — exactly what warp specialization needs.

### Producer/consumer pipelines

The kernel wraps a **pair** of mbarriers per ring slot into a `Pipeline`:

- **empty** mbarrier (consumer → producer): "slot is free to overwrite"
- **full** mbarrier (producer → consumer): "data is ready"

The four operations:

| Call | Meaning |
|---|---|
| `producer_acquire` | wait on *empty* (slot free?) |
| `producer_commit` | arrive on *full* (data ready) |
| `consumer_wait` | wait on *full* (data ready?) |
| `consumer_release` | arrive on *empty* (slot free) |

### Transaction count

For TMA loads, the *full* mbarrier completes when the expected **byte count** (`tx_count`) has been deposited, rather than on a thread arrival. The TMA engine bumps the counter as data lands; no thread is involved. This is how a wait synchronizes on async DMA.

### Worked handshake: one K tile

Following K-block 0 into ring slot 0 (TMA path):

1. **Load warp** `producer_acquire`s slot 0 (waits on *empty*; pre-satisfied at launch).
2. **Load warp** issues the TMA copy, handing it slot 0's *full* mbarrier as the completion target — and **returns immediately** (no `producer_commit`; the hardware rings *full* via `tx_count`).
3. **MMA warp** `consumer_wait`s on slot 0's *full* mbarrier; sleeps if the data hasn't landed.
4. TMA finishes → *full* completes, phase flips → MMA warp wakes.
5. MMA warp reads SMEM in the Q·Kᵀ GEMM (data guaranteed present and ordered).
6. MMA warp `consumer_release`s (arrives on *empty*) → slot free for reuse with a flipped phase.

`consumer_wait` **blocks the calling warp only** (not the SM): the warp goes ineligible and the scheduler runs other warps; if the data already arrived, it falls through with no stall.

### Ring buffers

`sK`/`sV` aren't a single tile — they have a trailing dimension of size `kv_stage` (the number of slots). A **ring buffer** indexes these slots *modulo `kv_stage`*: after slot `N-1`, the next write wraps to slot 0. The producer/consumer states track `(slot_index, phase)`; `.advance()` increments mod `kv_stage` and flips the phase on wrap. The phase distinguishes a slot's new occupant from its previous one.

Depth = how far the producer can run ahead = how much memory latency you hide. Example (`kv_stage=4`, K and V sharing one ring, 3 blocks streamed backwards):

| issue | item | block | slot | phase |
|---|---|---|---|---|
| 1 | K | 2 | 0 | p |
| 2 | V | 2 | 1 | p |
| 3 | K | 1 | 2 | p |
| 4 | V | 1 | 3 | p |
| 5 | K | 0 | 0 | ¬p ← wrapped |
| 6 | V | 0 | 1 | ¬p |

The pipelines connecting the compute warps: `pipeline_q`/`pipeline_kv` (load→MMA), `pipeline_s_p_o` (MMA↔softmax+correction, used bidirectionally), `pipeline_sm_stats` (softmax→correction), `pipeline_o_acc` (MMA→correction), `pipeline_o_epi` (correction→epilogue).

---

## 8. Warp scheduling

Why warp specialization works at all comes down to how SMs schedule warps.

- An SM has **4 sub-partitions**, each with its own warp scheduler issuing **one instruction per clock cycle** from one selected warp. Warps are assigned to a sub-partition at launch (≈ `warp_id mod 4`) and stay there.
- **Every cycle**, each scheduler classifies its resident warps as *eligible* or *stalled* (via a scoreboard) and selects one eligible warp to issue. There is no time slice — it's a fresh per-cycle pick.
- **Switching is free**: all resident warps' registers are physically present at once (this is why register usage caps occupancy), so there's no save/restore. While one warp's 500-cycle global load is outstanding, the scheduler issues from others — *latency hiding*.
- A stalled `mbarrier.try_wait` makes the warp ineligible (a hardware sleep, not a busy-spin); the SM stays productive running co-resident warps.

So when the MMA warp blocks on `consumer_wait`, its sub-partition's scheduler simply skips it each cycle and runs load/softmax/correction warps that share that sub-partition; when the mbarrier flips, the warp is eligible again as soon as the next cycle.

---

## 9. Tensors are layouts: transpose is free

A CuTe tensor is a **pointer** (`iterator`) plus a **layout** (shape + strides). A transpose/permute/reshape via `make_tensor` + layout select changes only the layout — **no bytes move**:

```python
# (b, s, h, d)  →  (s, d, h, b), a stride-only view
mK = cute.make_tensor(mK.iterator, cute.select(mK.layout, mode=[1, 3, 2, 0]))
```

The PyTorch input is `(batch, seqlen, heads, head_dim)` (batch first). The host permutes it to `(seqlen, head_dim, heads, batch)` — **batch last** — for Q and K, and to `(head_dim, seqlen, heads, batch)` for V. Why batch last? So per-work-tile selection `mK[None, None, head, batch]` fixes the *trailing* modes, leaving the `(seqlen, head_dim)` matrix to tile. The permutation is chosen to match how the kernel peels off (head, batch) and tiles the rest.

This is the same idea as PyTorch's `.transpose()` returning a non-contiguous view; the actual physical data movement is deferred to the TMA copy into SMEM, which is arranged to hit efficient access patterns.

**`Major.K` vs `Major.MN`.** SMEM descriptors declare which dimension is contiguous. The contraction dim of `Q·Kᵀ` is head_dim, so Q and K are `Major.K`. The contraction of `P·V` is the sequence dim, so V must present head_dim along the output axis — `Major.MN`. That is *why* V gets the extra transpose: the layout permute, the descriptor's `Major` flag, and the matmul's operand requirement are one decision viewed at three levels.

---

## 10. The MMA: fragments vs. descriptors

Two ways operand data is presented to the tensor core:

- **Fragment** — *per-thread* data held in registers; each thread owns *different* elements of the tile. The `t…r…` names.
- **Descriptor** — a compact, *uniform* spec (SMEM base address + layout + swizzle; or a TMEM address; or the instruction descriptor `idesc` for op config). It's a *reference*, the same value across threads; the hardware does the loading.

The distinguishing test: do the threads hold the **same** value (descriptor) or **different** values (fragment)?

The MMA-issue model changed across generations:

| Instruction | Arch | Issued by | Operand source | Accumulator | Sync |
|---|---|---|---|---|---|
| `mma.sync` | Ampere | all 32 threads (warp) | registers (fragments) | registers | sync |
| `wgmma` | Hopper | all 128 threads (warpgroup) | A: reg or SMEM-desc; B: SMEM-desc | registers (distributed) | async |
| `tcgen05.mma` | Blackwell | **a single thread** | A: SMEM-desc or TMEM; B: SMEM-desc | **TMEM** | async |

On Blackwell, one elected thread issues the MMA with descriptors + a TMEM accumulator address; the tensor core fetches operands from SMEM/TMEM itself and accumulates asynchronously, signalling completion via an mbarrier. **This is why the kernel dedicates a single MMA warp** instead of a whole warpgroup — and why the matmul needs *no* per-thread register fragments. The Q·Kᵀ call maps to: `qk_mma_idesc` (op), Q/K SMEM descriptors, `tmem_s_offset[stage]` (accumulator), `zero_init`, `cta_group`.

> A note on CuTe naming: `make_fragment_A/B/C` produces a thread-partitioned operand *view* regardless of backing store. For register operands it's literal registers; for SMEM operands (`tSrQ`, `tSrK` on Blackwell) it's a layout view used to build the descriptor. To find where bytes actually live, look at the `s…`/`…t…` tensor it was built from.

### Why per-thread granularity persists even for a collective MMA

The MMA being warp/single-thread issued doesn't remove the need for per-thread fragments, because (1) you program per-thread (SIMT) — even a collective instruction is *expressed* per-thread; (2) a tile is far too large for one thread's registers (a 128×128 fp32 tile is 64 KB vs ~1 KB/thread) and the hardware distributes it; and (3) all the elementwise work *around* the MMA — softmax reading S from TMEM, correction rescaling O, epilogue writing O — is ordinary distributed per-thread code. The `t` (thread) granularity never goes away because the *tiles* flowing in and out of the MMA must be partitioned per thread for everything else.

---

## 11. Atoms and the CuTe build hierarchy

An **Atom** is one indivisible hardware instruction bundled with its thread-value (TV) layout — the map `(thread_id, value_id) → coordinate in the tile`. Two kinds: **MMA atoms** (one `mma.sync`/`wgmma`/`tcgen05.mma`) and **copy atoms** (one `ldmatrix`/`cp.async`/TMA copy/TMEM `tcgen05.ld`/`st`).

### Objects vs. operations

This trips everyone up: `Atom`, `TiledMMA`/`TiledCopy`, `ThrMMA`/`ThrCopy` are **types/objects** — layout descriptions, resolved at compile time. They *do nothing*. The only **operations** are the verbs `cute.copy(...)` and `cute.gemm(...)`, which consume those objects and emit the actual instructions.

The four-beat build hierarchy:

1. **Atom** — pick the hardware instruction (+ dtype): `cute.make_copy_atom(op, dtype)`.
2. **Tiled** — replicate the atom across threads to cover a full tile → a plan: `make_tiled_copy_tv(...)` or `tcgen05.make_tmem_copy(atom, tensor)`.
3. **Thr** — slice the plan to one thread: `.get_slice(tidx)`.
4. **Partition** — bind a tensor: `.partition_S/D(...)` or `.partition_A/B/C(...)` / `make_fragment_*` → per-thread fragments.

Then `cute.copy` / `cute.gemm` runs it. Steps 1–3 produce only objects; step 4 binds memory; the verb is the only thing that survives to runtime as GPU instructions.

`get_slice` vs `partition`: `get_slice(tidx)` fixes **which thread** (data-agnostic — it narrows the TV map); `partition_S/D(tensor)` binds **which buffer** (projects this thread's elements into that tensor's memory). They're separate because one `ThrCopy` is reused across the source *and* the destination (different layouts, same TV map), guaranteeing `src_part[i]` ↔ `dst_part[i]`. (`partition_S` = Source, `partition_D` = Destination; both are layout projections, no data movement.)

> Subtlety: when a tensor is passed to `make_tmem_copy(atom, tSAcc)`, it's used as a **layout template** (TMEM tiling is layout-dependent) — its layout, not its data. The same tensor reappears at `partition_S(tSAcc)` where its *memory* is what matters. `get_slice` in between binds no data.

### Worked example: loading scores from TMEM to registers

The softmax warps read the 128×128 `S` tile out of TMEM into per-thread registers:

```python
tmem_load_atom = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.qk_acc_dtype)  # 1. atom: one tcgen05.ld
thr_tmem_load  = tcgen05.make_tmem_copy(tmem_load_atom, tSAcc).get_slice(tidx)          # 2+3. tiled, then this thread
tStS_t2r       = thr_tmem_load.partition_S(tSAcc)                                        # 4. this thread's TMEM source addrs
tSrS_t2r       = cute.make_fragment(thr_tmem_load.partition_D(tScS).shape, self.qk_acc_dtype)  # register destination
cute.copy(thr_tmem_load, tStS_t2r, tSrS_t2r)                                             # run: emit the tcgen05.ld(s)
```

After this, `tSrS_t2r.load()` gives each thread its scores in registers; softmax math runs on them.

### Repetition

`Ld32x32bOp` is the **base access shape** (32 lanes × 32 bits = one column's worth per lane). `Repetition(N)` (the PTX `.xN` modifier) packs N copies of that base access into **one** instruction, reading **N contiguous TMEM columns** into N consecutive registers per thread. It's a tuning knob trading registers-per-instruction against instruction count; contiguity is what makes the wide load efficient. The kernel varies it by purpose: score load `Repetition(32)`, correction rescale `Repetition(16)`, P store `Repetition(8|16)`.

---

## 12. The loop hierarchy

There are exactly **three** levels of iteration, plus a derived physical counter:

```
while work_tile.is_valid_tile:                   # LEVEL 1: work tiles (from the scheduler)
    m_block, head, batch, split = work_tile.tile_idx
    n_block_min, n_block_max = get_n_block_min_max(...)

    for i in range(n_block_max - n_block_min):    # LEVEL 2: stream key blocks (the online-softmax loop)
        for stage in range(q_stage):              # LEVEL 3: Q subtiles (constexpr, unrolled, 1 or 2)
            ... QK and PV matmuls ...

    work_tile = tile_scheduler.advance_to_next_work()
```

| Index | Level | Kind | Meaning |
|---|---|---|---|
| `m_block`, `head`, `batch`, `split` | 1 | logical coordinate | identify a work tile; distributed across SMs by the scheduler — **not** an inner loop |
| `n_block` | 2 | logical loop | which key block; the streaming online-softmax loop |
| `stage` (`q_stage`) | 3 | logical, unrolled | which Q subtile (0/1) |
| ring index (`.index`) | — | **physical** counter | which SMEM slot a block sits in; `count mod kv_stage`, auto-advances |

**The ring index is not a loop.** It's a physical address that rides along: each Level-2 iteration consumes one K and one V, advancing the ring state twice (mod `kv_stage`). The thing actually looped over is `n_block`; the ring index is dragged behind it. Total work tiles = `num_m_blocks × num_heads × batch × num_splits`, divided across the ~132 persistent CTAs by the tile scheduler (the Level-1 loop).

---

## 13. `q_stage` and ping-pong-style overlap

`q_stage = 2` processes **two independent Q subtiles** (256 query rows per CTA), each owned by its own softmax warp group, sharing the same K/V loads. Its purpose is **MMA/softmax overlap**: while `softmax0` exp's `S0` (SFU + TMEM traffic), the MMA warp produces `S1` (tensor core), and `softmax1` picks it up — neither hardware unit idles waiting for the other.

This is the goal of Hopper's **ping-pong** scheduling, but the mechanism differs:

- Hopper ping-pong = two symmetric warpgroups, each doing full attention, staggered so one is in its GEMM phase while the other is in its softmax phase, swapping via a scheduler barrier.
- Blackwell here = one centralized async MMA warp feeding two softmax streams (`q_stage=2`). Not role-swapping warpgroups.

There is also an *optional* explicit two-softmax stagger (`s0_s1_barrier` / `pipeline_s0_s1_sequence`) — the most ping-pong-like piece — but it's **disabled by default**.

---

## Glossary / quick reference

| Term | One-liner |
|---|---|
| **TMEM** | Blackwell's dedicated on-chip MMA accumulator memory (512 cols × 128 rows / block) |
| **TMA** | async DMA engine that streams tiles global↔SMEM; rings an mbarrier via byte count |
| **mbarrier** | 64-bit SMEM sync object: arrival/transaction count + phase bit; arrive/wait |
| **phase bit** | flips each time a reused slot completes; distinguishes new vs. old occupant |
| **tx_count** | "complete when N bytes deposited" — how a wait syncs on async TMA |
| **ring buffer** | `kv_stage` SMEM slots indexed mod `kv_stage`; depth = prefetch/overlap |
| **fragment** | per-thread, register-resident slice of a tile (threads hold *different* data) |
| **descriptor** | uniform spec (SMEM/TMEM address + layout) the MMA uses to fetch operands itself |
| **atom** | one hardware instruction + its thread-value layout (MMA atom / copy atom) |
| **TiledMMA/ThrMMA** | objects (compile-time layout plans), not operations; `cute.gemm`/`cute.copy` are the verbs |
| **`q_stage`** | number of Q subtiles processed together (1 or 2); enables MMA/softmax overlap |
| **work tile** | `(m_block, head, batch, split)` — one unit handed out by the tile scheduler |
| **correction factor** | $e^{m_{\text{old}}-m_{\text{new}}}$ — rescales prior `O`/`ℓ` when the max rises |
| **LSE** | $m + \log \ell$ — log-denominator; for backward pass and SplitKV combine |

---

## What's next (not covered here)

These are features/optimizations layered on the fundamentals above, for a future deep-dive:

- **Scheduling**: static-persistent vs. CLC work-stealing vs. LPT/varlen tile schedulers; persistence.
- **2-CTA MMA**: two CTAs in a cluster cooperating on one MMA (`cta_group_size`, cluster mbarriers).
- **KV-side**: paged KV cache (`PagedKVManager`), varlen (`cu_seqlens`), `pack_gqa`.
- **Numerics & masking**: FP8 path (descales, `ex2_emu`, `max_offset`), causal/local masking and block-skipping, `score_mod`/`mask_mod`, block sparsity.
- **Micro-overlap**: `split_P_arrive` (start `P·V` on the first half of P early).

## References

- FlashAttention-4 source: `flash_attn/cute/flash_fwd_sm100.py`, `interface.py`, `flash_fwd_combine.py`
- [Appendix A: GPU Architecture Primer]({{< ref "appendix-a-gpu-architecture" >}})
- [Flash Attention v1 & v2]({{< ref "09-flash-attention-v1-v2" >}}) · [Flash Attention v3 & v4]({{< ref "10-flash-attention-v3-v4" >}})
- NVIDIA CUTLASS / CuTe DSL documentation; PTX ISA (`tcgen05`, `mbarrier`, `cp.async.bulk.tensor`)

*Last updated: June 2026*
