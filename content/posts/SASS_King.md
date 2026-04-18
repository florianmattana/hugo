---
title: "SASS King, Part 1: Reading NVIDIA SASS from First Principles"
date: 2026-04-17
draft: false
tags: ["sass", "gpu", "cuda", "reverse engineering", "sm120", "blackwell"]
categories: ["GPU Kernel Engineering"]
description: "Reverse engineering NVIDIA SASS across architectures. First milestone of the SASS King project: four minimal kernels, one universal method."
summary: "SASS is the machine code NVIDIA GPUs actually execute. PTX is documented, SASS is not. This is the first entry in a long form project to build a complete, architecture by architecture reference for NVIDIA SASS, starting from SM120 where I have hardware."
ShowToc: true
TocOpen: true
---

## What SASS King is

**SASS King is a systematic reverse engineering of NVIDIA SASS across architectures.**

SASS is the machine code that NVIDIA GPUs actually execute. CUDA compiles to PTX (a virtual, documented ISA). PTX is then assembled to SASS (architecture specific and undocumented). Everything interesting for performance happens at the SASS level: instruction scheduling, register allocation, scoreboard management, fusion decisions, unrolling strategies. None of this is visible at the source level, and almost none of it is visible even at PTX.

The last systematic public work on SASS was Jia et al. from Citadel, in 2018 to 2019, covering Volta and Turing. Nothing comparable exists for Ampere, Hopper, or Blackwell. For SM120 (Blackwell consumer, RTX 5070 Ti and RTX 5090) specifically: zero.

The target architectures for this project are:

| Architecture | Code | Representative GPU | Key features |
|---|---|---|---|
| Ampere | SM80 | A100 | baseline datacenter Ampere, cp.async, HMMA |
| Ada Lovelace | SM89 | RTX 4090 | consumer, most common inference card |
| Hopper | SM90a | H100 | TMA, WGMMA, warp specialization, mbarrier, clusters |
| Blackwell datacenter | SM100a | B200 | tcgen05.mma, TMEM |
| Blackwell consumer | SM120 | RTX 5070 Ti / 5090 | hybrid SM90/SM100 ISA, mma.sync with mxf8f6f4 |

The work starts on SM120 because that is where I have direct hardware access, but the methodology and tooling are universal. Community contributions and public binaries will be used for the other architectures.

This post is the first milestone: four minimal kernels that establish the vocabulary and the reading method.

## Why SASS matters for kernel developers

This is not academic. If you write CUDA kernels for performance, you eventually have to read SASS. A few concrete examples from real work.

**You are debugging a performance regression between two compiler versions.** NCU shows your kernel 15% slower. The high level source is unchanged. The only way to find the cause is to diff the SASS of the two builds and identify which instructions changed. Maybe ptxas decided to spill a register that it previously kept live. Maybe it changed a scheduling decision under the hood. You cannot see any of this from PTX alone.

**You think you wrote an FMA but you got a multiply and an add.** You wrote `dot += a[i] * b[i]`. You expected an FFMA. You look at the SASS and you see `FMUL` followed by `FADD`. ptxas failed to fuse because of an intermediate variable that introduced a rounding boundary, or because of strict IEEE compliance under `Xptxas O0`, or because the expression was in a slightly different shape than what the fuser recognizes. You only notice this in the SASS.

**Your register pressure exploded after a minor source change.** You added one more `__shared__` cache buffer, and suddenly the kernel runs 5x slower. NCU tells you about spilling but not what got spilled. The SASS shows `STL` and `LDL` instructions (store and load local memory) that were not there before. Now you can trace exactly which variable is being spilled and change the source to keep it in registers.

**Your tight loop has 200 instructions.** You wrote a 10 line loop. The kernel is slow. The SASS shows that ptxas unrolled the loop by 16 and generated a four way dispatch cascade for different trip counts (see kernel 04 below). Your code bloat is a factor of 20x, which pushed the hot loop out of the instruction cache. You only know this from the SASS.

**Your kernel stalls on a scoreboard that you did not create.** You see a `wait={SB5}` in the middle of your compute section, but you never issued a variable latency operation. It turns out ptxas spilled a register to local memory earlier in the kernel, and the load local (`LDL`) is what set SB5. Without reading the SASS, you would blame memory bandwidth.

In every one of these cases, the SASS tells you something the higher level tooling does not. Knowing how to read SASS is not optional for high performance kernel work, but the barrier to entry is steep because the documentation does not exist. This project is an attempt to lower that barrier.

## The tools

Two tools are used throughout this series.

**`cuobjdump dump sass`** (from the CUDA toolkit). The official disassembler. Produces raw SASS as text. Essential but not sufficient: it shows opcodes and operands, but leaves the scheduler control codes (stall, yield, scoreboard) as cryptic prefixes like `:::3::5:0`.

**[gpuasm.com](https://gpuasm.com/)**. A web based SASS viewer that parses cuobjdump output and presents it with:

* Scoreboard set (SBS) and wait (SBW) shown as colored pills and index numbers, not bitmask prefixes.
* Stall count and yield flag as their own columns.
* Register bank coloring so you can see bank conflicts at a glance.
* Register pressure annotations per line.
* Visual arrows showing producer consumer dependencies across the instruction stream.

Without gpuasm.com, reading a non trivial SASS dump is painful. With it, the scheduler's decisions become visible. Every screenshot in this post comes from gpuasm, and every claim about scoreboards or stalls is something you can verify there yourself. Upload a `.cubin` or paste a `cuobjdump` output and the tool annotates it for you.

Credit to the gpuasm authors for making this public. It is the single most useful tool for this line of work.

## Methodology: controlled variation

The method used throughout this series is **controlled variation**:

1. Start from the simplest possible kernel.
2. Make **one** minimal change to the source.
3. Recompile, dump SASS, diff against the previous version.
4. Observe exactly what changed and what stayed the same.
5. Document the observation.
6. Repeat.

This is complementary to top down analysis of real kernels. In a real kernel there are too many moving parts to attribute any single observation to a specific source feature. In a controlled variation sequence, every change is isolated, so every change in the SASS can be traced to its cause.

The first four kernels in this series:

| Kernel | Source change | What the kernel exposes |
|---|---|---|
| 01 | `c[i] = a[i] + b[i]` | Baseline infrastructure: prologue, bounds check, memory operations |
| 02 | `c[i] = a[i] + b[i] + 1.0f` | Addition does not fuse into FMA, register recycling |
| 03 | `d[i] = a[i] * b[i] + c[i]` | FFMA fusion, scoreboard grouping |
| 04 | runtime bounded `for` loop | Unrolling cascade, HFMA2 constant loading trick |

Each kernel has its own post with the full annotated analysis. This guide covers the synthesis.

## Anatomy of a SASS line

Every SM120 SASS instruction is 128 bits (16 bytes), fixed size since Volta. A typical line in gpuasm shows:

```
0x0050  IMAD R11, R11, UR4, R0    stall=5   yield   SBS=.   SBW={0}
```

Seven pieces of information:

**Address**: where the instruction lives in the kernel. Spaced by 0x10 because every instruction is 16 bytes.

**Opcode with modifiers**: the base operation plus suffixes that alter behavior. `IMAD.WIDE` produces a 64 bit result. `LDG.E.128` loads 128 bits. `FSETP.GE.AND` combines its comparison with an existing predicate.

**Destination and source registers**: per thread (R0 to R255), uniform (UR0 to UR63), predicate (P0 to P6), uniform predicate (UP0 to UP6). `RZ`, `URZ`, `PT`, `UPT` are hard wired zero and true.

**Stall count (1 to 15)**: how many cycles the scheduler waits before emitting the next instruction from this warp. Covers fixed latency pipelines.

**Yield flag**: a hint that the scheduler can switch warps here. Set on instructions that are likely to stall.

**Scoreboard set (SBS)**: for variable latency producers like `LDG`, this is the scoreboard the instruction occupies while in flight.

**Scoreboard wait (SBW)**: for consumers, this is the set of scoreboards the instruction must wait on before issuing.

Stall count and scoreboards are independent mechanisms. Stall count controls the **cadence** of instruction issue (how often this warp can send something to the pipeline). Scoreboards control the **correctness** of data dependencies when the latency is not known at compile time. Both must be satisfied for an instruction to be emitted.

## The two file register model

SM120 has two physically separate register files.

**Per thread registers** (R0 to R255). 32 copies per warp, one per thread. In the main register file SRAM. Used for values that vary between threads (thread index, loaded data, computed results).

**Uniform registers** (UR0 to UR63). One copy per warp. In a smaller dedicated SRAM with its own datapath. Used for values shared across all threads of the warp (block index, kernel arguments, broadcast constants).

Uniform registers were added on Turing (SM75) and expanded on SM100 and above (256 uniform registers instead of 64). A single instruction can mix sources from both files. `IMAD R11, R11, UR4, R0` reads UR4 once via the uniform datapath and broadcasts to all 32 threads, while R11 and R0 are read per thread. The destination is per thread because at least one per thread source is involved.

ptxas performs **uniformity analysis automatically**. Values detected as uniform go into UR without programmer involvement. This is invisible at the source level but visible in the SASS.

The benefit is real: a value stored in UR is stored once per warp instead of 32 times in the main register file. This reduces register pressure, reduces register file bandwidth, and reduces energy per operation.

## Special registers

Some values are not held in R or UR but in read only hardware registers called special registers (SR). They hold thread identity, block identity, clocks, and various configuration values. A kernel accesses them via `S2R` (read into a per thread R) or `S2UR` (read into a uniform UR).

The name can appear either symbolically (`SR_TID.X`) or by numeric index (`SR33`), depending on the disassembler. Per thread values (like threadIdx) vary across the warp. Uniform values (like blockIdx, blockDim, gridDim) are the same for all threads of the warp.

The most commonly seen special registers on SM120:

| Index | Symbolic name | Content |
|---|---|---|
| SR0 | SR_LANEID | Lane ID within the warp (0 to 31) |
| SR1 | SR_CLOCK | Cycle counter (lower 32 bits) |
| SR33 | SR_TID.X | threadIdx.x |
| SR34 | SR_TID.Y | threadIdx.y |
| SR35 | SR_TID.Z | threadIdx.z |
| SR37 | SR_CTAID.X | blockIdx.x |
| SR38 | SR_CTAID.Y | blockIdx.y |
| SR39 | SR_CTAID.Z | blockIdx.z |
| SR40 | SR_NTID.X | blockDim.x |
| SR41 | SR_NTID.Y | blockDim.y |
| SR42 | SR_NTID.Z | blockDim.z |
| SR44 | SR_NCTAID.X | gridDim.x |
| SR45 | SR_NCTAID.Y | gridDim.y |
| SR46 | SR_NCTAID.Z | gridDim.z |
| SR56 | SR_EQMASK | Equality mask (active threads with same value) |
| SR57 | SR_LTMASK | Less than mask |
| SR58 | SR_LEMASK | Less or equal mask |
| SR59 | SR_GTMASK | Greater than mask |
| SR60 | SR_GEMASK | Greater or equal mask |
| SR80 | SR_CLOCKLO | Clock low (32 bit cycle counter) |
| SR81 | SR_CLOCKHI | Clock high |
| SR82 | SR_GLOBALTIMERLO | Global timer low (nanoseconds) |
| SR83 | SR_GLOBALTIMERHI | Global timer high |

Less commonly seen special registers include performance counters (SR8 to SR15, SR79 to SR86), shared and local memory window configuration (SR48 to SR55), and error status registers (SR64, SR66, SR67). Some indices are partially documented or undocumented and may differ slightly between architectures.

Special registers have variable latency. S2R and S2UR always emit a scoreboard because ptxas cannot know the exact read latency at compile time. For thread and block identity reads (SR_TID, SR_CTAID, SR_NTID) the observed latency on SM120 is roughly 20 to 50 cycles (hypothesis, to be microbenchmarked). Clock and timer reads are slower.

## The six section kernel skeleton

Every CUDA kernel begins with the same skeleton, regardless of complexity. Once recognized it can be skipped on a first read.

1. **Stack pointer init**. `LDC R1, c[0x0][0x37c]`. Loads the local memory base pointer. Present even in kernels that do not spill.
2. **Thread and block identity**. `S2R Rn, SR_TID.X` and `S2UR URn, SR_CTAID.X`. Reads the per thread ID and the block ID.
3. **Index computation**. `IMAD Rn, blockDim, blockIdx, TID`. Produces the global index `i` in one instruction.
4. **Bounds check**. `ISETP.GE.AND P0, PT, Ri, Rn, PT` followed by `@P0 EXIT`. Out of bounds threads exit immediately, without divergence, via predication rather than branching.
5. **Pointer loads**. `LDC.64` and `LDCU.64` from constant memory. Every kernel argument that is a pointer or a scalar is pulled from a fixed offset in constant bank 0.
6. **Global descriptor load**. `LDCU.64 UR*, c[0x0][0x358]`. On SM90 and later (inherited by SM120), every global memory access uses a descriptor from a fixed constant memory offset. The descriptor carries cache hints and permissions.

After these sections, the body of the kernel begins. This is where the actual algorithm lives.

## Scoreboards in practice

The six scoreboards (SB0 to SB5) per warp are a shared budget that ptxas manages across the whole kernel. Key observations from the first four kernels.

**Independent producers get distinct scoreboards.** When multiple `LDC` or `LDCU` load different pointers that will be used at different times, ptxas assigns each to its own SB. This allows downstream operations to proceed as soon as any one pointer is ready.

**Co consumed producers share a scoreboard.** When multiple `LDG`s all feed the same FFMA (as in kernel 03 with three LDGs on SB4), they share a single SB. The consumer does one wait, which covers all of them. This saves SB slots for other purposes.

**Fixed latency operations do not use scoreboards.** IMAD, FFMA, ISETP, and similar arithmetic instructions have known latency. ptxas encodes the latency directly in the stall count and skips the scoreboard. SBs are reserved for unpredictable latencies.

**Yield flag accompanies scoreboard waits.** Every instruction observed with a scoreboard wait also has the yield flag set. The compiler is telling the scheduler that this is a profitable moment to switch warps if another is ready.

**Cross pipeline predicate transfer has unusual latency.** A predicate produced by `ISETP` and consumed by `@P EXIT` or `BRA` incurs a stall count of 9 to 13 cycles on SM120, much higher than the 5 cycles typical for ALU arithmetic. Hypothesized cause: cross pipeline transfer from ALU to CBU (control branch unit). This will be microbenchmarked in a later installment.

## What these four kernels teach a kernel developer

This is the practical payoff section. Reading the SASS of simple kernels builds the intuition needed to diagnose real performance problems. Here is what each of the four kernels teaches, with actionable insights.

### Kernel 01 (baseline): measure the useful compute ratio first

Twenty instructions total, one of which (`FADD`) does useful work. The remaining nineteen are address arithmetic, memory operations, constant loads, and bookkeeping. That is a 5% useful compute ratio, the signature of a deeply memory bound kernel.

**Insight**: before optimizing, look at the ratio. If your kernel spends 95% of its instructions moving data and calculating addresses, the bottleneck is almost certainly memory, not compute. No amount of FMA fusion or instruction scheduling will help. You either need to pack more compute per memory access (raise arithmetic intensity), or accept that the kernel is bandwidth bound and focus on memory access patterns.

**How to measure it**: Count the arithmetic instructions (FFMA, FADD, FMUL, IMAD for non address work) and divide by total instructions in the body (after the prologue). If under 20%, you are memory bound. If over 50%, compute bound. In between, latency bound.

**Practical check**: In your real kernel, find the loop body in the SASS. Count the FFMAs versus everything else. Compare to what you would need for the arithmetic intensity to cross the ridge point of your GPU. If you cannot reach the ridge point with this ratio, you need a different algorithm.

### Kernel 02 (no fusion): FMA fusion is syntactic, not semantic

`a + b + c` does not fuse. `a*b + c` does. The compiler does direct pattern matching on the expression tree, not algebraic reasoning. If your source expression does not literally contain a multiply feeding an add, you get two instructions.

**Insight**: when accuracy permits and performance matters, structure your arithmetic as explicit multiply adds. The canonical example is Horner's method for polynomial evaluation. Instead of `a0 + a1*x + a2*x*x + a3*x*x*x`, write `a0 + x*(a1 + x*(a2 + x*a3))`. The second form compiles to a chain of three FFMAs. The first compiles to something much more complicated with more rounding steps.

**Practical check**: Look at the SASS of any polynomial evaluation, dot product, or FIR filter in your code. Count the FFMAs. Compare to the number of multiply add operations in your source. If there is a gap, restructure the expression. Common culprits: intermediate variables that force the compiler to materialize the multiply result separately, explicit casts that insert a rounding step, or pure add chains where a multiply by one would unlock fusion.

**Bonus observation**: ptxas will recycle dead registers aggressively. In kernel 02, the register that held `threadIdx.x` at the start became a scratch register for the intermediate sum by instruction 0x110. Adding operations to your source does not linearly increase register pressure. The allocator finds dead slots. This is good news for accumulator loops.

### Kernel 03 (FFMA with three operands): the scoreboard budget is real

When I added a third input array, I added one LDC (to load the new pointer), one IMAD.WIDE (to compute the address), and one LDG (to read the value). That is three more instructions for one more operand.

**Insight**: the plumbing cost of each additional memory operand is approximately 3 instructions. If you are considering whether to pass an additional small array or to pack it into an existing structure, the SASS cost tells you. Three instructions times N threads times the number of iterations is a non trivial overhead in a tight loop.

The more subtle lesson is about scoreboard grouping. The three LDGs in kernel 03 all share SB4, because they are all consumed by the same FFMA. This is optimal: one wait, one latency hit. If ptxas had assigned them to three separate scoreboards, the consumer would still have to wait for all three, but the SB budget would be exhausted and other variable latency operations would have to queue up.

**Practical check**: In a kernel that loads many independent data streams (attention key and value pairs, GEMM tile loads), look at the SBS column in gpuasm. Are all your LDGs on one or two scoreboards, or on five or six? If you are using five or six, you are close to the budget. Any additional variable latency operation may serialize.

**Diagnostic**: if NCU reports "stall long scoreboard" in the profile, the warp is blocked on one of these SBs. The SASS tells you which load produced that SB, so you can trace the stall back to a specific memory access.

### Kernel 04 (runtime loop): the compiler generates a lot more code than you wrote

This kernel was supposed to have a single backward branch. Instead ptxas generated four execution paths (fully unrolled by 16, partially unrolled by 8, by 4, and a scalar tail) and dispatched between them at runtime based on the trip count. The final SASS is approximately 80 instructions for what looked like a 5 line loop.

**Insight 1: code size can be a problem.** The SM has an instruction cache per partition (around 8 KB on SM120, hypothesis). A kernel with a very large body may push past this limit and thrash. The symptom is mysterious throughput loss that does not correlate with anything in the metrics except instruction fetches. If your SASS is 10x larger than you expect, consider `#pragma unroll 1` or otherwise restricting the compiler's aggressive unrolling.

**Insight 2: the HFMA2 constant loading trick.** Kernel 04 exposed that ptxas uses the half precision FMA (`HFMA2`) to load FP32 constants by exploiting bit pattern concatenation:

```
HFMA2 R2, RZ, RZ, 1.875, 0.00931549072265625   // loads 0x3F8020C5 = 1.001f
```

The two FP16 immediates `{1.875, 0.00931549}` happen to have the combined 32 bit bit pattern of `1.001f`. ptxas prefers this over a plain `MOV R2, 0x3F8020C5` when the ALU pipeline is busier than the FMA pipeline. The choice is a scheduling heuristic.

**Practical consequence**: the compiler is actively balancing instructions across pipelines. If you are hand tuning a hot loop, be aware that adding an instruction on one pipeline can displace a constant load onto another. Tools like NCU's "Stalled Pipe Busy" metric can tell you which pipeline is saturated.

**Insight 3: watch for register spilling.** The first four kernels did not spill, so `STL` and `LDL` did not appear. But in real kernels this is the first thing to look for. If you see these instructions inside your hot loop, you have a spill. Options:

* Reduce the number of live values by restructuring the algorithm.
* Use `__launch_bounds__(blockSize, minBlocks)` to give ptxas more register budget.
* Use `maxrregcount` (ptxas flag) to force a specific register count. This may spill but may also improve occupancy.

**Insight 4: watch for division and transcendentals.** You did not see `CALL.REL.NOINC` in kernels 01 to 04 because there were no divisions or math library calls. If you see `CALL` in a kernel that should be straight line arithmetic, you have inadvertently triggered a slowpath. Division of variable integers is the most common culprit. Use `__fdividef` (approximate division) or shift and mask when possible.

### The general diagnostic workflow

Putting it together, here is the sequence to use when opening a SASS dump for performance work:

1. **Skip the prologue**. Pattern match the six section skeleton. Move past it.
2. **Find the hot region**. Look for the loop body (backward BRA), or the main compute block.
3. **Count the useful instructions**. Compute the ratio of arithmetic (FFMA, FADD, IMMA, QMMA) to total instructions in the hot region.
4. **Check for spills**. Grep for `STL` and `LDL`. If present in the hot region, address register pressure first.
5. **Check for slow calls**. Grep for `CALL`. If present, identify the source expression and replace it with a hardware supported alternative.
6. **Inspect the memory operations**. Are LDGs grouped on shared scoreboards, or fragmented? Are they vectorized (`.128`, `.64`) or scalar?
7. **Inspect the scheduling**. Look at gpuasm's pressure and stall annotations. Where are the wait points? Are they unavoidable (long data dependencies) or artificial (poor interleaving)?

Every time you investigate a kernel this way, your intuition for the next one improves. The goal of SASS King is to codify this intuition into a shared reference so that the ramp up time for new kernel developers is measured in weeks instead of years.

## The reading strategy as a checklist

Condensed form of the above:

* Skip the prologue (6 section skeleton).
* Find the compute section and count useful arithmetic instructions.
* Trace memory operations to their scoreboards and their consumers.
* Check for backward branches (loops), `@P EXIT` (early exits), BSSY and BSYNC (divergent reconvergence).
* Inspect stall counts and yield flags for scheduling decisions.
* Look for anomalies: `STL` and `LDL` (spills), `CALL` (function calls), oversized kernels (unrolling blowup), unusual stall counts (cross pipeline transfers).

## What is not yet covered

The first four kernels establish the basic vocabulary. Many important patterns have not yet been observed because no kernel has needed them:

* Shared memory operations (`LDS`, `STS`, `BAR.SYNC`)
* Warp level primitives (`SHFL`, `VOTE`, `WARPSYNC`)
* Vectorized memory access (`LDG.E.128`)
* Asynchronous memory (`LDGSTS`, `DEPBAR`, `cp.async`)
* Tensor cores (`QMMA` on SM120, `HMMA` and `IMMA` elsewhere)
* Atomics (`ATOMG`, `ATOMS`, `RED`)
* Math functions via `MUFU` (transcendentals)
* Register spilling (`STL`, `LDL`) observed in context
* Cluster and TMA (SM90 and above, SM120 inherits some of this)
* Divergent control flow (`BSSY`, `BSYNC` with real divergence)
* `WGMMA` (SM90), `tcgen05.mma` with TMEM (SM100)

Each will be introduced by a minimal source change in a subsequent kernel, on SM120 initially, and then compared against the same kernel on other architectures as the project expands.

## Roadmap

Near term kernels:

1. Loop with small fixed trip count. Observe full unrolling versus the cascade from kernel 04.
2. Shared memory scalar. Introduce LDS, STS, BAR.SYNC.
3. Vectorized load. `float4*` source to trigger `LDG.E.128`.
4. Warp level reduction. `__shfl_xor_sync` to trigger `SHFL.BFLY`.
5. Division. Observe the slowpath and the `CALL` instruction.
6. Forced register spill. Generate STL and LDL by exceeding the register budget.
7. First tensor core kernel. Smallest possible `QMMA` or `HMMA` call.

Medium term: the same controlled variations, reproduced on SM80, SM89, SM90a, SM100a. Comparing the same source across architectures exposes the ISA differences directly.

Long term: a complete instruction level reference for each architecture, and a corpus of annotated real kernel audits (FlashAttention, Marlin, CUTLASS mainloop, FP4 fused attention).

## Sources and acknowledgments

* Jia et al., Citadel (2018 to 2019). *Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking*. The methodological ancestor.
* Jarmusch et al. (2024 to 2025). Hopper and Blackwell microbenchmarking. Reference data for SM100.
* SemiAnalysis (2026). *Dissecting Nvidia Blackwell: Tensor Cores, PTX Instructions, SASS, Floorsweep, Yield*. Practical performance ceiling measurements.
* Scott Gray, maxas (2014). Proof that SASS level understanding yields a meaningful performance advantage.
* [0xD0GF00D/DocumentSASS](https://github.com/0xD0GF00D/DocumentSASS). Community maintained reverse engineered SASS reference.
* NVIDIA cuda binary utilities documentation, Blackwell ISA table 8.
* [gpuasm.com](https://gpuasm.com/). The tool that makes this work possible.

Source code for all kernels lives at [github.com/florianmattana/sass-king](https://github.com/florianmattana/sass-king). Per kernel detailed analyses are available as separate posts in this series.
