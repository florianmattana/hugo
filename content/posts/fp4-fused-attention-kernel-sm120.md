---
title: "Building an FP4 Fused Attention Kernel on Consumer Blackwell (SM120) (WIP)"
date: 2026-03-17
draft: false
tags: ["cuda", "blackwell", "sm120", "fp4", "attention", "tensor-cores"]
description: "A deep dive into writing a fused FP4 attention kernel for the RTX 5070 Ti, using inline PTX and warp-level MMA instructions."
showToc: true
TocOpen: true
---

## 1. Why FP4 Fused Attention on Consumer Blackwell?
 
The attention mechanism in transformers scales quadratically with sequence length. On a consumer GPU with 12 GB of VRAM and 672 GB/s of memory bandwidth, that becomes a hard wall very quickly. The interesting thing about the RTX 5070 Ti (SM120, 46 SMs) is the raw throughput the Tensor Cores can deliver:
 
| Precision | Throughput |
|---|---|
| FP16 | 123.5 TFLOPS |
| INT8 | 246.9 TFLOPS |
| FP4 | ~474 TFLOPS |
 
That is roughly a 4x advantage going from FP16 to FP4, and since FP4 values are four times smaller, you also move four times less data through memory. On paper, that is a massive win for attention. If you can actually use the FP4 Tensor Cores.
 
The ecosystem support for FP4 on consumer Blackwell is recent and still thin. SageAttention3 does support SM120 and achieves over 1000 TOPS on the RTX 5090. But it is built on CUTLASS templates, which makes it very difficult to understand what happens between the data load and the Tensor Core instruction. If you want to know exactly how bytes are packed into MMA registers, how scale factors are distributed across lanes, or why a specific shared memory layout was chosen, the CUTLASS abstraction does not help you. The same is true for the emerging FlashInfer and vLLM backends that are adding SM120 paths.
 
There are also non-fused FP4 kernels for this hardware. For example [VincentKaufmann fp4-cuda-kernel](https://github.com/VincentKaufwormann/fp4-cuda-kernel) reaches about 143 TFLOPS. But non-fused means you compute QxK, write the full NxN score matrix to VRAM, read it back, apply softmax, write again, then compute the attention output. For 4096 tokens, that score matrix alone is 64 MB. On a 12 GB card, that is a dealbreaker.
 
The whole point of a *fused* kernel is to keep the intermediate score matrix in registers and never write it to global memory. That is what FlashAttention does for FP16 and I wanted to do the same thing for FP4.
 
This article documents the full process of building that kernel from scratch using inline PTX assembly on the RTX 5070 Ti. The goal is not to compete with SageAttention3 on throughput. It is to make every step of the FP4 fused attention pipeline visible and understandable: the MMA instruction, the fragment layout, the quantization, the scale factors, the softmax, the profiling. Most of this is undocumented for SM120 and had to be figured out empirically.

---

## 2. Choosing the Programming Model

I considered three approaches:

**Option A -- Inline PTX.** Write the kernel in CUDA C++ and embed the Tensor Core MMA instructions as inline PTX assembly. This gives full control over register allocation, meaning I can guarantee the score matrix stays in registers.

**Option B -- CuTe (CUTLASS 3.x).** Use NVIDIA template library. CuTe is powerful, but it abstracts away register placement. I was not confident I could prevent it from spilling the score matrix to shared or global memory, especially for a non-standard fused pattern.

**Option C -- Patch an existing INT8 kernel.** Take a working fused INT8 attention kernel and swap the MMA instructions for FP4 equivalents. Faster to prototype, but brittle, the register layouts differ between INT8 and FP4 MMA, so the whole data flow would need reworking anyway.

I went with **Option A**. The trade-off is clear: more manual work, more room for bugs, but absolute certainty about where every value lives. For a fused kernel where the entire point is keeping data in registers, that certainty is worth it.

---

## 3. What the Kernel Needs to Do

Before writing a single line of code, it helps to map out the full problem.

A fused FP4 attention kernel has to solve five things in sequence, and each one has a hardware dependency that constrains everything that follows.

**Load the input tiles.** Q, K, and V are too large to fit in registers. They live in global memory and must be loaded into shared memory tile by tile. The size of each tile is bounded by the shared memory budget per SM, which on SM120 turns out to be 99 KiB and not the 128 KiB I initially assumed.

**Quantize on the fly.** The Tensor Core does not consume float32. Before running the matrix multiply, each tile must be converted to FP4 E2M1 and its block scale factors must be computed. This has to happen in shared memory, which means a two-pass approach: load the tile as float32 first, compute the scale, then encode.

**Run the matrix multiply in FP4.** The core operation is S = Q times K-transpose, computed with FP4 Tensor Cores. The score matrix S is 64 times 64 floats and must never be written to global memory.It lives entirely in registers across the warp throughout the computation. A warp is a group of 32 threads that execute together on the GPU, the fundamental unit of execution on Tensor Cores.

**Apply online softmax.** Softmax over a full row of S requires knowing the row maximum. But in the MMA output layout, each row is distributed across four threads. That forces a cross-thread reduction before every softmax step, using warp shuffle instructions.

**Accumulate the output.** The final output O is computed as softmax(S) times V. This second matrix multiply accumulates incrementally as each column tile of S is processed, again never materializing the full score matrix.

Each of these steps depends on knowing exactly which MMA instruction is available on SM120, what register layout it expects, and what quantization format it accepts.

That is what the next section is about.

---

## 4. Picking the Right MMA Instruction

This is where I hit the first major wall. I started by reading the PTX ISA docs looking for FP4 MMA instructions on Blackwell. The datacenter SM100 chips use `tcgen05.mma`, a new-generation instruction that operates on large tiles and uses a dedicated hardware unit called Tensor Memory. I assumed SM120 would have something similar.

It does not.

After digging through [CUTLASS issue #2800](https://github.com/NVIDIA/cutlass/issues/2800), a [thread on the NVIDIA developer forums](https://forums.developer.nvidia.com/), and [CUTLASS issue #3044](https://github.com/NVIDIA/cutlass/issues/3044), I pieced together the reality: SM120 uses the older Ampere-style **warp-level `mma.sync`** instructions. No Tensor Memory, no `tcgen05`. The specific instruction I need is:

`mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e2m1.f32.ue8m0`

Let me unpack that:

- `mma.sync.aligned` -- warp-synchronous, all 32 threads participate.
- `kind::mxf8f6f4` -- the MX (microscaling) family that covers FP4/FP6/FP8.
- `block_scale.scale_vec::1X` -- each group of 32 FP4 values shares one 8-bit scale factor. I initially tried `scale_vec::2X` (one scale per 16 values, finer granularity) but it does not compile on SM120. Only 1X is supported, which means 6.25% overhead for the scale factors.
- `m16n8k32` -- tile shape: 16 rows x 8 columns, with K=32 (32 FP4 values along the reduction dimension per instruction).
- `f32.e2m1.e2m1.f32` -- FP32 accumulators, FP4 E2M1 inputs for both A and B matrices.
- `ue8m0` -- the scale factor format (unsigned 8-bit exponent, no mantissa i.e., powers of two only).

The register budget for one MMA call is roughly 7 registers per thread: 2 for the A fragment, 1 for B, and 4 for the FP32 accumulator. This assumption turned out to be wrong. The correct count is 10: 4 registers for A, 2 for B, and 4 for the accumulator. 
Discovering that cost several weeks of debugging, and section 11 explains how.

The fused kernel will process tiles of Q, K, V through shared memory, roughly 9 KB by my initial estimate. That number turned out to be wrong in two ways.

First, the actual shared memory usage ended up closer to 49 KiB once the full quantization pipeline was in place. Second, the available budget on SM120 is not 128 KB as I'd assumed from the general Blackwell documentation, but 99 KiB. I found this while browsing open CUTLASS issues: [issue #3144](https://github.com/NVIDIA/cutlass/issues/3144), titled "StageCountAutoCarveout assumes max family SMEM, breaks SM121 (99 KiB vs SM120 228 KiB)", reporteda bug where CUTLASS was incorrectly assuming all SM12x GPUs share the same shared memory size. A contributor clarified in the thread that SM120 consumer Blackwell has 99 KiB, while SM100 datacenter Blackwell has 228 KiB. The distinction matters: at 49 KiB, the kernel fits within the 99 KiB optin budget, but only just. 

The distinction matters: at 49 KiB, the kernel fits within the 99 KiB optin budget but only just. On any new GPU, checking the actual shared memory limit with `nvidia-smi -q | grep "Max Shared Memory"` before making assumptions about the budget would have saved me time. Section 8 covers the shared memory layout in detail.

Section 7 covers the shared memory layout in detail.

---

## 5. Testing the MMA Instruction (and Everything That Went Wrong)

Before building the full fused kernel, I needed to verify that a single FP4 MMA instruction actually works. The idea is simple: load known values into registers A and B, run the MMA, and check that the FP32 accumulators contain the expected result.

I wrote a minimal warp-synchronous kernel, launched with `<<<1, 32>>>` so that exactly one warp executes. The kernel fills the A and B registers with constant FP4 values, calls the MMA via inline PTX, and prints the four accumulator floats from thread 0.

### The tile shape surprise

My first attempt used `m16n8k64` and I reasoned that since FP4 values are 4 bits each, 64 of them would fit in 32 bytes (the same as 32 FP8 values). The PTX assembler disagreed. It turns out the correct shape for FP4 on SM120 is **m16n8k32**: the k-dimension counts 8-bit *containers*, not individual FP4 values. Each container holds one FP4 nibble in bits 5-2, padded with zeros. This means you are effectively wasting half the container, but that is what the hardware expects.

### The encoding bug that cost me a full day

FP4 E2M1 encodes the value 1.0 as the 4-bit pattern `0b1000`. The container is an 8-bit byte, and the nibble must sit in bits 5-2, not bits 3-0. That means the correct byte for 1.0 is `0x08`: the pattern `0b00001000`, with the nibble in the upper half of the low byte. If you place the nibble in bits 3-0 instead, you get `0x02`, which the hardware reads as a completely different value.

I initially filled every register with `0x22222222`, four bytes of `0x22` packed together. I thought I was encoding 2.0 in every position. What I was actually doing was placing the nibble in the wrong bit positions. The hardware read each byte as `0b00100010`, extracted the nibble from bits 5-2, which gives `0b1000` — the encoding for 1.0, not 2.0. So the MMA computed 32 multiplications of 1.0 times 1.0 and returned 32.0. I was expecting 128.0 (32 times 2.0 times 2.0 with scale 1.0).

After staring at bit layouts for longer than I would like to admit, I realized the nibble was in the wrong position. Switching to `0x08080808`, which places the 1.0 nibble correctly in bits 5-2 of each byte, and setting scale to 1.0, the MMA returned 32.0 exactly. That is 32 multiply-accumulates of 1.0 times 1.0. Correct.

The lesson: the FP4 container format is `00_SEMM_00` where the nibble occupies bits 5 through 2. Get the shift wrong and the hardware silently reads a different value with no error.

### The inline PTX

There is a reason this instruction appears as raw inline assembly rather than a clean C++ wrapper. The CUDA Core Compute Libraries (CCCL) expose `cuda::ptx` wrappers for many PTX instructions, which would normally be the right abstraction to use here. But at the time of writing, `cuda::ptx` does not provide wrappers for warp-level `mma.sync` on SM120. I exchanged with Federico Busato, who maintains CCCL at NVIDIA, on this exact gap. His read was that the wrappers would be useful but the decision was pending. I opened [CCCL issue #8146](https://github.com/NVIDIA/cccl/issues/8146) to track it. In the meantime, inline PTX is the only path.
Here is the `asm volatile` block as I first wrote it:

```cpp
asm volatile(
    "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4"
    ".block_scale.scale_vec::1X.f32.e2m1.e2m1.f32.ue8m0"
    " {%0, %1, %2, %3},"
    " {%4, %5},"
    " {%6},"
    " {%7, %8, %9, %10},"
    " %11,"
    " %12;"
    : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
    : "r"(a0), "r"(a1),
      "r"(b0),
      "f"(c0), "f"(c1), "f"(c2), "f"(c3),
      "r"(scale_a), "r"(scale_b)
);
```

This version has two registers for A and one for B. It is wrong. The correct instruction requires four A registers and two B registers. This assumption cost several weeks of debugging and is corrected in section 11. The block above is shown as written at this stage because it compiled and passed the isolated MMA test described below. The test was not thorough enough to catch the error.

The `"=f"` constraints are FP32 output registers, `"r"` are 32-bit integer input registers. The accumulator C is passed through as input (initialized to zero for the first call), and the result lands in D. The scale registers each pack four UE8M0 bytes into a single `uint32`.

With `a0 = a1 = 0x08080808` (all 1.0), `b0 = 0x08080808` (all 1.0), and scales set to 1.0 (`0x7F7F7F7F`, since UE8M0 byte 127 = 2^0 = 1.0), the result was 32.0 in every accumulator lane. That is 32 multiply-accumulates of 1.0 x 1.0, which is exactly right.

---

## 6. Encoding FP32 to FP4 E2M1

Once the MMA worked with hardcoded constants, the next step was encoding arbitrary `float` values into FP4 E2M1 at runtime. The encoding function is covered in detail in a dedicated post on my [MXFP4 quantization kernel](https://florianmattana.com/posts/mxfp4_article/). 
What follows here is a summary of the key points as they apply to this kernel. If you are already familiar with my previous article or have already experience with FP4 quantization, you can skip this section.

### The FP4 E2M1 format

FP4 has 1 sign bit, 2 exponent bits (bias 1), and 1 mantissa bit. That gives you exactly 16 representable values:

| Binary | Value |
|--------|-------|
| `0000` | +0.0 |
| `0001` | +0.5 |
| `0010` | +1.0 |
| `0011` | +1.5 |
| `0100` | +2.0 |
| `0101` | +3.0 |
| `0110` | +4.0 |
| `0111` | +6.0 |
| `1000` | -0.0 |
| `1001` | -0.5 |
| `1010` | -1.0 |
| `1011` | -1.5 |
| `1100` | -2.0 |
| `1101` | -3.0 |
| `1110` | -4.0 |
| `1111` | -6.0 |

The maximum representable magnitude is 6.0. Anything larger saturates.

### The encoding function

The device function takes a `float`, determines the closest FP4 magnitude through a chain of comparisons, assembles the 4-bit nibble, and shifts it left by 2 to place it in bits 5-2 of the 8-bit container:

```cuda
__device__ uint8_t encode_fp4_e2m1(float val) {
    uint8_t sign = (val < 0.0f) ? 1 : 0;
    float abs_val = fabsf(val);

    uint8_t encoded;
    if      (abs_val >= 5.0f) encoded = 0x07; // 6.0
    else if (abs_val >= 3.5f) encoded = 0x06; // 4.0
    else if (abs_val >= 2.5f) encoded = 0x05; // 3.0
    else if (abs_val >= 1.75f) encoded = 0x04; // 2.0
    else if (abs_val >= 1.25f) encoded = 0x03; // 1.5
    else if (abs_val >= 0.75f) encoded = 0x02; // 1.0
    else if (abs_val >= 0.25f) encoded = 0x01; // 0.5
    else                       encoded = 0x00; // 0.0

    uint8_t nibble = (sign << 3) | encoded;
    return nibble << 2; // place in bits 5-2
}
```

Quick sanity checks: `encode_fp4_e2m1(1.0f)` returns `0x08`, `encode_fp4_e2m1(-1.0f)` returns `0x28`, `encode_fp4_e2m1(6.0f)` returns `0x1C`. To pack four encoded bytes into a `uint32_t` for the MMA register:

```cuda
uint32_t pack = e0 | (e1 << 8) | (e2 << 16) | (e3 << 24);
```

End-to-end test: encode 1.0 into every position, pack, run MMA -> 32.0. Encode 2.0 -> 128.0. Both matched. The encoding function was correct.

## 7. Block Scaling: Why the Encoding Function Is Not Enough

Here is the problem I ran into immediately: FP4 E2M1 maxes out at 6.0. If your input values are larger and in attention, they absolutely will be, everything above 5.0 clamps to 6.0 and you lose all relative differences. For example, `encode_fp4_e2m1(12.0f)` and `encode_fp4_e2m1(10.0f)` both return `0x1C` (6.0). That is catastrophic for attention scores where the relative ordering is everything.

### The solution: block scaling

The MX format handles this with a shared scale factor per block. Before encoding, you divide every value in a block by a common scale factor, bringing them into the representable FP4 range. The MMA hardware then multiplies the scale back in during the accumulation for free, no extra instructions.

Take a block of values: {12.0, 10.0, 3.0, -7.0}. The maximum absolute value is 12.0. If I choose a scale of 16, dividing gives {0.75, 0.625, 0.1875, -0.4375}. Now every value fits in FP4 range and critically, they encode to *different* FP4 values, preserving the relative ordering.

### Why the scale must be a power of two

The scale factor is stored in UE8M0 format: an 8-bit unsigned exponent with no mantissa. The actual scale value is 2^(byte - 127). This means only powers of two are representable. That is not a limitation it is a feature. Multiplying by a power of two is just a bit shift in the exponent of a floating-point number, so the Tensor Core applies the scale with zero additional cost.

### Choosing the right scale

You want the smallest power of two that is greater than or equal to the maximum absolute value in the block. Rounding up avoids overflow (values that would exceed 6.0 after division). Rounding down would risk saturation for the largest values, exactly the ones you most need to preserve.

The formula: find `max_abs` across the block, compute `exponent = ceil(log2(max_abs))`, and the UE8M0 byte is `exponent + 127`.

Example: block maximum is 12.0. `ceil(log2(12.0))` = `ceil(3.58)` = 4. Scale = 2^4 = 16. UE8M0 byte = 4 + 127 = 131 = `0x83`.

### The device function

```cuda
__device__ uint8_t compute_scale_ue8m0(float* block, int size) {
    float max_abs = 0.0f;
    for (int i = 0; i < size; i++) {
        float a = fabsf(block[i]);
        if (a > max_abs) max_abs = a;
    }
    if (max_abs == 0.0f) return 127; // scale = 1.0

    int exponent = (int)ceilf(log2f(max_abs));
    return (uint8_t)(exponent + 127);
}
```

### Validation

I set up a test: A = 8.0 everywhere, B = 1.0 everywhere. Scale for A: `max_abs` = 8.0, `ceil(log2(8))` = 3, UE8M0 byte = 130 = `0x82` (scale = 8). Scale for B: `max_abs` = 1.0, `ceil(log2(1))` = 0, UE8M0 byte = 127 = `0x7F` (scale = 1). After dividing A by 8, every element becomes 1.0, encoded as `0x08`. The MMA computes 32 x (1.0 x 1.0) = 32.0, then the hardware applies the scales: 32.0 x 8 x 1 = 256.0.

The kernel printed 256.0. The block scaling pipeline works end to end.

### The trade-off

On SM120, each scale factor covers a block of 32 elements along the K dimension. If one element is an outlier, say 100.0 in a block where everything else is around 1.0 the scale gets set to 128 and all the small values round to zero after division. Smaller blocks would preserve more detail, but `scale_vec::1X` is all we get on this hardware. It is a real limitation, and for attention (where softmax creates sharp distributions), it matters. I will revisit this when profiling the full kernel.

---

The kernel now has three validated building blocks: `encode_fp4_e2m1`,`compute_scale_ue8m0`, and the inline PTX MMA call. The next step is loading Q, K, V tiles into shared memory and wiring everything together into the fused attention loop.

---

## 8. Assembling the Kernel: Where Things Got Real

The three building blocks worked in isolation. Encoding, scaling, MMA, all validated with hardcoded test values. Now I had to wire them together into an actual kernel that loads real data from VRAM, quantizes it on the fly, and runs the Tensor Core multiply. This is where the gap between "I have working pieces" and "I have a working kernel" became very real.

### The first decision: how much shared memory

My first attempt allocated two separate FP32 buffers in shared memory, one for Q and one for K. Each tile is 64 tokens times 128 dimensions = 8192 floats = 32 KB. Two of them: 64 KB. Add the quantized buffers and scales, and I was over 80 KB, which exceeds the 99 KiB optin budget established in section 3.

The fix was simple: Q and K are never needed in FP32 at the same time. Load Q as FP32, quantize it into `Q_quant`, then reuse the same staging buffer for K. One FP32 buffer instead of two. That brought the total down to about 49 KiB.

| Buffer | Type | Size | Purpose |
| --- | --- | --- | --- |
| `staging` | float | 32,768 B | Reusable FP32 buffer for loading from VRAM |
| `Q_quant` | uint8 | 8,192 B | Q tile after FP4 quantization |
| `K_quant` | uint8 | 8,192 B | K tile after FP4 quantization |
| `Q_scales` | uint8 | 256 B | One UE8M0 scale per 32 Q elements |
| `K_scales` | uint8 | 256 B | One UE8M0 scale per 32 K elements |

Still only one block per SM due to register pressure from the accumulators but the shared memory budget is now accounted for.

### The loading pattern that almost tripped me up

The kernel uses 128 threads per block: 4 warps of 32 threads each, where each warp is responsible for 16 rows of the output tile. That gives exactly 4 times 16 = 64 rows, matching the Q tile size. The thread count is a direct consequence of the MMA tile geometry, not an arbitrary choice.

Those 128 threads need to load 8192 floats from VRAM into shared memory. The standard approach is a strided loop: each thread starts at its own index and jumps by 128 each iteration. Thread 0 loads elements 0, 128, 256, and so on. Thread 1 loads 1, 129, 257. This guarantees that on every iteration, consecutive threads read consecutive addresses, which is the definition of coalesced access. A non-coalesced load would serialize the memory transactions and cost significant bandwidth.

I initially wrote this with row and column indexing, computing `row = idx / Bd` and `col = idx % Bd` and then calculating the global address from there. It worked, but it was unnecessary complexity. Since Q is row-major and the tile is a contiguous block of rows, the linear index maps directly:

```cpp
for (int k = 0; k < TILE_SIZE; k += NUM_THREADS) {
    int idx   = tid + k;
    int g_idx = blockIdx.x * TILE_SIZE + idx;
    staging[idx] = Q[g_idx];
}
__syncthreads();
```

`blockIdx.x * TILE_SIZE` is the offset for this block's group of 64 tokens. No division, no modulo. Sometimes the clever approach is the dumb one.

### The quantization two-pass problem

I wanted to quantize the floats as I loaded them from VRAM, avoiding the staging buffer entirely. But block scaling killed that idea. To compute the scale factor for a block of 32 elements, you need the maximum absolute value across all 32. During the strided load, a single thread does not see all 32 elements of the same scaling block because they are spread across multiple iterations. Thread 0 sees elements 0, 128, 256, never elements 1 through 31.

So quantization has to be a separate pass after loading. The staging buffer exists specifically because of this dependency. First load everything as FP32, barrier, then quantize in a second pass where each thread handles complete 32-element blocks:

```cuda
for (int i = tid; i < NUM_SCALE_BLOCKS; i += NUM_THREADS) {
    uint8_t scale   = compute_scale_ue8m0(&staging[i * BLOCK_ELEMENT]);
    Q_scales[i]     = scale;
    float   scale_f = exp2f((float)(scale - 127));

    for (int j = 0; j < BLOCK_ELEMENT; j++) {
        float val = staging[i * BLOCK_ELEMENT + j] / scale_f;
        Q_quant[i * BLOCK_ELEMENT + j] = encode_fp4_e2m1(val);
    }
}
__syncthreads();
```

256 scaling blocks, 128 threads, 2 blocks per thread. Each thread processes its blocks sequentially, scanning for the max, computing the scale, dividing, and encoding all 32 values. It is not fast, there is a lot of branching in `encode_fp4_e2m1` and the inner loop is purely sequential, but it works. Optimization comes later.

K gets the exact same treatment: load into `staging` (overwriting Q's FP32 data, which is already quantized and safe in `Q_quant`), barrier, quantize into `K_quant` and `K_scales`, barrier.

### The MMA fragment loading: the part I could not figure out alone

This is where I spent the most time. The MMA instruction expects each of the 32 threads in a warp to hold specific bytes from the A and B matrices in its registers. Not just any bytes but the exact bytes that correspond to that thread's position in the matrix tile. The mapping is defined in the PTX ISA fragment layout tables.
For fragment A, a [16 x 32] slice of Q, each thread holds registers that pack FP4 values in their 8-bit containers. The lane ID determines which row and which K-column range the thread is responsible for. The registers are not contiguous in memory: they cover two different K-column ranges separated by a stride of 16 columns.

For fragment B, the K matrix accessed as its transpose, each thread holds registers covering one token and one range of head-dimension positions.
The scale factors follow the same lane-based assignment: each thread looks up the scale for the block it loaded and passes it to the MMA.

What I thought I understood at this stage turned out to be wrong in almost every detail. The register count, the lane grouping, the K stride, the scale index: all of it had to be corrected later through empirical testing. The code I had at this point compiled and passed isolated tests with hardcoded values, which masked the errors. Sections 11 through 14 document what was wrong and how it was fixed.

### The MMA loop

With fragments loaded, the MMA itself is straightforward. The outer loop covers 8 column tiles of S, the inner loop accumulates 4 K-chunks along the head dimension:

```cpp
for (int n_tile = 0; n_tile < N_TILES; n_tile++) {
    float acc[ACC_PER_THREAD] = {0.f};

    for (int k_tile = 0; k_tile < K_TILES; k_tile++) {
        // load fragments and scales
        // call asm volatile MMA
    }
}
```

The accumulators are both input and output. On the first K-chunk they are zero. Each subsequent MMA adds its partial product. After 4 iterations, the accumulators hold the complete dot products for this thread's slice of the output tile. Four warps, 8 column tiles each, 4 accumulators per thread: the full [64 x 64] score matrix lives entirely in registers. No global memory is touched for the intermediate result. That is the whole point of the fused kernel.

The code shown here is intentionally simplified. The correct register count, lane assignments, and index formulas are established in sections 11 through 14 after the correctness failures described in this section 8.

---

## 9. The First Correctness Test

### The result

Section 8 ended with the MMA loop assembled and the score matrix living in registers. I ran the correctness test against a float32 CPU reference.

Cosine similarity: 0.06.

*Cosine similarity measures the angle between two output vectors. A value of 1.0 means the GPU and CPU outputs point in exactly the same direction. A value of 0 means they are completely uncorrelated. At 0.06, the GPU was producing noise.*

### What the number does not tell you

The three validated components were not the problem. The encoding function worked. The block scaling worked. The isolated MMA test from section 4 still printed 256.0 correctly. The problem had to be in how the full kernel assembled those pieces together.

The most obvious candidate was the fragment loading. In section 7, I described loading `a0` and `a1` for the A fragment and `b0` for B. That description was my best understanding at the time. It was wrong in two ways I had not yet discovered.

The issue with cosine as a diagnostic is that it gives you one number. It tells you whether the final answer is right or wrong. It tells you nothing about where it went wrong.

---

## 10. The Fragment Layout Problem

### No documentation

The fragment layout — the mapping between lane IDs and matrix positions is defined by the hardware for every MMA instruction variant. The hardware defines precisely which matrix positions belong to which thread.

For FP16 and BF16 MMA variants, NVIDIA documents these layouts with diagrams in the PTX ISA specification. For `mxf8f6f4 m16n8k32` on SM120, those diagrams do not exist. The instruction is listed, the operand counts are given, but the per-lane mapping is absent.

I posted a question on the [NVIDIA developer forums](https://forums.developer.nvidia.com/t/fragment-layout-for-mma-sync-aligned-m16n8k64-with-fp4-e2m1-and-block-scaling-on-sm120/364020) on March 19. No response.

### The guessing phase

So I started guessing. The PTX ISA gives you the operand count for the instruction: 4 registers for A, 2 for B, 4 for the accumulator. But it does not tell you which matrix row or column each register corresponds to for a given lane. That mapping is what I was guessing.

*A lane is a thread within a warp, numbered 0 to 31. For a 16x32 tile of Q spread across 32 threads, each lane owns a specific slice. The question is: which slice?*

I tried `lane % 16` for the row index and `lane / 16` for the K-column group. I tried different strides between registers: 4, 8, 16 columns apart. Each attempt compiled without errors and produced a different cosine between 0.01 and 0.09. A cosine of 0.01 and a cosine of 0.09 both mean wrong. There was no gradient to follow. I could not tell if I was getting closer to the correct layout or further away, because the only signal I had was a single float that told me nothing about where the error was.

The problem was not the guesses. The problem was the test.

---

## 11. Finding the Right Diagnostic

### The identity matrix test

I replaced the attention test with a simpler one designed to give precise information. I set Q and K to the 64x64 identity matrix, zero-padded to the [64, 128] shape the kernel expects. The expected output is S = Q times K-transpose = I_64. Every diagonal entry is 1.0, every off-diagonal entry is 0.

*When you multiply a matrix by its own transpose, the result at position [i][j] is the dot product of row i with row j. For the identity matrix, row i contains a single 1.0 at position i and zeros everywhere else. The dot product of row i with row j is 1.0 only when i equals j, and 0 in every other case. That is why the result is the identity matrix again.*

I also isolated the fragment loading from everything else by writing a separate debug kernel that loads directly from global memory, skips shared memory entirely, and hardcodes all scales to 1.0. One variable at a time.

Running this with my best guess at the fragment layout: 20 non-zeros instead of 8. Columns 0 through 7 of the identity matrix should produce exactly 8 ones on the diagonal. There were 12 phantom values in wrong positions.

<img src="/identity_matrix_test.svg" alt="Identity matrix test" style="width:100%;max-width:680px;">

### What it reveals

This test has two properties the attention test does not. First, 1.0 is exactly representable in FP4 E2M1, so quantization cannot explain wrong results. Second, each non-zero in S comes from exactly one dot product. If S[2][5] is non-zero when it should be zero, it means the threads responsible for row 2 of Q and column 5 of K loaded data they were not supposed to load. The wrong value points directly to the wrong lane.

For the first time, I could see where the problem was.

---

## 12. Fixing the A Fragment

### The wrong register count

We are inside the k_tile loop, at the moment where each thread loads its slice of Qinto registers before calling the MMA. This is the fragment load: the step thatdistributes the A matrix across the 32 threads of the warp so the Tensor Core can consume it. The A tile here is a [16 x 32] slice of Q: 16 token rows, each spanning 32 elementsalong the head dimension. 512 values in total, divided exactly across 32 threads.Each thread holds 4 registers of 4 FP4 elements each: 32 × 4 × 4 = 512. The distribution is exact. There is no slack.

My original code loaded two registers per thread. That covered only half the A tile. The other 256 elements had no owner. The hardware read those positions from whatever happened to be in the registers at that point, uninitialized memory, and computed dot products against it. No error was raised. The kernel ran to completion and wroteoutput that looked like real attention scores.

### The wrong lane grouping

The register count was not the only problem. Even with the right number of registers, the data inside them was wrong.
The K-column each thread loads depends on its lane ID. My formula used `lane / 16`, which divides the 32 lanes into two groups of 16. Every thread in the first grouploaded the same K-column positions, and every thread in the second group loaded another identical set. Two groups of 16 threads duplicating each other's work means only two distinct K-column ranges were ever read. The other half of the A tile, the K-columns that no group was assigned to, was never touched.

The correct formula is `lane % 4`, which creates four groups of eight threads. Each group gets a different K-column subgroup, and together the four groups cover all 32 K-columns of the A tile without overlap or gap.

### The fix

With `lane / 4` for the row and `lane % 4` for the K-column subgroup, each of the32 threads gets a unique assignment: 8 row positions times 4 K-column groups. No two threads duplicate each other's work and no position in the A tile goes unloaded.
The four registers follow a pattern that is worth making explicit because it is easy to get wrong. The first two registers, a0 and a1, cover the same K-column range but different rows: a0 for row0, a1 for row0+8. The second two registers, a2 and a3, shift the K-columns by 16 and repeat the same row pattern: a2 for row0, a3 for row0+8. The row alternates across registers rather than incrementing, which is the opposite of what feels natural.

After this fix: 8 non-zeros, correct diagonal positions for columns 0 through 7.

---

## 13. Fixing the B Fragment

### The same class of error

With the A fragment correct, I extended the debug kernel to loop over all 8 column tiles and check the full 64x64 score matrix. The non-zero count dropped back to wrong values. A different fragment, the same mistake.

B is the K matrix, accessed as its transpose by the MMA `.col` modifier. Each thread loads a slice of K corresponding to one token and one range of head-dimension positions. My original code used `lane / 16` to select the token and `lane % 16` for the head-dimension offset. Two groups of 16 lanes, each duplicating the other within its group. Only two distinct tokens were ever loaded per tile. The rest of K was uninitialized memory.

### The fix

The correction follows the same logic as A. `lane / 4` selects the token within the column tile, giving 8 distinct token assignments. `lane % 4` selects the head-dimension subgroup, giving 4 distinct K-column ranges. The two B registers, b0 and b1, cover the same token at K-column positions 16 apart, mirroring the stride pattern from the A fragment.

### Confirmation

After this fix: 64 non-zeros, S equals I_64 exactly.

A second test with Q and K filled from {-2, -1, 0, 1, 2}, values all exactly representable in FP4 E2M1, gave cosine similarity 1.000000 and max absolute error 0.000 across all 4096 elements of S. The fragment loading was correct.

The identity matrix test took less than an afternoon to build. The previous weeks of guessing produced nothing because I had no way to see where the errors were. Once I could observe which specific cells in S were wrong, both fixes took less than an hour.

---

## 14. The Remaining Bugs

With the fragment layout validated on clean data, I moved the corrected loading code into the full kernel that uses shared memory and quantization. The cosine dropped back to 0.06. A different set of problems, same symptom.

### The K stride

The fragment loading reads from `K_quant`, the quantized K tile in shared memory. K is stored row-major: `K_quant[token * Bd + head_dim]`, where `Bd` is 128. My indexing used `token * BQ` instead, treating `BQ` (64) as the stride between tokens. Every K access was reading from roughly half the correct address in shared memory. The kernel was computing dot products between Q rows and bytes from the wrong tokens.

Fix: replace `BQ` with `Bd` in the K stride.

Cosine after fix: still 0.06. The next bug was masking any improvement.

### The K scale index

Each 32-element block of K has a scale factor stored in `K_scales`. The index into that array is `token * (Bd / BLOCK_ELEMENT) + k_tile`. The stride between tokens is `Bd / BLOCK_ELEMENT`, which is 4. My code used `BQ / BLOCK_ELEMENT` instead, which is 2. Every scale lookup was landing at the wrong position, applying an incorrect normalization factor to the K values before the MMA.

Fix: replace `BQ` with `Bd` in the scale stride.

Cosine after fix: still 0.06. Two more bugs waiting.

### The scope errors

`q_row0` holds the output row index for this thread. It was declared inside the k_tile loop but needed by the output write that happens after both loops. Out of scope, the compiler silently reused whatever value was on the stack, writing output to arbitrary row positions.

`out_col` was declared twice: once inside the n_tile loop and once in the output write. The second declaration shadowed the first. Both used the same formula so the values were identical, but the shadowing masked a structural problem that would have caused bugs if the formula had ever changed.

Both variables needed to be declared once before both loops.

Cosine after fix: still 0.06. One more.

### The lane collision

The output write used `q_row0 = warp_id * MMA_M + (lane % 16)`. With `lane % 16`, lane 0 and lane 16 both compute `q_row0 = 0` and write to the same memory address. Lane 4 and lane 20 both write to row 1, lanes 8 and 24 to row 2, and so on. Every row was being written twice by two different threads, one overwriting the other.

The correct formula is `lane / 4`, which gives each row a unique group of four lanes. No two threads write to the same address.

Cosine after fix: 0.19. The output was wrong but no longer completely random.

### The V accumulation

With S computed correctly in registers, the kernel accumulates the attention output as `O = softmax(S) times V`. Each thread is responsible for two output columns. For each column tile of S, it needs to sum the softmax-weighted V rows across all eight K-tokens in that tile.

The problem: each thread only holds the softmax weights for two of those eight tokens. The other six are in neighboring threads. My first implementation used the butterfly reduce to collect the V contributions, the same pattern used earlier for the row maximum. That was the wrong operation here.

The butterfly adds values from neighboring threads. Thread 0 accumulates for output column 0. Thread 1 accumulates for output column 2. After the butterfly, thread 0 was adding its column-0 contribution to thread 1's column-2 contribution. Different output dimensions mixed together. The result had no meaning.

The correct approach is different: each thread uses `__shfl_sync` to fetch the softmax weights from its three neighbors explicitly, then multiplies each neighbor's weights against the V values for its own output columns. The accumulation stays local to each thread's assigned dimensions. No cross-dimension mixing.

Cosine after fix: 0.81.

### The race condition

The kernel was launched with four blocks of 128 threads each. One block already covers the full 64-row Q tile: four warps times 16 rows each. Four blocks meant four independent groups of threads all writing to the same output array simultaneously, with the last writer winning arbitrarily.

One block is enough.

Cosine after fix: 0.81 confirmed. The remaining gap from 1.0 is quantization noise, not a correctness issue. Section 17 confirms this.

---

## 15. The Scale Layout

### The same problem

The MMA instruction takes one `uint32_t` per thread for `scale_a` and one for `scale_b`. But the A tile has 16 rows and each row needs its own scale factor. The B tile has 8 columns and each column needs its own scale. One register per thread cannot hold all of that.

The hardware must be reading specific bytes from specific lanes, distributing the scale responsibility across the warp just as it distributes the fragment data. But the PTX ISA does not document this mapping for `mxf8f6f4 m16n8k32` on SM120.

The same situation as the fragment layout. The same approach.

### The probing method

I filled Q and K with all 1.0 and set all scales to 127, which is `2^(127-127) = 1.0`. With all inputs equal to 1.0 and scale equal to 1.0, every MMA output is 32.0: the sum of 32 multiplications of 1.0 times 1.0.

Then I ran the kernel 32 times. Each run set exactly one lane's `scale_a` to 128, which is `2^(128-127) = 2.0`, while all other lanes kept 127. If lane L's scale register controls row R of the output, then row R doubles from 32.0 to 64.0. I recorded which rows changed for each target lane.

### The results

| Lane condition | Row affected |
| --- | --- |
| lane % 4 == 0 | lane / 4, rows 0 through 7 |
| lane % 4 == 1 | lane / 4 + 8, rows 8 through 15 |
| lane % 4 == 2 | no effect |
| lane % 4 == 3 | no effect |

The same probing on `scale_b`: only lanes where `lane % 4 == 0` have any effect, and lane L controls column `lane / 4`.

This is consistent with the A fragment structure established in section 11. The register `a0` covers row0 and its scale is supplied by the lane with `lane % 4 == 0`. The register `a1` covers row0+8 and its scale comes from `lane % 4 == 1`. Lanes 2 and 3 load fragment data normally through their A registers, but the hardware does not read their scale values. They are silently ignored.

This also resolved a secondary issue from the original code. The kernel had been packing the scale byte four times into the `uint32_t`: `sa | (sa << 8) | (sa << 16) | (sa << 24)`. This happened to work because the hardware reads byte 0 of the register, and packing the same byte four times keeps byte 0 correct. Once the actual mapping was clear, the cast became a direct `(uint32_t)sa`. Same result, explicit intent.

---

## 16. Online Softmax and the Accumulation of V

### Why a warp reduction is unavoidable

With correct fragment loading, scale indexing, and output addressing, the kernel was computing S = Q times K-transpose correctly in registers. The second half of the fused attention is `Out = softmax(S) times V`.

Writing S to global memory, running softmax separately, and reading it back would defeat the entire purpose of the fused kernel. Instead, I used the online softmax algorithm introduced in the Flash Attention paper. The idea is to maintain a running state that updates as each column tile of S is computed, so the softmax normalization is applied incrementally without ever materializing the full score matrix.

The D output layout from the MMA places a row's scores across four threads. The eight scores for a complete row are split: thread 0 holds columns 0 and 1, thread 1 holds columns 2 and 3, thread 2 holds columns 4 and 5, thread 3 holds columns 6 and 7. To compute the row maximum needed for numerically stable softmax, those four threads must communicate.

*A butterfly reduce is a communication pattern where threads exchange values in log2(N) rounds, such that after the rounds every thread holds the result of the operation across all N threads.* Running `__shfl_xor_sync` twice with masks 1 and 2 covers all four pairings in two rounds. After these two rounds, all four threads in the group hold the global maximum of their row. This warp reduction is not a choice. It is a direct consequence of the fragment layout.

### The online state

The running state for each row has three components: `m`, the maximum score seen so far; `l`, the sum of exponentials seen so far; and `O`, the unnormalized output accumulated so far. When a new tile arrives:
new_m = max(m, tile_max)
alpha = exp(m - new_m)
new_l = alpha * l + sum(exp(score - new_m) for score in tile)
new_O = (alpha * l * O + weighted V contribution) / new_l

`alpha` is the rescaling factor. If the new tile contains a score larger than anything seen before, `alpha` is less than 1 and the previous accumulator shrinks proportionally. If the maximum does not change, `alpha` is 1 and the old output is unchanged.

### The V accumulation bug

The V accumulation bug and its fix are described in section 13. After the fix, the cosine with random inputs reached 0.81.

### The race condition

The kernel was launched with four blocks and 128 threads per block. Four blocks meant four independent groups of four warps, all writing their outputs to the same memory addresses with no coordination. Each block computed correct results and then overwrote what the previous block had written. One block is enough. Changing `<<<4, NUM_THREADS>>>` to `<<<1, NUM_THREADS>>>` fixed it.

### Validation

At this point every correctness bug had been fixed. The V accumulation and race condition bugs are documented in section 14 — they were resolved before the softmax was wired in. With those fixes in place, the full pipeline from Q and K loading to softmax to V accumulation produced: 
cosine similarity : 1.0000  PASS
ref[0..7] : -0.446  -0.879  -0.450   0.511   0.940   0.968  -0.947  -0.049
out[0..7] : -0.446  -0.879  -0.450   0.511   0.940   0.968  -0.947  -0.049

Bit-exact. This confirms that the entire pipeline is correct: FP4 quantization, block scaling, MMA fragment loading, online softmax, and V accumulation all produce the right answer when given inputs that are exactly representable in FP4 E2M1.

The 0.81 cosine observed earlier with random inputs in [-1, 1] is the intrinsic precision cost of MXFP4 at `scale_vec::1X` granularity. FP4 E2M1 has only eight representable magnitudes. With one scale covering 32 elements, a single outlier sets the scale for the entire block and the remaining values lose resolution. The CPU reference operates on the original float32 values, so the comparison is unfair. The kernel is correct. The 0.81 is an architectural constraint, not a bug.

---

## 17. The K Tile Loop

The kernel validated in section 16 had one hard limitation: K was loaded from a hardcoded offset, meaning it only ever saw the first 64 tokens of the key sequence.
For any real attention computation, K can have thousands of tokens. The kernel needed a loop.

The change is conceptually simple. Instead of loading one K tile before the MMA loop, the outer structure becomes a loop over sequence tiles. For each tile, the kernel loads 64 rows of K into shared memory, quantizes them, runs the full MMA and softmax update, then moves to the next tile. The online softmax state — `m`, `l`, and `O` is declared before the loop and persists across all tiles. Each tile's scores are folded into the running state via the rescaling factor `alpha`.

One detail worth noting: the V access index must account for the tile offset. When accumulating the attention output, the V row index is not just the local token position within the tile but `seq_tile * BQ + local_token`. Without that offset, every tile reads from the beginning of V regardless of which K tokens it just scored.

A `__syncthreads()` at the end of each iteration ensures the staging buffer is free before the next tile's load overwrites it.

Validation with two test cases confirms correctness. With `seq_k = 64` (single tile, regression test): cosine 1.0000. With `seq_k = 128` (two tiles, first real test of the loop): cosine 1.0000.

The kernel now processes attention over arbitrary key sequence lengths, in multiples of 64 tokens.

---

## 18. Softmax Scaling

The attention formula is softmax(Q×Kᵀ / sqrt(d)) × V. The division by sqrt(d) was missing from the kernel until this point.

Without it, the scores in S grow with the head dimension. Each score is a dot product of two vectors of length d. If Q and K have values around 1, the scores are on the order of sqrt(d) for d=128, that is around 11. Feeding large values into softmax pushes it toward saturation: the maximum score gets a weight close to 1 and everything else collapses toward 0. The attention output becomes a near-copy of the V row corresponding to the single highest score, losing all the nuance of the weighted average.

Dividing by sqrt(d) brings the scores back to order of magnitude 1 before the softmax, keeping the output distribution balanced.

In the kernel, this is a single multiply applied to the four accumulators immediately after the inner k_tile loop and before the softmax reduction:

```cpp
for (int i = 0; i < ACC_PER_THREAD; i++)
    acc[i] *= softmax_scale;  // softmax_scale = 1/sqrt(Bd) = 1/sqrt(128)
```

The CPU reference was updated to apply the same scaling, and both test cases pass at cosine 1.0000.

---

## 19. Multi-Head Attention and Arbitrary Head Dimensions

### The limitation

Up to this point, the kernel had three hard constraints that made it unusable on real models. The head dimension was fixed at 128. The output covered only 8 columns per thread, which happened to match the previous test setup but was wrong for any real head dimension. And the kernel processed a single head with no concept of batch or head index.

### Template parameter for head dimension

Different models use different head dimensions. GPT-2 uses 64, LLaMA uses 128, some recent models use 256. Hardcoding 128 excludes most of them.

The solution is a C++ template parameter. Instead of a fixed constant, the kernel becomes `template<int HEAD_DIM>`. The compiler generates a separate binary for each instantiation: `fused_fp4_attention<128>` and `fused_fp4_attention<64>` are two distinct kernels, each with their own compile-time constants for tile sizes, loop bounds, and register counts. No runtime branching, no overhead.

The only constraint is that `HEAD_DIM` must be a multiple of 32, which is the MMA reduction dimension. Values like 64, 96, 128, 160, and 256 all work.

### The output accumulator bug

Fixing the head dimension revealed a deeper bug. The original kernel kept two scalar accumulators per thread for the V output: `O0_c0` and `O0_c1`. That was correct when the output had 8 columns total, but wrong for any real head dimension.

For `HEAD_DIM=128`, each thread is responsible for 128 / 4 = 32 output column pairs, not 2. The previous kernel was writing 2 values and leaving 126 columns at zero.

The fix replaces the two scalars with an array `O0[V_COL_TILES * 2]` where `V_COL_TILES = HEAD_DIM / MMA_N`. For `HEAD_DIM=128` that is 32 floats per row per thread. The V accumulation becomes a loop over all output column tiles, and the online softmax rescaling (`alpha`) must be applied to every element of that array at each update step.

### Multi-head and batching

Each block processes one `(batch, head)` pair independently. The mapping is:

```cpp
int batch_idx = blockIdx.x / heads;
int head_idx  = blockIdx.x % heads;
```

Each block computes its own offset into the Q, K, V, and Out tensors and works without any coordination with other blocks. The launch becomes
`<<<batch * heads, NUM_THREADS>>>`.

### Validation

Six test cases confirm correctness across configurations:

| Config | Result |
| --- | --- |
| head_dim=128, seq_k=64, 1 head | cosine 1.0000 PASS |
| head_dim=128, seq_k=128, 1 head | cosine 1.0000 PASS |
| head_dim=64, seq_k=64, 1 head | cosine 1.0000 PASS |
| head_dim=64, seq_k=128, 1 head | cosine 1.0000 PASS |
| head_dim=128, seq_k=128, batch=1 heads=4 | cosine 1.0000 PASS |
| head_dim=128, seq_k=128, batch=2 heads=4 | cosine 1.0000 PASS |

The kernel now handles arbitrary head dimensions, multiple heads, and batched inputs.

---

## 20. First Benchmark and the NCU Diagnosis

### The numbers

With the kernel functionally complete, I ran a first benchmark on the RTX 5070 Ti. Configuration: batch=1, heads=32, seq_q=64.

| head_dim | seq_k | kern_ms | TFLOPS | BW GB/s |
| --- | --- | --- | --- | --- |
| 128 | 128 | 0.072 | 1.87 | 87.8 |
| 128 | 512 | 0.255 | 2.11 | 74.0 |
| 128 | 1024 | 0.530 | 2.03 | 67.3 |
| 64 | 128 | 0.037 | 1.82 | 85.2 |
| 64 | 512 | 0.102 | 2.62 | 92.1 |
| 64 | 1024 | 0.192 | 2.80 | 92.9 |

The RTX 5070 Ti has a theoretical FP4 throughput of 474 TFLOPS. We are at 2.8, which is 0.6% utilization. Before optimizing, I needed to know exactly where the time was going.

### What NCU said

The first metric that jumped out was "No Eligible" at 95.23%. This means the warp scheduler found no warp ready to execute 95% of the time. The GPU was spending almost all its time waiting.

*A warp scheduler looks for eligible warps — warps that have their input data ready and can execute an instruction. When none are eligible, the SM is idle. High "No Eligible" is the definition of a latency-bound kernel.*

The occupancy was 7.94% against a theoretical maximum of 8.33%. The reason:
Block Limit Shared Mem  : 2 blocks per SM
Block Limit Registers   : 8 blocks per SM

The shared memory was the binding constraint. With 41 KB of static shared memory per block and 99 KiB available per SM, only 2 blocks could fit per SM simultaneously. With 2 blocks of 2 warps each, the SM had 4 active warps. A GPU needs roughly 32 warps per SM to fully hide memory latencies.

The long scoreboard stall was at 81%. Warps were stalling on data from global memory.
The culprit was V: the V accumulation loop was accessing V directly from global memory inside a double loop, generating thousands of uncoalesced accesses per thread.

### The fix: V in shared memory

The solution was to load each V tile into shared memory before the MMA loop, exactly as K is loaded. This adds 32 KB to the shared memory budget for HEAD_DIM=128, bringing the total to about 80 KB. The double buffering for K was removed to fit within the 99 KiB limit.

After this change, the TFLOPS roughly doubled:

| head_dim | seq_k | before | after |
| --- | --- | --- | --- |
| 128 | 1024 | 1.26 TFLOPS | 2.03 TFLOPS |
| 64 | 1024 | 1.41 TFLOPS | 2.80 TFLOPS |

The bandwidth on the kernel side went from ~50 GB/s to ~92 GB/s on the best configurations, confirming that the V global memory accesses were the dominant stall.

### What NCU says now

After loading V into shared memory, the new binding constraint is occupancy. With 80 KB of shared memory per block, only 1 block can fit per SM instead of 2. Combined with 128 registers per thread — driven by the large `O0` accumulator array the theoretical occupancy stays at 8.33%.

The kernel is still latency-bound, but for a different reason. The shared memory budget is now the wall. Reducing it requires either a smaller tile size or a different strategy for the output accumulator. That is the next step.

---

## 21. First Profiling Round: What We Tried and What We Learned

Running NCU on the first functional version of the kernel revealed a single dominant metric: No Eligible at 95.23%. The warp scheduler found no eligible warp to execute 95% of the time. The SM was idle almost continuously.

The cause was V. The V accumulation loop was reading directly from global memory inside a double loop over output column tiles and softmax weight indices. For HEAD_DIM=128 and a single n_tile, each thread generated 128 uncoalesced global memory accesses. Across 8 n_tiles and multiple seq_tiles, that amounted to thousands of scattered reads per block. The long scoreboard stall confirmed it at 81%.

The fix was straightforward: load each V tile into shared memory before the MMA loop, exactly as K is already loaded. This eliminated the global memory dependency during accumulation. The result was roughly a 2x improvement in TFLOPS, and the kernel bandwidth jumped from ~50 GB/s to ~92 GB/s on the best configurations.

That introduced a new constraint. Adding a 32 KB V tile to the shared memory budget brought the total to 80 KB per block. With 99 KiB available per SM on SM120, only one block could fit at a time. The occupancy stayed at 8.33% with four active warps per SM instead of the ~32 needed to hide latencies.

The next move was to reduce the shared memory footprint. The staging buffer, used to load K as float32 before quantization, was the largest single consumer at 32 KB. By switching it from float32 to `__half`, the footprint dropped to 16 KB. The same buffer then gets reused for V, eliminating the need for a separate V tile buffer.

The precision trade-off is negligible. FP16 has 10 bits of mantissa. FP4 E2M1 has 1. Any rounding introduced by the float32 → float16 → float32 conversion disappears completely in the FP4 quantization step. The test suite confirmed cosine 1.0000 on all configurations after the change.

The shared memory budget dropped to ~32.5 KB, which should allow 3 blocks per SM instead of 1. The TFLOPS improved further, reaching 3.1 on head_dim=64, seq_k=1024.

A comparison against PyTorch SDPA on identical configurations put the gap in perspective. PyTorch FP16 reaches 15 TFLOPS on the same hardware for the same
problem size. We are at 3 TFLOPS, roughly 5x slower.

The gap is not algorithmic. It is architectural. PyTorch SDPA via FlashAttention receives data already in FP16. No quantization happens inside the kernel. Our kernel quantizes Q, K, and V on the fly at every call, inside the main loop. The quantization pass `encode_fp4_e2m1` called 8192 times per tile, with a chain of
eight comparisons per call takes roughly as long as the MMA itself. The Tensor Cores are idle most of the time, waiting for the quantization pass to finish.

This is the fundamental tension of the current design. On-the-fly quantization keep sthe interface simple: the kernel accepts float32 inputs just like any standard
attention kernel. But it means the FP4 Tensor Cores are not the bottleneck the scalar quantization loop is. Closing the gap with PyTorch requires either vectorizing the quantization, or moving it outside the kernel entirely and accepting pre-quantized inputs. Both directions are worth exploring, and that is where the next round of
optimization begins.

---

## 22. Deep NCU Analysis: What the SASS Revealed
 
Section 21 identified the core problem: on-the-fly quantization dominates the kernel runtime. But saying "quantization is slow" is not actionable. I needed to see exactly which instructions were responsible and how much they cost. That meant reading the SASS, the actual machine code the GPU executes.
 
I ran `ncu --set full` on the kernel and exported the source-level report. The kernel compiled to about 5,900 SASS instructions. Four of them were QMMA, the FP4 Tensor Core multiply. The other 5,896 were overhead.
 
### The division that was not a division
 
The first thing that stood out was 129 calls to a function called `__cuda_sm3x_div_rn_noftz_f32_slowpath`. Each call appeared as a real `CALL.REL.NOINC` instruction in the SASS, with register save/restore and dozens of instructions per invocation.
 
The GPU does not have a hardware division unit. What it has is `MUFU.RCP`, an instruction that computes the reciprocal 1/b on the Special Function Unit pipe in about one cycle. The result is accurate to roughly 22 bits of mantissa. A float32 has 23. That last bit matters to the IEEE-754 standard, so by default nvcc generates a software routine that starts with `MUFU.RCP`, then refines the result with two or three rounds of Newton-Raphson. That routine is the "slowpath". It is a full function call, not an inlined instruction.
 
In my kernel, the division happened in `compute_scale_ue8m0`. After finding the maximum absolute value in a block of 32 elements, the code divided each value by the scale factor before encoding to FP4. The scale is a power of two (UE8M0 format), so the division is exact in floating point. IEEE-754 precision on a result that will be rounded to one of eight magnitudes.
 
The fix was to replace the division with a multiply by the inverse: `exp2f((float)(127 - scale))` computes `2^(-exponent)` exactly, and multiplying is a single `FMUL` instruction. After the change, the 129 CALL instructions disappeared entirely. The kernel dropped from 5,900 to 4,200 SASS instructions, a 28% reduction.
 
I later found a thread on the NVIDIA developer forums where njuffa, a former NVIDIA engineer who designed FPU hardware, confirmed the behavior. The "sm3x" in the function name is misleading: the routine was written for Kepler (compute capability 3.x) around 2012 and has been reused on every architecture since, including Blackwell. The user in that thread measured a 3x speedup by replacing `/` with `__fdividef`.
 
### The 647 comparisons for a 32-element max
 
The second problem was more subtle. `compute_scale_ue8m0` finds the maximum absolute value across 32 elements using a sequential loop: `if (fabsf(block[i]) > max_abs) max_abs = fabsf(block[i])`. The compiler translated each iteration into two SASS instructions: `FSETP.GT` (compare, write predicate) followed by `FSEL` (conditional select). Two instructions per comparison, 31 comparisons per block, hundreds of blocks per tile. The total was 647 FSETP/FSEL instructions in the kernel.
 
The GPU has an instruction that does this in one step: `FMNMX`. It takes two values and a predicate that selects min or max, and produces the result in a single cycle on the same ALU pipe. The compiler was not using it because the C++ code was written as an `if` statement, which the optimizer does not always reduce to `FMNMX`. Replacing `if (a > max_abs) max_abs = a` with `max_abs = fmaxf(fabsf(block[i]), max_abs)` gives nvcc the hint it needs. The `fmaxf` intrinsic maps directly to `FMNMX`.
 
This cut the instruction count for the scale computation roughly in half. The effect on total kernel time was smaller than the division fix, because the FSETP/FSEL chain does not stall the pipeline as badly as a function call, but it reduced pressure on the ALU pipe that was already saturated by the quantization work.
 
### The byte-by-byte problem
 
The quantization pipeline writes each encoded FP4 byte to shared memory individually with `STS.U8`, an 8-bit store. The kernel had 66 of them. Each STS.U8 consumes a full shared memory transaction (32 bytes of bandwidth) to write a single byte. Packing four bytes into a `uint32_t` and writing once with `STS.32` would use the same bandwidth for 4x the data.
 
The same pattern appeared on the read side. Before each QMMA, the kernel loaded FP4 operands from shared memory with `LDS.U8`, one byte at a time, then reconstructed 32-bit registers using chains of `IMAD` (multiply-accumulate to shift and combine bytes). 104 LDS.U8 instructions, each generating shared memory bank conflicts measured at L1 Wavefronts Shared Excessive = 14,336.
 
These two problems, STS.U8 and LDS.U8, are the largest remaining performance bottleneck. They are a direct consequence of writing FP4 values as individual bytes in C++ rather than packing them into wider words before the store.
 
### The full picture
 
After the division fix and the fmaxf change, the SASS profile looked like this:
 
| Category | Instructions | % of total |
|---|---|---|
| FP4 quantization (encode + scale) | ~2,800 | 66% |
| Data movement (LDG, STS, LDS, STG) | ~800 | 19% |
| Softmax + V accumulation | ~400 | 10% |
| QMMA (Tensor Core compute) | 4 | 0.1% |
| Other (control, sync, address math) | ~200 | 5% |
 
The Tensor Cores executed four instructions out of 4,200. Everything else was preparation. The kernel is not compute-bound. It is quantization-bound.
 
---

## 23. Where the Time Goes and Why the Gap Is Expected
 
PyTorch SDPA with FlashAttention reaches 15 to 16 TFLOPS on the RTX 5070 Ti for the same problem size. This kernel reaches 2.4 to 3.4 TFLOPS, roughly 4 to 5 times slower.
 
The gap is not a bug. It is a design consequence.
 
FlashAttention receives Q, K, and V already in FP16. The kernel's main loop is almost entirely MMA instructions and softmax arithmetic. There is no format conversion inside the hot path.
 
This kernel receives Q, K, and V in float32. For every tile of 8,192 elements, it computes 256 block scales (one per 32 elements), finds the absolute maximum of each block, converts each value to the nearest FP4 representation through a chain of eight comparisons, packs the results into shared memory, and only then feeds them to the Tensor Core. That quantization pass runs twice per sequence tile, once for Q and once for K.
 
The quantization is doing useful work. It is not wasted computation. But it is scalar work on data that the Tensor Core will process in a single instruction. The ratio between the two is the gap.
 
For a production inference kernel, the solution is to move the quantization outside. In a decode loop, K and V live in a KV cache that is already quantized to FP4. Q is a single token that can be quantized in a separate, lightweight kernel. The attention kernel itself receives pre-packed `uint8` inputs and spends its time on MMA and softmax. That is what SageAttention3 does, and it is the natural next step for this project.
 
But the current kernel was never designed to compete on throughput. It was designed to make every step of the FP4 fused attention pipeline visible: the MMA fragment layout that is not documented, the container format that silently reads the wrong value if you shift by one bit, the scale distribution across lanes that required 32 probing runs to reverse-engineer, the division operator that turns into 129 function calls. None of that is visible in a CUTLASS template. Writing it from scratch with inline PTX was the only way to see it.
 
---
 
## 24. What I Would Do Differently
 
Looking back at several months of work, a few things stand out.
 
**Test with structured inputs first.** The weeks I spent guessing the fragment layout produced nothing because I was testing against random data. The identity matrix test from section 11 gave me precise, per-cell information about which lane loaded which position. Both the A and B fragment fixes took less than an hour once the right test existed. Every new MMA instruction variant should be validated with identity matrices before anything else.
 
**Read the SASS earlier.** The division slowpath was invisible at the C++ level. The scale computation looked like a single line of code. It took NCU and the SASS source view to reveal that one line was generating 129 function calls. Profiling should not be the last step. It should happen after every major code change.
 
**Do not optimize the wrong design.** The on-the-fly quantization was never going to be fast. I knew this conceptually from the start, but I kept optimizing around it (vectorized loads, FP16 staging, shared memory reuse) instead of changing the fundamental approach. The optimization that would have mattered most, pre-quantized inputs, was the one I deferred the longest.
 
**The fragment layout is the real contribution.** The MMA m16n8k32 fragment layout for FP4 E2M1 on SM120 is not documented anywhere in the PTX ISA. The scale distribution across lanes is not documented. The container format (nibble in bits 5-2, not bits 3-0) is mentioned in one sentence in the spec but never shown in a worked example. Figuring this out empirically and publishing it is the part of this project that will be useful to other people writing SM120 kernels. The kernel performance is secondary.
 
**The ecosystem is catching up.** When I started this project, SM120 support in the open-source stack was minimal. SageAttention3 was 5090-only in practice, FlashInfer had no SM120 path, and vLLM fell back to Marlin for FP4 on consumer Blackwell. By the time I am writing this, all three have added or are adding SM120 support. The gap I set out to fill is closing, which is a good thing. The documentation gap remains open.
 
*Code: [github.com/florianmattana/fp4-fused-attention-sm120](https://github.com/florianmattana/fp4-fused-attention-sm120)*
