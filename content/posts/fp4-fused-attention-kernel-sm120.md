---
title: "Building an FP4 Fused Attention Kernel on Consumer Blackwell (SM120)"
date: 2026-03-17
draft: false
tags: ["CUDA", "Blackwell", "SM120", "FP4", "Attention", "Tensor-Cores"]
description: "A deep dive into writing a fused FP4 attention kernel for the RTX 5070 Ti, using inline PTX and warp-level MMA instructions."
---

## 1. Why FP4 Fused Attention on Consumer Blackwell?

The attention mechanism in transformers scales quadratically with sequence length. On a consumer GPU with 12 GB of VRAM and 672 GB/s of memory bandwidth, that becomes a hard wall very quickly. The interesting thing about the RTX 5070 Ti (SM120, 46 SMs) is the raw throughput the Tensor Cores can deliver:

| Precision | Throughput |
| --- | --- |
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

**Option A – Inline PTX.** Write the kernel in CUDA C++ and embed the Tensor Core MMA instructions as inline PTX assembly. Each register is named in the `asm volatile` block, each byte is packed by hand, each scale is assigned to a specific lane. When something goes wrong, the debugging surface is the instruction itself, not a template instantiation stack. That visibility is why this article exists.

**Option B – CuTe (CUTLASS 3.x).** Use the NVIDIA template library that powers CUTLASS. CuTe handles tile indexing, shared memory swizzling, and MMA dispatch through a compile-time algebra. SageAttention3 takes this path for its SM120 FP4 kernel. The trade-off is visibility: CuTe generates the correct PTX, but the fragment layout, the scale distribution across lanes, and the container bit packing are all resolved inside the templates. If something goes wrong, the debugging surface is the template instantiation stack, not the instruction. Since the goal of this project was to document exactly how the FP4 pipeline works at the instruction level, using CuTe would have hidden the very thing I was trying to see.

**Option C – Patch an existing INT8 kernel.** Take a working fused INT8 attention kernel and swap the MMA instructions for FP4 equivalents. Faster to prototype, but brittle: the register layouts differ between INT8 and FP4 MMA, so the whole data flow would need reworking anyway.

I went with **Option A**. More manual work, more room for bugs, but every decision is visible in the code: which bytes go into which register, which lane reads which scale, how the nibble sits inside its container. That visibility is the point.

---

## 3. What the Kernel Needs to Do

Before writing a single line of code, it helps to map out the full problem.

Attention takes three inputs: Q (what each token is looking for), K (what each token offers), and V (the content each token carries). The computation is: score every query against every key, normalize those scores into probabilities with softmax, then use those probabilities to take a weighted average of the values.

On a GPU, this becomes two matrix multiplies with a softmax in between. S = Q times K-transpose produces the scores. P = softmax(S) normalizes each row. O = P times V produces the output. A naive implementation writes S to global memory after the first multiply, reads it back for softmax, writes again, reads again for the second multiply. For 4096 tokens, S alone is 64 MB. On a 12 GB card, that is a dealbreaker.

A fused kernel keeps S in registers the entire time. Softmax runs on the accumulators directly, and the second multiply consumes them in place. Making that work requires solving five things in sequence, each constrained by the hardware.

**Load the input tiles.** Q, K, and V live in global memory and must be brought into shared memory tile by tile. The tile size is bounded by the shared memory budget per SM, which on SM120 turns out to be 99 KiB, not 128 KiB as I initially assumed.

**Quantize on the fly.** The Tensor Core does not consume float32. Each tile must be converted to FP4 E2M1 with block scale factors before the multiply. This forces a two-pass approach: load as float32, compute the scale, then encode.

**Compute S = Q times K-transpose.** This is the first matrix multiply, executed on the FP4 Tensor Cores. The resulting score matrix is 64 times 64 floats distributed across four warps of 32 threads each, entirely in registers.

**Apply online softmax.** Each row of S is split across four threads by the MMA output layout. Computing the row maximum requires a cross-thread reduction using warp shuffle instructions.

**Compute O = softmax(S) times V.** This is the second matrix multiply. It accumulates incrementally as each tile of K is processed, consuming the softmax output directly from the registers where S was just computed.

Each step depends on which MMA instruction SM120 actually supports, what register layout it expects, and what format it accepts. That is the next section.

---

## 4. Picking the Right MMA Instruction

This is where I hit the first major wall. I started by reading the [PTX ISA docs](https://docs.nvidia.com/cuda/parallel-thread-execution/) looking for FP4 MMA instructions on Blackwell. The datacenter SM100 chips use `tcgen05.mma`, a new-generation instruction that operates on large tiles and uses a dedicated hardware unit called Tensor Memory. I assumed SM120 would have something similar.

It does not.

After digging through [CUTLASS issue #2800](https://github.com/NVIDIA/cutlass/issues/2800), a [thread on the NVIDIA developer forums](https://forums.developer.nvidia.com/), and [CUTLASS issue #3044](https://github.com/NVIDIA/cutlass/issues/3044), I pieced together the reality: SM120 uses the older Ampere-style **warp-level `mma.sync`** instructions. No Tensor Memory, no `tcgen05`. The specific instruction I need is:

`mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e2m1.f32.ue8m0`

Let me unpack that:

* `mma.sync.aligned` – warp-synchronous, all 32 threads participate.
* `kind::mxf8f6f4` – the MX (microscaling) family that covers FP4/FP6/FP8.
* `block_scale.scale_vec::1X` – each group of 32 FP4 values shares one 8-bit scale factor. I initially tried `scale_vec::2X` (one scale per 16 values, finer granularity) but it does not compile on SM120. Only 1X is supported, which means 6.25% overhead for the scale factors.
* `m16n8k32` – tile shape: 16 rows x 8 columns, with K=32 (32 FP4 values along the reduction dimension per instruction).
* `f32.e2m1.e2m1.f32` – FP32 accumulators, FP4 E2M1 inputs for both A and B matrices.
* `ue8m0` – the scale factor format (unsigned 8-bit exponent, no mantissa i.e., powers of two only).

The register budget for one MMA call is roughly 7 registers per thread: 2 for the A fragment, 1 for B, and 4 for the FP32 accumulator. This assumption turned out to be wrong. The correct count is 10: 4 registers for A, 2 for B, and 4 for the accumulator. Discovering that cost several weeks of debugging, and section 8 explains how.

### The shared memory budget

The shared memory budget is worth computing on paper before writing any code. Start with the tile dimensions and count the bytes at each stage:

| Buffer | Shape | Element size | Total |
| --- | --- | --- | --- |
| Q staging | 64 × 128 | 4 B (float32) | 32,768 B |
| K staging | 64 × 128 | 4 B (float32) | 32,768 B |
| Q quantized | 64 × 128 | 1 B (uint8) | 8,192 B |
| K quantized | 64 × 128 | 1 B (uint8) | 8,192 B |
| Q scales | 256 entries | 1 B (ue8m0) | 256 B |
| K scales | 256 entries | 1 B (ue8m0) | 256 B |

*Scale count: each tile has 64 × 128 = 8,192 elements. One scale per block of 32 gives 256 scales.*

If Q and K staging buffers are live at the same time, that is 64 KB before the quantized buffers are even counted. SM120 gives you 99 KiB with the optin path, and V is not in this table yet. Something has to give. Section 7 shows which buffers can share the same memory and which must coexist.

---

## 5. Testing the MMA Instruction

Before building the full fused kernel, I needed to verify that a single FP4 MMA instruction actually works. The idea is simple: load known values into registers A and B, run the MMA, and check that the FP32 accumulators contain the expected result.

I wrote a minimal warp-synchronous kernel, launched with `<<<1, 32>>>` so that exactly one warp executes. The kernel fills the A and B registers with constant FP4 values, calls the MMA via inline PTX, and prints the four accumulator floats from thread 0.

### The tile shape surprise

My first attempt used `m16n8k64` and I reasoned that since FP4 values are 4 bits each, 64 of them would fit in 32 bytes (the same as 32 FP8 values). The PTX assembler disagreed. It turns out the correct shape for FP4 on SM120 is **m16n8k32**: the k-dimension counts 8-bit *containers*, not individual FP4 values. Each container holds one FP4 nibble in bits 5-2, padded with zeros. This means you are effectively wasting half the container, but that is what the hardware expects.

### The encoding bug that cost me a full day

FP4 E2M1 encodes the value 1.0 as the 4-bit pattern `0b0010`. The container is an 8-bit byte, and the nibble must sit in bits 5-2, not bits 3-0. That means the correct byte for 1.0 is `0x08`: the nibble `0010` shifted left by two, giving the pattern `0b00001000`. If you place the nibble in bits 3-0 instead, you get `0x02`, which the hardware reads as a completely different value.

I initially filled every register with `0x22222222`, four bytes of `0x22` packed together. I thought I was encoding 2.0 in every position. What I was actually doing was placing the nibble in the wrong bit positions. The hardware read each byte as `0b00100010`, extracted the nibble from bits 5-2, which gives `0b1000` — the encoding for -0.0, not 2.0. So the MMA computed 32 multiplications of the wrong value.

After staring at bit layouts for longer than I would like to admit, I realized the nibble was in the wrong position. Switching to `0x08080808`, which places the 1.0 nibble correctly in bits 5-2 of each byte, and setting scale to 1.0, the MMA returned 32.0 exactly. That is 32 multiply-accumulates of 1.0 times 1.0. Correct.

The lesson: the FP4 container format is `00_SEMM_00` where the nibble occupies bits 5 through 2. Get the shift wrong and the hardware silently reads a different value with no error.

### The inline PTX

There is a reason this instruction appears as raw inline assembly rather than a clean C++ wrapper. The CUDA Core Compute Libraries (CCCL) expose `cuda::ptx` wrappers for many PTX instructions, which would normally be the right abstraction to use here. But at the time of writing, `cuda::ptx` does not provide wrappers for warp-level `mma.sync` on SM120. I exchanged with Federico Busato, who maintains CCCL at NVIDIA, on this exact gap. His read was that the wrappers would be useful but the decision was pending. I opened [CCCL issue #8146](https://github.com/NVIDIA/cccl/issues/8146) to track it. In the meantime, inline PTX is the only path.

Here is the `asm volatile` block as I first wrote it:

```c
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

This version has two registers for A and one for B. It is wrong. The correct instruction requires four A registers and two B registers. This assumption cost several weeks of debugging and is corrected in section 8. The block above is shown as written at this stage because it compiled and passed the isolated MMA test. The test was not thorough enough to catch the error.

The `"=f"` constraints are FP32 output registers, `"r"` are 32-bit integer input registers. The accumulator C is passed through as input (initialized to zero for the first call), and the result lands in D. The scale registers each pack four UE8M0 bytes into a single `uint32`.

With `a0 = a1 = 0x08080808` (all 1.0), `b0 = 0x08080808` (all 1.0), and scales set to 1.0 (`0x7F7F7F7F`, since UE8M0 byte 127 = 2^0 = 1.0), the result was 32.0 in every accumulator lane. That is 32 multiply-accumulates of 1.0 x 1.0, which is exactly right.

---

## 6. Encoding and Block Scaling

With the MMA validated on hardcoded constants, the next step was encoding arbitrary float values into FP4 E2M1 at runtime and computing block scale factors. The encoding function is covered in detail in a dedicated post on my [MXFP4 quantization kernel](https://florianmattana.com/posts/mxfp4_article/). What follows here is a summary of the key points as they apply to this kernel.

### The FP4 E2M1 format

FP4 has 1 sign bit, 2 exponent bits (bias 1), and 1 mantissa bit. That gives exactly 16 representable values:

| Binary | Value |
| --- | --- |
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

```c
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

```c
uint32_t pack = e0 | (e1 << 8) | (e2 << 16) | (e3 << 24);
```

End-to-end test: encode 1.0 into every position, pack, run MMA → 32.0. Encode 2.0 → 128.0. Both matched. The encoding function was correct.

### Why the encoding function is not enough

FP4 E2M1 maxes out at 6.0. If your input values are larger — and in attention, they absolutely will be — everything above 5.0 clamps to 6.0 and you lose all relative differences. For example, `encode_fp4_e2m1(12.0f)` and `encode_fp4_e2m1(10.0f)` both return `0x1C` (6.0). That is catastrophic for attention scores where the relative ordering is everything.

### Block scaling

The MX format handles this with a shared scale factor per block. Before encoding, you divide every value in a block by a common scale factor, bringing them into the representable FP4 range. The MMA hardware then multiplies the scale back in during the accumulation for free, no extra instructions.

Take a block of values: {12.0, 10.0, 3.0, -7.0}. The maximum absolute value is 12.0. If I choose a scale of 16, dividing gives {0.75, 0.625, 0.1875, -0.4375}. Now every value fits in FP4 range and critically, they encode to *different* FP4 values, preserving the relative ordering.

### Why the scale must be a power of two

The scale factor is stored in UE8M0 format: an 8-bit unsigned exponent with no mantissa. The actual scale value is 2^(byte - 127). This means only powers of two are representable. That is not a limitation, it is a feature. Multiplying by a power of two is just a bit shift in the exponent of a floating-point number, so the Tensor Core applies the scale with zero additional cost.

### Choosing the right scale

You want the smallest power of two that is greater than or equal to the maximum absolute value in the block. Rounding up avoids overflow. Rounding down would risk saturation for the largest values, exactly the ones you most need to preserve.

The formula: find `max_abs` across the block, compute `exponent = ceil(log2(max_abs))`, and the UE8M0 byte is `exponent + 127`.

Example: block maximum is 12.0. `ceil(log2(12.0))` = `ceil(3.58)` = 4. Scale = 2^4 = 16. UE8M0 byte = 4 + 127 = 131 = `0x83`.

### The device function

```c
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

Test: A = 8.0 everywhere, B = 1.0 everywhere. Scale for A: `max_abs` = 8.0, `ceil(log2(8))` = 3, UE8M0 byte = 130 = `0x82` (scale = 8). Scale for B: `max_abs` = 1.0, `ceil(log2(1))` = 0, UE8M0 byte = 127 = `0x7F` (scale = 1). After dividing A by 8, every element becomes 1.0, encoded as `0x08`. The MMA computes 32 x (1.0 x 1.0) = 32.0, then the hardware applies the scales: 32.0 x 8 x 1 = 256.0.

The kernel printed 256.0. The block scaling pipeline works end to end.

### The trade-off

On SM120, each scale factor covers a block of 32 elements along the K dimension. If one element is an outlier, say 100.0 in a block where everything else is around 1.0, the scale gets set to 128 and all the small values round to zero after division. Smaller blocks would preserve more detail, but `scale_vec::1X` is all we get on this hardware. It is a real limitation, and for attention (where softmax creates sharp distributions), it matters.

The kernel now has three validated building blocks: `encode_fp4_e2m1`, `compute_scale_ue8m0`, and the inline PTX MMA call. The next step is loading Q, K, V tiles into shared memory and wiring everything together into the fused attention loop.

---

## 7. Assembling the Kernel

The three building blocks worked in isolation. Encoding, scaling, MMA — all validated with hardcoded test values. Now I had to wire them together into an actual kernel that loads real data from VRAM, quantizes it on the fly, and runs the Tensor Core multiply. This is where the gap between "I have working pieces" and "I have a working kernel" became very real.

### The shared memory layout

Section 4 laid out the budget: if Q and K staging buffers are live at the same time, that is 80.5 KB before V. Too much.

The fix is simple. Q and K are never needed in FP32 at the same time. Load Q as FP32, quantize it into `Q_quant`, then reuse the same staging buffer for K. One 32 KB buffer instead of two:

| Buffer | Type | Size | Purpose |
| --- | --- | --- | --- |
| `staging` | float | 32,768 B | Reusable FP32 buffer for Q then K |
| `Q_quant` | uint8 | 8,192 B | Q tile after FP4 quantization |
| `K_quant` | uint8 | 8,192 B | K tile after FP4 quantization |
| `Q_scales` | uint8 | 256 B | One UE8M0 scale per 32 Q elements |
| `K_scales` | uint8 | 256 B | One UE8M0 scale per 32 K elements |

The 99 KiB number is not from the Blackwell marketing docs. I found it while browsing [CUTLASS issue #3144](https://github.com/NVIDIA/cutlass/issues/3144), where a contributor clarified that SM120 consumer Blackwell has 99 KiB of shared memory per SM, while SM100 datacenter Blackwell has 228 KiB. At 48.5 KB the kernel fits, but adding V later will push the budget close to the limit.

Still only one block per SM due to register pressure from the accumulators, but the shared memory budget is now accounted for.

### The loading pattern

The kernel uses 128 threads per block: 4 warps of 32 threads each, where each warp is responsible for 16 rows of the output tile. That gives exactly 4 times 16 = 64 rows, matching the Q tile size. The thread count is a direct consequence of the MMA tile geometry, not an arbitrary choice.

Those 128 threads need to load 8192 floats from VRAM into shared memory. The standard approach is a strided loop: each thread starts at its own index and jumps by 128 each iteration. Thread 0 loads elements 0, 128, 256, and so on. Thread 1 loads 1, 129, 257. This guarantees that on every iteration, consecutive threads read consecutive addresses, which is the definition of coalesced access. A non-coalesced load would serialize the memory transactions and cost significant bandwidth.

I initially wrote this with row and column indexing, computing `row = idx / Bd` and `col = idx % Bd` and then calculating the global address from there. It worked, but it was unnecessary complexity. Since Q is row-major and the tile is a contiguous block of rows, the linear index maps directly:

```c
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

```c
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

256 scaling blocks, 128 threads, 2 blocks per thread. Each thread processes its blocks sequentially, scanning for the max, computing the scale, dividing, and encoding all 32 values. It is not fast — there is a lot of branching in `encode_fp4_e2m1` and the inner loop is purely sequential — but it works. Optimization comes later.

K gets the exact same treatment: load into `staging` (overwriting Q's FP32 data, which is already quantized and safe in `Q_quant`), barrier, quantize into `K_quant` and `K_scales`, barrier.

### The MMA loop

With the data quantized in shared memory, the MMA itself follows the tiling from section 3. The outer loop covers 8 column tiles of S (64 columns / 8 per MMA), the inner loop accumulates 4 K-chunks along the head dimension (128 / 32 per MMA):

```c
for (int n_tile = 0; n_tile < N_TILES; n_tile++) {
    float acc[ACC_PER_THREAD] = {0.f};

    for (int k_tile = 0; k_tile < K_TILES; k_tile++) {
        // load fragments from Q_quant and K_quant
        // load scales from Q_scales and K_scales
        // call asm volatile MMA
    }
}
```

The accumulators are both input and output. On the first K-chunk they are zero. Each subsequent MMA adds its partial product. After 4 iterations, the accumulators hold the complete dot products for this thread's slice of the output tile. Four warps, 8 column tiles each, 4 accumulators per thread: the full [64 × 64] score matrix lives entirely in registers. No global memory is touched for the intermediate result.

The fragment loading — which bytes each lane loads from shared memory into its MMA registers — is the critical step. What I thought I understood at this stage turned out to be wrong in almost every detail. The register count, the lane grouping, the K stride, the scale index: all of it had to be corrected through empirical testing. The code I had at this point compiled and passed isolated tests with hardcoded values, which masked the errors.

---

## 8. Reverse-Engineering the Fragment Layout

*We are at step 3 of the pipeline: the first GEMM. The MMA is wired into the kernel, the data flows from global memory through shared memory to the Tensor Core. The output is garbage.*

I ran the full kernel against a float32 CPU reference. Cosine similarity: 0.06.

*Cosine similarity measures the angle between two output vectors. 1.0 means identical direction. 0.0 means completely uncorrelated. At 0.06, the kernel was producing noise.*

The building blocks were not the problem. The encoding still worked. The scaling still worked. The isolated MMA test still returned 256.0. The error had to be in the fragment loading: which bytes each of the 32 threads loads from shared memory into its MMA registers.

### No documentation

The fragment layout — the mapping between lane IDs and matrix positions — is defined by the hardware for every MMA instruction variant. For FP16 and BF16 MMA variants, NVIDIA documents these layouts with diagrams in the PTX ISA specification. For `mxf8f6f4 m16n8k32` on SM120, those diagrams do not exist. The instruction is listed, the operand counts are given, but the per-lane mapping is absent.

I posted a question on the [NVIDIA developer forums](https://forums.developer.nvidia.com/t/fragment-layout-for-mma-sync-aligned-m16n8k64-with-fp4-e2m1-and-block-scaling-on-sm120/364020). No response.

### The wrong diagnostic

I spent weeks trying different lane-to-position formulas against random input data. Each attempt compiled, ran, and produced a cosine between 0.01 and 0.09. With random data, a cosine of 0.03 and a cosine of 0.08 both mean wrong. There was no gradient to follow.

### The right diagnostic

I replaced the random test with a structured one. Q and K set to the 64×64 identity matrix, zero-padded to [64, 128]. The expected output is S = I × Iᵀ = I₆₄: ones on the diagonal, zeros everywhere else.

*When you multiply a matrix by its own transpose, the result at position [i][j] is the dot product of row i with row j. For the identity matrix, row i contains a single 1.0 at position i and zeros everywhere else. The dot product of row i with row j is 1.0 only when i equals j, and 0 in every other case. That is why the result is the identity matrix again.*

This test has two properties the random test does not. First, 1.0 is exactly representable in FP4 E2M1, so quantization cannot explain wrong results. Second, each non-zero in S comes from exactly one dot product. If S[2][5] is non-zero when it should be zero, it means the threads responsible for row 2 and column 5 loaded data they should not have loaded. The wrong value points directly to the wrong lane assignment.

I ran it: 20 non-zeros instead of 8. For the first time, I could see where the error was.

![Identity matrix test](identity_matrix_test.svg)

### Fixing the A fragment

The A tile is a [16 × 32] slice of Q: 16 rows, 32 elements along the head dimension. 512 values total, distributed across 32 threads. Each thread holds 4 registers of 4 FP4 elements: 32 × 4 × 4 = 512. The distribution is exact.

Two things were wrong.

First, my code loaded two registers per thread instead of four. Half the A tile had no owner. The hardware read uninitialized register contents for those positions and computed dot products against them. No error was raised.

Second, the lane grouping was wrong. My formula used `lane / 16`, which creates two groups of 16 threads. Both groups loaded the same K-column positions. Half the A tile was duplicated, the other half was never touched.

The correct formula is `lane / 4` for the row (8 distinct row assignments) and `lane % 4` for the K-column subgroup (4 distinct column ranges). Together: 8 × 4 = 32 unique assignments, no overlap, no gap. The four registers follow a specific pattern: a0 and a1 cover the same K-column range but rows 0 and 8 apart. a2 and a3 shift the K-columns by 16 and repeat the same row pattern.

After this fix: 8 non-zeros, correct diagonal positions for columns 0–7.

### Fixing the B fragment

With A correct, I looped over all 8 column tiles. The non-zero count went wrong again. Same class of error on the B side.

My code used `lane / 16` for the token index: two groups of 16 lanes, each duplicating the other. Only two distinct K tokens were ever loaded per tile. The correct formula mirrors A: `lane / 4` for the token (8 distinct assignments), `lane % 4` for the K-column subgroup. The two B registers cover the same token at K-column positions 16 apart.

After this fix: S = I₆₄ exactly. 64 non-zeros, all on the diagonal.

A second test with Q and K filled from {−2, −1, 0, 1, 2}, all exactly representable in FP4, gave cosine 1.000000 and max absolute error 0.000 across all 4096 entries. The fragment layout was correct.

The identity matrix test took an afternoon to build. The previous weeks of guessing produced nothing. Once I could see which cells in S were wrong, both fixes took less than an hour.

---

## 9. Integration: Six Bugs in the Full Kernel

*The fragment layout is correct on clean data loaded directly from global memory with scales hardcoded to 1.0. Now I move the corrected code into the full kernel that uses shared memory and on-the-fly quantization. The cosine drops back to 0.06.*

Six bugs, each masking the next. I list them in the order I found them, with the cosine after each fix, because the progression is the only way to tell which fixes actually mattered.

| # | Bug | Cause | Fix | Cosine |
|---|-----|-------|-----|--------|
| 1 | K stride | K indexed with `BQ` (64) instead of `Bd` (128) as row stride. Every K access landed at the wrong token. | Replace `BQ` with `Bd`. | 0.06 |
| 2 | K scale index | Scale lookup used `BQ / 32` instead of `Bd / 32` as stride. Wrong scale applied to every K block. | Replace `BQ` with `Bd`. | 0.06 |
| 3 | Scope errors | `q_row0` declared inside k_tile loop but needed after both loops. Compiler reused stale stack values. | Move declaration before both loops. | 0.06 |
| 4 | Lane collision | Output write used `lane % 16`. Lane 0 and lane 16 both wrote to row 0. Every row written twice. | Replace with `lane / 4`. | 0.19 |
| 5 | V accumulation | Butterfly reduce mixed output columns: thread 0's column-0 contribution added to thread 1's column-2. | Replace with explicit `__shfl_sync` per neighbor. | 0.81 |
| 6 | Race condition | Four blocks of 128 threads all writing to the same output. Last writer wins. | Launch one block. | 0.81 |

The cosine stalled at 0.06 for the first three fixes because each subsequent bug was still corrupting the output. Only fix 4, the lane collision, produced a visible improvement. Fix 5 jumped to 0.81. Fix 6 confirmed it.

The remaining gap from 1.0 is not a bug. It is quantization noise: FP4 E2M1 has eight representable magnitudes, and one scale covers 32 elements. The CPU reference operates on float32. Section 11 confirms this with exactly representable inputs at cosine 1.0000.

---

## 10. The Scale Layout

*The first GEMM now produces correct results. But one piece of the MMA is still empirical: how the scale factors are distributed across the 32 lanes of the warp.*

The MMA instruction takes one `uint32_t` per thread for `scale_a` and one for `scale_b`. But the A tile has 16 rows and each row needs its own scale. The B tile has 8 columns and each column needs its own scale. One register per thread cannot hold all of that.

The hardware distributes the scale responsibility across lanes, just as it distributes the fragment data. But the PTX ISA does not document this mapping for SM120. Same situation as the fragment layout. Same approach.

### The probing method

All inputs set to 1.0. All scales set to 127 (2⁰ = 1.0). Every MMA output is 32.0.

Then 32 runs. Each run sets exactly one lane's `scale_a` to 128 (2¹ = 2.0), all others at 127. If lane L's scale controls row R, then row R doubles from 32.0 to 64.0. Record which rows change for each lane.

### The result

| Lane condition | Row affected |
|---|---|
| `lane % 4 == 0` | `lane / 4`, rows 0–7 |
| `lane % 4 == 1` | `lane / 4 + 8`, rows 8–15 |
| `lane % 4 == 2` | no effect |
| `lane % 4 == 3` | no effect |

Same probing on `scale_b`: only lanes where `lane % 4 == 0` have an effect, and lane L controls column `lane / 4`.

This is consistent with the A fragment structure from section 8. Register a0 covers row0 and its scale is read from the lane with `lane % 4 == 0`. Register a1 covers row0+8 and its scale comes from `lane % 4 == 1`. Lanes 2 and 3 carry fragment data but the hardware ignores their scale values.

Neither the fragment layout nor the scale distribution for FP4 E2M1 `m16n8k32` on SM120 is documented anywhere in the PTX ISA. Both were determined empirically with the probing methods described in this section and section 8.

---

## 11. Online Softmax and V Accumulation

*The first GEMM produces correct scores in registers. The scale layout is known. We move to step 4 of the pipeline: softmax, then step 5: the second GEMM.*

### Why a warp reduction is unavoidable

Writing S to global memory, running softmax separately, and reading it back would defeat the entire purpose of the fused kernel. Instead, I used the online softmax algorithm from the FlashAttention paper. The idea is to maintain a running state that updates as each column tile of S is computed, so the softmax normalization is applied incrementally without ever materializing the full score matrix.

The MMA output layout places a row's scores across four threads. The eight scores for a complete row are split: thread 0 holds columns 0 and 1, thread 1 holds columns 2 and 3, thread 2 holds columns 4 and 5, thread 3 holds columns 6 and 7. To compute the row maximum needed for numerically stable softmax, those four threads must communicate.

*A butterfly reduce is a communication pattern where threads exchange values in log2(N) rounds, such that after the rounds every thread holds the result of the operation across all N threads.* Running `__shfl_xor_sync` twice with masks 1 and 2 covers all four pairings in two rounds. This warp reduction is not a choice. It is a direct consequence of the fragment layout.

### The online state

The running state for each row has three components: `m`, the maximum score seen so far; `l`, the sum of exponentials seen so far; and `O`, the unnormalized output accumulated so far. When a new tile arrives:

new_m = max(m, tile_max)
alpha = exp(m - new_m)
new_l = alpha * l + sum(exp(score - new_m) for score in tile)
new_O = (alpha * l * O + weighted V contribution) / new_l

`alpha` is the rescaling factor. If the new tile contains a score larger than anything seen before, `alpha` is less than 1 and the previous accumulator shrinks proportionally. If the maximum does not change, `alpha` is 1 and the old output is unchanged.

### The V accumulation

Each thread uses `__shfl_sync` to fetch the softmax weights from its three neighbors explicitly, then multiplies each neighbor's weights against the V values for its own output columns. The accumulation stays local to each thread's assigned dimensions. No cross-dimension mixing.

### Validation

With all fixes in place, the full pipeline from Q and K loading through softmax through V accumulation produced:

```
cosine similarity : 1.0000 PASS
ref[0..7] : -0.446 -0.879 -0.450  0.511  0.940  0.968 -0.947 -0.049
out[0..7] : -0.446 -0.879 -0.450  0.511  0.940  0.968 -0.947 -0.049
```

Bit-exact. This confirms that the entire pipeline is correct when given inputs that are exactly representable in FP4 E2M1.

The 0.81 cosine observed earlier with random inputs in [-1, 1] is the intrinsic precision cost of MXFP4 at `scale_vec::1X` granularity. FP4 E2M1 has only eight representable magnitudes. With one scale covering 32 elements, a single outlier sets the scale for the entire block and the remaining values lose resolution. The CPU reference operates on the original float32 values, so the comparison is unfair. The kernel is correct. The 0.81 is an architectural constraint, not a bug.

---

## 12. Completing the Kernel

*The fused pipeline works for a single K tile of 64 tokens, a single head, and a fixed head dimension of 128. Three things remain before the kernel is usable on real models.*

### The K tile loop

The kernel validated in section 11 only ever saw the first 64 tokens of the key sequence. For any real attention computation, K can have thousands of tokens.

The change is conceptually simple. Instead of loading one K tile before the MMA loop, the outer structure becomes a loop over sequence tiles. For each tile, the kernel loads 64 rows of K into shared memory, quantizes them, runs the full MMA and softmax update, then moves to the next tile. The online softmax state — `m`, `l`, and `O` — is declared before the loop and persists across all tiles.

One detail worth noting: the V access index must account for the tile offset. When accumulating the attention output, the V row index is `seq_tile * BQ + local_token`, not just the local token position. Without that offset, every tile reads from the beginning of V.

Validation: `seq_k = 64` (single tile, regression): cosine 1.0000. `seq_k = 128` (two tiles): cosine 1.0000.

### Softmax scaling

The attention formula is softmax(Q×Kᵀ / sqrt(d)) × V. The division by sqrt(d) was missing until this point.

Without it, the scores grow with the head dimension. Each score is a dot product of two vectors of length d. If Q and K have values around 1, the scores are on the order of sqrt(d) — for d=128, that is around 11. Feeding large values into softmax pushes it toward saturation: the maximum score gets a weight close to 1 and everything else collapses toward 0. The attention output becomes a near-copy of one V row.

Dividing by sqrt(d) brings the scores back to order of magnitude 1 before the softmax, keeping the output distribution balanced. In the kernel, this is a single multiply applied to the accumulators after the k_tile loop and before the softmax reduction:

```c
for (int i = 0; i < ACC_PER_THREAD; i++)
    acc[i] *= softmax_scale;  // softmax_scale = 1/sqrt(Bd)
```

Both test cases pass at cosine 1.0000.

### Multi-head attention and arbitrary head dimensions

Different models use different head dimensions. GPT-2 uses 64, LLaMA uses 128, some recent models use 256. Hardcoding 128 excludes most of them.

The solution is a C++ template parameter: `template<int HEAD_DIM>`. The compiler generates a separate binary for each instantiation. No runtime branching, no overhead. The only constraint is that `HEAD_DIM` must be a multiple of 32, the MMA reduction dimension.

Fixing the head dimension revealed a deeper bug. The original kernel kept two scalar accumulators for the V output. For `HEAD_DIM=128`, each thread is responsible for 128 / 4 = 32 output column pairs, not 2. The previous kernel was writing 2 values and leaving 126 columns at zero. The fix replaces the scalars with an array `O0[V_COL_TILES * 2]` where `V_COL_TILES = HEAD_DIM / MMA_N`.

Each block processes one `(batch, head)` pair independently:

```c
int batch_idx = blockIdx.x / heads;
int head_idx  = blockIdx.x % heads;
```

Launch: `<<<batch * heads, NUM_THREADS>>>`.

Six test cases confirm correctness:

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

## 13. Profiling: Where the Time Goes

*The kernel is functionally complete. Every test passes. Now we measure.*

### First benchmark

Configuration: RTX 5070 Ti, batch=1, heads=32, seq_q=64.

| head_dim | seq_k | kern_ms | TFLOPS | BW GB/s |
| --- | --- | --- | --- | --- |
| 128 | 128 | 0.072 | 1.87 | 87.8 |
| 128 | 512 | 0.255 | 2.11 | 74.0 |
| 128 | 1024 | 0.530 | 2.03 | 67.3 |
| 64 | 128 | 0.037 | 1.82 | 85.2 |
| 64 | 512 | 0.102 | 2.62 | 92.1 |
| 64 | 1024 | 0.192 | 2.80 | 92.9 |

The RTX 5070 Ti has a theoretical FP4 throughput of 474 TFLOPS. We are at 2.8, which is 0.6% utilization.

### What NCU revealed

The first metric that jumped out was "No Eligible" at 95.23%. The warp scheduler found no warp ready to execute 95% of the time.

*A warp scheduler looks for eligible warps — warps that have their input data ready and can execute an instruction. When none are eligible, the SM is idle. High "No Eligible" is the definition of a latency-bound kernel.*

The occupancy was 7.94% against a theoretical maximum of 8.33%. Shared memory was the binding constraint: 2 blocks per SM, 4 active warps. A GPU needs roughly 32 warps per SM to fully hide memory latencies. The long scoreboard stall was at 81%, caused by V being read directly from global memory inside a double loop.

### Fix 1: V in shared memory

Load each V tile into shared memory before the MMA loop, exactly as K is loaded. This adds 32 KB to the shared memory budget, bringing the total to about 80 KB. TFLOPS roughly doubled.

### Fix 2: FP16 staging

The staging buffer was the largest consumer at 32 KB. Switching from float32 to `__half` dropped it to 16 KB. The same buffer gets reused for V. The precision trade-off is negligible: FP16 has 10 bits of mantissa, FP4 E2M1 has 1. Any rounding disappears in the quantization step. All tests still pass at cosine 1.0000.

After these two fixes: 3.1 TFLOPS on head_dim=64, seq_k=1024. Still 5x slower than PyTorch SDPA (15 TFLOPS in FP16 on the same hardware).

### What the SASS revealed

I ran `ncu --set full` and exported the source-level report. The kernel compiled to about 5,900 SASS instructions. Four of them were QMMA, the FP4 Tensor Core multiply. The other 5,896 were overhead.

**The division that was not a division.** 129 calls to `__cuda_sm3x_div_rn_noftz_f32_slowpath`. The GPU does not have a hardware division unit. `MUFU.RCP` computes the reciprocal in one cycle, but nvcc generates a software refinement routine for IEEE-754 precision — a full function call per division. In my kernel, the division happened in `compute_scale_ue8m0` to normalize values before encoding. The scale is a power of two, so the division is exact. IEEE-754 precision on a result that will be rounded to one of eight magnitudes.

Fix: replace `val / scale_f` with `val * exp2f((float)(127 - scale))`. The 129 CALL instructions disappeared. The kernel dropped from 5,900 to 4,200 SASS instructions, a 28% reduction.

**The 647 comparisons for a 32-element max.** `compute_scale_ue8m0` finds the maximum using `if (a > max_abs) max_abs = a`. The compiler generated `FSETP.GT` + `FSEL` (two instructions) instead of `FMNMX` (one instruction). Replacing with `max_abs = fmaxf(fabsf(block[i]), max_abs)` gives nvcc the hint. The `fmaxf` intrinsic maps directly to `FMNMX`.

**The byte-by-byte problem.** The quantization pipeline writes each FP4 byte to shared memory individually with `STS.U8` (66 of them) and reads them back with `LDS.U8` (104 of them). Each 8-bit access consumes a full 32-byte shared memory transaction. Packing four bytes into `uint32` before the store would use the same bandwidth for 4x the data. L1 Wavefronts Shared Excessive: 14,336.

### The full picture

After the division fix and the fmaxf change:

| Category | Instructions | % of total |
| --- | --- | --- |
| FP4 quantization (encode + scale) | ~2,800 | 66% |
| Data movement (LDG, STS, LDS, STG) | ~800 | 19% |
| Softmax + V accumulation | ~400 | 10% |
| QMMA (Tensor Core compute) | 4 | 0.1% |
| Other (control, sync, address math) | ~200 | 5% |

The Tensor Cores executed four instructions out of 4,200. Everything else was preparation. The kernel is not compute-bound. It is quantization-bound.

---

## 14. Why the Gap Is Expected

PyTorch SDPA with FlashAttention reaches 15 to 16 TFLOPS on the RTX 5070 Ti for the same problem size. This kernel reaches 2.4 to 3.4 TFLOPS, roughly 4 to 5 times slower.

The gap is not a bug. It is a design consequence.

FlashAttention receives Q, K, and V already in FP16. The kernel's main loop is almost entirely MMA instructions and softmax arithmetic. There is no format conversion inside the hot path.

This kernel receives Q, K, and V in float32. For every tile of 8,192 elements, it computes 256 block scales, finds the absolute maximum of each block, converts each value to the nearest FP4 representation through a chain of eight comparisons, packs the results into shared memory, and only then feeds them to the Tensor Core. That quantization pass runs twice per sequence tile, once for Q and once for K.

The quantization is doing useful work. It is not wasted computation. But it is scalar work on data that the Tensor Core will process in a single instruction. The ratio between the two is the gap.

For a production inference kernel, the solution is to move the quantization outside. In a decode loop, K and V live in a KV cache that is already quantized to FP4. Q is a single token that can be quantized in a separate, lightweight kernel. The attention kernel itself receives pre-packed `uint8` inputs and spends its time on MMA and softmax. That is what SageAttention3 does, and it is the natural next step for this project.

But the current kernel was never designed to compete on throughput. It was designed to make every step of the FP4 fused attention pipeline visible: the MMA fragment layout that is not documented, the container format that silently reads the wrong value if you shift by one bit, the scale distribution across lanes that required 32 probing runs to reverse-engineer, the division operator that turns into 129 function calls. None of that is visible in a CUTLASS template. Writing it from scratch with inline PTX was the only way to see it.

---

## 15. What I Would Do Differently

Looking back at several months of work, a few things stand out.

**Test with structured inputs first.** The weeks I spent guessing the fragment layout produced nothing because I was testing against random data. The identity matrix test from section 8 gave me precise, per-cell information about which lane loaded which position. Both the A and B fragment fixes took less than an hour once the right test existed. Every new MMA instruction variant should be validated with identity matrices before anything else.

**Read the SASS earlier.** The division slowpath was invisible at the C++ level. The scale computation looked like a single line of code. It took NCU and the SASS source view to reveal that one line was generating 129 function calls. Profiling should not be the last step. It should happen after every major code change.

**Do not optimize the wrong design.** The on-the-fly quantization was never going to be fast. I knew this conceptually from the start, but I kept optimizing around it (vectorized loads, FP16 staging, shared memory reuse) instead of changing the fundamental approach. The optimization that would have mattered most — pre-quantized inputs — was the one I deferred the longest.

**The fragment layout is the real contribution.** The MMA m16n8k32 fragment layout for FP4 E2M1 on SM120 is not documented anywhere in the PTX ISA. The scale distribution across lanes is not documented. The container format (nibble in bits 5-2, not bits 3-0) is mentioned in one sentence in the spec but never shown in a worked example. Figuring this out empirically and publishing it is the part of this project that will be useful to other people writing SM120 kernels. The kernel performance is secondary.

**The ecosystem is catching up.** When I started this project, SM120 support in the open-source stack was minimal. SageAttention3 was 5090-only in practice, FlashInfer had no SM120 path, and vLLM fell back to Marlin for FP4 on consumer Blackwell. By the time I am writing this, all three have added or are adding SM120 support. The gap I set out to fill is closing, which is a good thing. The documentation gap remains open.

*Code: [github.com/florianmattana/fp4-fused-attention-sm120](https://github.com/florianmattana/fp4-fused-attention-sm120)*
