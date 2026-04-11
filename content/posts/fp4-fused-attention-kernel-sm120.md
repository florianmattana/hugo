---
title: "Building an FP4 Fused Attention Kernel on Consumer Blackwell (SM120) (WIP)"
date: 2026-03-17
draft: false
tags: ["cuda", "blackwell", "sm120", "fp4", "attention", "tensor-cores"]
description: "A deep dive into writing a fused FP4 attention kernel for the RTX 5070 Ti, using inline PTX and warp-level MMA instructions."
showToc: true
TocOpen: true
---

I have been trying to write a fused FP4 attention kernel that runs on consumer Blackwell GPUs and specifically the RTX 5070 Ti. This post documents the full journey: every wrong turn, every hardware surprise, and every trade-off I had to make along the way.

## 1. Why FP4 Fused Attention on Consumer Blackwell?

The attention mechanism in transformers scales quadratically with sequence length. On a consumer GPU with 12 GB of VRAM and 672 GB/s of memory bandwidth, that becomes a hard wall very quickly. The interesting thing about the RTX 5070 Ti (SM120, 46 SMs) is the raw throughput the Tensor Cores can deliver:

| Precision | Throughput |
|-----------|-----------|
| FP16 | 123.5 TFLOPS |
| INT8 | 246.9 TFLOPS |
| FP4 | ~474 TFLOPS |

That is roughly a 4x advantage going from FP16 to FP4, and since FP4 values are four times smaller, you also move four times less data through memory. On paper, that is a massive win for attention. If you can actually use the FP4 Tensor Cores.

The problem is that nobody has done this yet on consumer Blackwell. Existing fused attention kernels like SageAttention3 and FlashAttention-4 target SM100 (datacenter Blackwell). They use instructions and hardware features (`tcgen05.mma`, Tensor Memory) that simply do not exist on SM120. If you try to compile them for `sm_120`, they either crash or fail silently.

There are non-fused FP4 kernels out there for this UC. For example [VincentKaufmann fp4-cuda-kernel](https://github.com/VincentKaufwormann/fp4-cuda-kernel) reaches about 143 TFLOPS. But non-fused means you compute QxK, write the full NxN score matrix to VRAM, read it back, apply softmax, write again, then compute the attention output. For 4096 tokens, that score matrix alone is 64 MB. On a 12 GB card, that is a dealbreaker.

The whole point of a *fused* kernel is to keep the intermediate score matrix in registers and never write it to global memory. That is what FlashAttention does for FP16 and I wanted to do the same thing for FP4.

## 2. Choosing the Programming Model

I considered three approaches:

**Option A -- Inline PTX.** Write the kernel in CUDA C++ and embed the Tensor Core MMA instructions as inline PTX assembly. This gives full control over register allocation, meaning I can guarantee the score matrix stays in registers.

**Option B -- CuTe (CUTLASS 3.x).** Use NVIDIA template library. CuTe is powerful, but it abstracts away register placement. I was not confident I could prevent it from spilling the score matrix to shared or global memory, especially for a non-standard fused pattern.

**Option C -- Patch an existing INT8 kernel.** Take a working fused INT8 attention kernel and swap the MMA instructions for FP4 equivalents. Faster to prototype, but brittle, the register layouts differ between INT8 and FP4 MMA, so the whole data flow would need reworking anyway.

I went with **Option A**. The trade-off is clear: more manual work, more room for bugs, but absolute certainty about where every value lives. For a fused kernel where the entire point is keeping data in registers, that certainty is worth it.

## 3. Picking the Right MMA Instruction

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

The register budget for one MMA call is roughly 7 registers per thread: 2 for the A fragment, 1 for B, and 4 for the FP32 accumulator. With my tile size of BQ=BK=64, I need about 50 registers per thread for the accumulators alone, which gives me around 85% SM occupancy. Going larger would blow the register file and kill occupancy. That is the binding constraint on this architecture.

The fused kernel will process tiles of Q, K, V through shared memory (~9 KB out of the 128 KB available), while the score accumulators live permanently in registers. That is the core idea: shared memory for inputs, registers for intermediates, never touch global memory for the NxN attention matrix.

## 4. Testing the MMA Instruction (and Everything That Went Wrong)

Before building the full fused kernel, I needed to verify that a single FP4 MMA instruction actually works. The idea is simple: load known values into registers A and B, run the MMA, and check that the FP32 accumulators contain the expected result.

I wrote a minimal warp-synchronous kernel, launched with `<<<1, 32>>>` so that exactly one warp executes. The kernel fills the A and B registers with constant FP4 values, calls the MMA via inline PTX, and prints the four accumulator floats from thread 0.

### The tile shape surprise

My first attempt used `m16n8k64` and I reasoned that since FP4 values are 4 bits each, 64 of them would fit in 32 bytes (the same as 32 FP8 values). The PTX assembler disagreed. It turns out the correct shape for FP4 on SM120 is **m16n8k32**: the k-dimension counts 8-bit *containers*, not individual FP4 values. Each container holds one FP4 nibble in bits 5-2, padded with zeros. This means you are effectively wasting half the container, but that is what the hardware expects.

### The encoding bug that cost me a full day

FP4 E2M1 encodes the value 1.0 as the 4-bit pattern `0b1000`. Since this nibble must sit in bits 5-2 of the 8-bit container, the correct byte for 1.0 is `0x08` not `0x02`, which is what you would get if you just placed the nibble in bits 3-0.

I initially filled every register with `0x22222222`, thinking I was encoding 2.0 in each position. The MMA gave me 2.0 in the accumulators instead of the 32.0 I expected. After staring at bit layouts for longer than I would like to admit, I realized the nibble was in the wrong position. Switching to `0x08080808` (1.0 in the correct bit position) and expecting 32.0. That worked.

The lesson: the FP4 container format is `00_SEEM_00` (sign, exponent, exponent, mantissa in bits 5-2). Get the shift wrong and the hardware silently interprets garbage.

### The inline PTX

Here is the actual `asm volatile` block that calls the MMA:

```cuda
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

The `"=f"` constraints are FP32 output registers, `"r"` are 32-bit integer input registers. The accumulator C is passed through as input (initialized to zero for the first call), and the result lands in D. The scale registers each pack four UE8M0 bytes into a single `uint32`.

With `a0 = a1 = 0x08080808` (all 1.0), `b0 = 0x08080808` (all 1.0), and scales set to 1.0 (`0x7F7F7F7F`, since UE8M0 byte 127 = 2^0 = 1.0), the result was 32.0 in every accumulator lane. That is 32 multiply-accumulates of 1.0 x 1.0, which is exactly right.

## 5. Encoding FP32 to FP4 E2M1

Once the MMA worked with hardcoded constants, the next step was encoding arbitrary `float` values into FP4 E2M1 at runtime.

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

## 6. Block Scaling: Why the Encoding Function Is Not Enough

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

The kernel now has three validated building blocks: `encode_fp4_e2m1`, `compute_scale_ue8m0`, and the inline PTX MMA call. The next step is loading Q, K, V tiles into shared memory with async copies (`cp.async`) and wiring everything together into the fused attention loop.

## 7. Assembling the Kernel: Where Things Got Real

The three building blocks worked in isolation. Encoding, scaling, MMA, all validated with hardcoded test values. Now I had to wire them together into an actual kernel that loads real data from VRAM, quantizes it on the fly, and runs the Tensor Core multiply. This is where the gap between "I have working pieces" and "I have a working kernel" became very real.

### The first decision: how much shared memory

My first attempt allocated two separate FP32 buffers in shared memory, one for Q and one for K. Each tile is 64 tokens times 128 dimensions = 8192 floats = 32 KB. Two of them: 64 KB. Add the quantized buffers and scales, and I was over 80 KB. That works on paper since the SM has 128 KB, but it means only one thread block per SM, and I had not even started thinking about the MMA accumulators eating into the register file.

I realized Q and K are never needed in FP32 at the same time. Load Q as FP32, quantize it, then reuse the same buffer for K. One staging buffer instead of two. That brought shared memory down to about 50 KB:

| Buffer | Type | Size | Purpose |
| --- | --- | --- | --- |
| `staging` | float | 32,768 B | Reusable FP32 buffer for loading from VRAM |
| `Q_quant` | uint8 | 8,192 B | Q tile after FP4 quantization |
| `K_quant` | uint8 | 8,192 B | K tile after FP4 quantization |
| `Q_scales` | uint8 | 256 B | One UE8M0 scale per 32 Q elements |
| `K_scales` | uint8 | 256 B | One UE8M0 scale per 32 K elements |

Still only one block per SM due to the register pressure from the accumulators, but at least I was not wasting shared memory.

### The loading pattern that almost tripped me up

128 threads need to load 8192 floats from VRAM into shared memory. The standard approach is a strided loop: each thread starts at its own index and jumps by 128 each iteration. Thread 0 loads elements 0, 128, 256, and so on. Thread 1 loads 1, 129, 257. This guarantees that on every iteration, consecutive threads read consecutive addresses, which is the definition of coalesced access.

I initially wrote this with row/column indexing, computing `row = idx / Bd` and `col = idx % Bd` and then calculating the global address from there. It worked, but it was unnecessary complexity. Since Q is row-major and the tile is a contiguous block of rows, the linear index maps directly:

```cuda
for (int k = 0; k < TILE_SIZE; k += NUM_THREADS) { int idx = tid + k; int g_idx = blockIdx.x * TILE_SIZE + idx; staging[idx] = Q[g_idx]; } __syncthreads();
```

`blockIdx.x * TILE_SIZE` is the offset for this block's group of 64 tokens. No division, no modulo. I kept the row/column version in an early commit before realizing the simplification. Sometimes the clever approach is the dumb one.

### The quantization two-pass problem

I wanted to quantize the floats as I loaded them from VRAM, avoiding the staging buffer entirely. But block scaling killed that idea. To compute the scale factor for a block of 32 elements, you need the maximum absolute value across all 32. During the strided load, a single thread does not see all 32 elements of the same scaling block because they are spread across multiple iterations. Thread 0 sees elements 0, 128, 256, never elements 1 through 31.

So quantization has to be a separate pass after loading. The staging buffer exists specifically because of this dependency. First load everything as FP32, barrier, then quantize in a second pass where each thread handles complete 32-element blocks:

```cuda
for (int i = tid; i < NUM_SCALE_BLOCKS; i += NUM_THREADS) { uint8_t scale = compute_scale_ue8m0(&staging[i * BLOCK_ELEMENT]); Q_scales[i] = scale; float scale_f = exp2f((float)(scale - 127));

for (int j = 0; j < BLOCK_ELEMENT; j++)
{
    float val = staging[i * BLOCK_ELEMENT + j] / scale_f;
    Q_quant[i * BLOCK_ELEMENT + j] = encode_fp4_e2m1(val);
}
} __syncthreads();
```

256 scaling blocks, 128 threads, 2 blocks per thread. Each thread processes its blocks sequentially, scanning for the max, computing the scale, dividing, and encoding all 32 values. It is not fast, there is a lot of branching in `encode_fp4_e2m1` and the inner loop is purely sequential, but it works. Optimization comes later.

K gets the exact same treatment: load into `staging` (overwriting Q's FP32 data, which is already quantized and safe in `Q_quant`), barrier, quantize into `K_quant` and `K_scales`, barrier.

### The MMA fragment loading: the part I could not figure out alone

This is where I spent the most time. The MMA instruction `mma.sync.m16n8k32` expects each of the 32 threads in a warp to hold specific bytes from the A and B matrices in its registers. Not just any bytes, the exact bytes that the hardware expects for that lane. The mapping is defined in the PTX ISA fragment layout tables, and it is not intuitive.

For fragment A (a 16x32 slice of Q), each thread holds two 32-bit registers: `a0` and `a1`. Each register packs 4 FP4 values in their 8-bit containers. The lane ID determines which row the thread reads from (`lane % 16` gives a row from 0 to 15, meaning lanes 0-15 and lanes 16-31 cover the same rows but different column groups). The two registers `a0` and `a1` are not contiguous in memory: they are separated by a stride of 16 columns.

I first tried loading them contiguously, thinking `a0` covers columns 0-3 and `a1` covers columns 4-7. The MMA gave garbage. After going back to the ISA tables, I found that `a0` and `a1` are 16 columns apart, not 4. The columns in between are held by other threads in the warp.

The working version:

```cuda
int q_row = warp_id * MMA_M + (lane % 16); int q_col = k_tile * MMA_K + (lane / 16) * BYTES_PER_REG;
int q_idx_0 = q_row * Bd + q_col; int q_idx_1 = q_row * Bd + q_col + MMA_A_STRIDE;
uint32_t a0 = Q_quant[q_idx_0] | (Q_quant[q_idx_0 + 1] << 8) | (Q_quant[q_idx_0 + 2] << 16) | (Q_quant[q_idx_0 + 3] << 24);
uint32_t a1 = Q_quant[q_idx_1] | (Q_quant[q_idx_1 + 1] << 8) | (Q_quant[q_idx_1 + 2] << 16) | (Q_quant[q_idx_1 + 3] << 24);
```

`MMA_A_STRIDE = 16` is the gap between the two registers. `BYTES_PER_REG = 4` is how many 8-bit containers fit in one 32-bit register. These are hardware constants that I only discovered through trial and error and reading the ISA diagrams three times.

Fragment B (a 32x8 slice of K) is simpler because each thread only loads one register:

```cuda
int k_row = k_tile * MMA_K + (lane % 16); int k_col = n_tile * MMA_N + (lane / 16) * BYTES_PER_REG;
int k_idx_0 = k_row * BQ + k_col;
uint32_t b0 = K_quant[k_idx_0] | (K_quant[k_idx_0 + 1] << 8) | (K_quant[k_idx_0 + 2] << 16) | (K_quant[k_idx_0 + 3] << 24);
```

The scale factors slot in naturally. Each thread looks up the scale for the scaling block that contains its loaded values and duplicates it four times across a `uint32_t`:

```cuda
uint8_t sa = Q_scales[q_row * (Bd / BLOCK_ELEMENT) + k_tile]; uint8_t sb = K_scales[k_row * (BQ / BLOCK_ELEMENT) + (n_tile * MMA_N) / BLOCK_ELEMENT]; uint32_t scale_a = sa | (sa << 8) | (sa << 16) | (sa << 24); uint32_t scale_b = sb | (sb << 8) | (sb << 16) | (sb << 24);
```

### The MMA loop

With fragments loaded, the MMA itself is anticlimactic. It is the same `asm volatile` block from section 4, just wired into a loop. The outer loop covers 8 column fragments of S, the inner loop accumulates 4 k-chunks:

```cuda
for (int n_tile = 0; n_tile < N_TILES; n_tile++) { float acc[ACC_PER_THREAD] = {0.f};
for (int k_tile = 0; k_tile < K_TILES; k_tile++)
{
    // ... load a0, a1, b0, scale_a, scale_b ...

    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4"
        ".block_scale.scale_vec::1X.f32.e2m1.e2m1.f32.ue8m0"
        " {%0, %1, %2, %3},"
        " {%4, %5},"
        " {%6},"
        " {%7, %8, %9, %10},"
        " %11,"
        " %12;"
        : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
        : "r"(a0), "r"(a1),
          "r"(b0),
          "f"(acc[0]), "f"(acc[1]), "f"(acc[2]), "f"(acc[3]),
          "r"(scale_a), "r"(scale_b)
    );
}
}
```

The accumulators are both input and output. On the first k-chunk they are zero. Each subsequent MMA adds its partial product. After 4 iterations, `acc[0]` through `acc[3]` hold the final values for this thread's piece of the (16, 8) output fragment. Four warps, 8 column tiles each, 4 accumulators per thread: the full (64, 64) score matrix lives entirely in registers. No global memory was touched for the intermediate result. That was the whole point.

## 8. The First Correctness Test

### The result

Section 7 ended with the MMA loop assembled and the score matrix living in registers. I ran the correctness test against a float32 CPU reference.

Cosine similarity: 0.06.

*Cosine similarity measures the angle between two output vectors. A value of 1.0 means the GPU and CPU outputs point in exactly the same direction. A value of 0 means they are completely uncorrelated. At 0.06, the GPU was producing noise.*

### What the number does not tell you

The three validated components were not the problem. The encoding function worked. The block scaling worked. The isolated MMA test from section 4 still printed 256.0 correctly. The problem had to be in how the full kernel assembled those pieces together.

The most obvious candidate was the fragment loading. In section 7, I described loading `a0` and `a1` for the A fragment and `b0` for B. That description was my best understanding at the time. It was wrong in two ways I had not yet discovered.

The issue with cosine as a diagnostic is that it gives you one number. It tells you whether the final answer is right or wrong. It tells you nothing about where it went wrong.

## 9. The Fragment Layout Problem

### No documentation

To understand what went wrong, it helps to understand what a fragment actually is. When the GPU executes an MMA instruction, the 32 threads in a warp collectively own a tile of the A matrix, a tile of B, and the accumulator tile D. Each individual thread holds a specific slice of that tile in its registers. That slice is the thread's fragment. The hardware defines precisely which matrix positions belong to which thread, a mapping called the fragment layout.

For FP16 and BF16 MMA variants, NVIDIA documents these layouts with diagrams in the PTX ISA specification. For `mxf8f6f4 m16n8k32` on SM120, those diagrams do not exist. The instruction is listed, the operand counts are given, but the per-lane mapping is absent.

I posted a question on the [NVIDIA developer forums](https://forums.developer.nvidia.com/t/fragment-layout-for-mma-sync-aligned-m16n8k64-with-fp4-e2m1-and-block-scaling-on-sm120/364020) on March 19. No response.

### The guessing phase

So I started guessing. The PTX ISA gives you the operand count for the instruction: 4 registers for A, 2 for B, 4 for the accumulator. But it does not tell you which matrix row or column each register corresponds to for a given lane. That mapping is what I was guessing.

*A lane is a thread within a warp, numbered 0 to 31. For a 16x32 tile of Q spread across 32 threads, each lane owns a specific slice. The question is: which slice?*

I tried `lane % 16` for the row index and `lane / 16` for the K-column group. I tried different strides between registers: 4, 8, 16 columns apart. Each attempt compiled without errors and produced a different cosine between 0.01 and 0.09. A cosine of 0.01 and a cosine of 0.09 both mean wrong. There was no gradient to follow. I could not tell if I was getting closer to the correct layout or further away, because the only signal I had was a single float that told me nothing about where the error was.

The problem was not the guesses. The problem was the test.

## 10. Finding the Right Diagnostic

### The identity matrix test

I replaced the attention test with a simpler one designed to give precise information. I set Q and K to the 64x64 identity matrix, zero-padded to the [64, 128] shape the kernel expects. The expected output is S = Q times K-transpose = I_64. Every diagonal entry is 1.0, every off-diagonal entry is 0.

*When you multiply a matrix by its own transpose, the result at position [i][j] is the dot product of row i with row j. For the identity matrix, row i contains a single 1.0 at position i and zeros everywhere else. The dot product of row i with row j is 1.0 only when i equals j, and 0 in every other case. That is why the result is the identity matrix again.*

I also isolated the fragment loading from everything else by writing a separate debug kernel that loads directly from global memory, skips shared memory entirely, and hardcodes all scales to 1.0. One variable at a time.

Running this with my best guess at the fragment layout: 20 non-zeros instead of 8. Columns 0 through 7 of the identity matrix should produce exactly 8 ones on the diagonal. There were 12 phantom values in wrong positions.

<img src="/identity_matrix_test.svg" alt="Identity matrix test" style="width:100%;max-width:680px;">

### What it reveals

This test has two properties the attention test does not. First, 1.0 is exactly representable in FP4 E2M1, so quantization cannot explain wrong results. Second, each non-zero in S comes from exactly one dot product. If S[2][5] is non-zero when it should be zero, it means the threads responsible for row 2 of Q and column 5 of K loaded data they were not supposed to load. The wrong value points directly to the wrong lane.

For the first time, I could see where the problem was.

## 11. Fixing the A Fragment

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

Running the identity matrix test after this fix: 8 non-zeros, on the diagonal at the correct positions for columns 0 through 7.

After this fix: 8 non-zeros, correct diagonal positions for columns 0 through 7.

## 12. Fixing the B Fragment

### The same class of error

Adding the loop over all 8 column tiles to cover the full 64x64 matrix revealed the same problem in the K loading. The original code used `lane % 16` for the head-dimension index and `lane / 16` for the token index. Same result: only two distinct token groups, half the lanes loading duplicate data.

The fix mirrors the A correction. `lane / 4` selects the token within the column tile, `lane % 4` selects the head-dimension subgroup. The two B registers cover the same token at K-column positions that are 16 apart.

### Confirmation

After this fix: 64 non-zeros, S equals I_64 exactly.

I confirmed with a second test using Q and K filled from the set {-2, -1, 0, 1, 2}. These values are all exactly representable in FP4 E2M1. Cosine similarity: 1.000000. Max absolute error: 0.000. The fragment loading was correct.

The identity matrix test took less than an afternoon to implement and confirmed both fixes in one run. The previous weeks of guessing produced nothing because I was testing the wrong thing.

## 13. The Remaining Bugs

With the fragment layout validated on clean data, I moved the corrected loading code into the full kernel that uses shared memory and quantization. The cosine dropped back to 0.06. A different set of problems, same symptom.

### The K stride

K is stored row-major in shared memory as `K_quant[token * Bd + head_dim]`, where `Bd` is 128. My indexing used `token * BQ + head_dim`, treating `BQ` (64) as the row stride instead of `Bd` (128). Every K access was reading from roughly half the correct address. The kernel was computing dot products between Q rows and completely wrong K bytes.

Fix: replace `BQ` with `Bd` in the K stride.

Cosine after fix: still 0.06. The next bug was masking any improvement.

### The K scale index

The scale array for K is indexed by token first, then by k_tile. The stride between tokens is `Bd / BLOCK_ELEMENT`, which is 4. My code was using `BQ / BLOCK_ELEMENT` instead, which is 2. Every scale lookup was hitting the wrong position in `K_scales`, applying the wrong normalization factor to every K block.

Fix: replace `BQ` with `Bd` in the scale stride.

Cosine after fix: still 0.06. Two more bugs waiting.

### The scope errors

`q_row0` was declared inside the k_tile loop. The output write, which happens after both loops, needed `q_row0` but it was out of scope. The compiler silently reused whatever value was on the stack at that point, producing output at arbitrary row positions.

`out_col` was declared twice: once inside the n_tile loop and once in the output write. The second declaration shadowed the first. Since both used the same formula, the values happened to be the same, but the shadowing masked a real structural problem.

Both variables needed to be declared once before both loops.

Cosine after fix: still 0.06. One more.

### The lane collision

The output write computed `q_row0 = warp_id * MMA_M + (lane % 16)`. With `lane % 16`, lane 0 and lane 16 both produce `q_row0 = 0`. They write to identical positions in the output array and one overwrites the other. Lanes 4 and 20 collide on row 1, lanes 8 and 24 on row 2, and so on. Half the output was being written twice and the other half was never written at all.

The correct formula is `lane / 4`, which assigns each unique row to exactly one group of four lanes with no collisions.

Cosine after fix: 0.19. The output was wrong but no longer completely random. The V accumulation was now the dominant error.

### The V accumulation

Each thread accumulates the output contribution from one column tile of S by multiplying softmax weights against the corresponding V rows. My first implementation reduced these contributions with the same butterfly pattern used for the row maximum.

The butterfly adds values from neighboring threads. Thread 0 accumulates for output column 0. Thread 1 accumulates for output column 2. After the butterfly, thread 0 was summing its column-0 contribution with thread 1's column-2 contribution. These are different output dimensions. The result was meaningless.

The fix uses `__shfl_sync` to fetch each neighbor's softmax weights explicitly, then multiplies them against V values for this thread's own output columns only. No cross-dimension mixing.

Cosine after fix: 0.81.

### The race condition

The kernel was launched with four blocks. Each block contained four warps covering 64 rows, exactly the full Q tile. Four blocks meant four independent groups all writing to the same output addresses with no coordination. Each block computed correct results and then overwrote the previous block's output.

One block is enough.

Cosine after fix: 0.81 confirmed. The remaining gap from 1.0 is quantization noise, not a bug. The validation in section 17 confirms this.

## 14. The Scale Layout

### The same problem

One register per thread for `scale_a`, one for `scale_b`. But A has 16 rows and each needs its own scale. B has 8 columns and each needs its own scale. A single `uint32_t` per thread cannot hold all of that. The hardware must be reading specific bytes from specific lanes, but the PTX ISA does not document which ones for this instruction variant.

### The probing method

Same approach as the fragment layout: probe empirically.

I filled Q and K with all 1.0 and set all scales to 127, which corresponds to 2 to the power of 0, meaning scale equals 1.0. The baseline output is 32.0 everywhere. Then I ran the kernel 32 times. Each run set exactly one lane's `scale_a` to 128, meaning scale equals 2.0, while all other lanes kept 127. If the hardware reads lane L's scale for row R, then row R of the output doubles from 32.0 to 64.0.

### The results

| Lane condition | Row affected |
| --- | --- |
| lane % 4 == 0 | lane / 4, covering rows 0 through 7 |
| lane % 4 == 1 | lane / 4 + 8, covering rows 8 through 15 |
| lane % 4 == 2 | no effect |
| lane % 4 == 3 | no effect |

The same probing on `scale_b` showed that only lanes where `lane % 4 == 0` have any effect, and lane L controls column `lane / 4`.

This is consistent with the A fragment structure. The register `a0` covers row0 and its scale is read from the lane with `lane % 4 == 0`. The register `a1` covers row0+8 and its scale comes from `lane % 4 == 1`. Lanes 2 and 3 contribute data through their A registers but the hardware ignores whatever scale value they hold.

## 15. Online Softmax and the Accumulation of V

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

With one block, the cosine was 0.81 with random inputs in the range [-1, 1]. To confirm this was quantization loss and not a remaining bug, I used inputs from the set {-1.0, -0.5, 0.0, 0.5, 1.0}. These are exactly representable in FP4 E2M1, so quantization is lossless and the GPU and CPU see identical data.
cosine similarity : 1.0000  PASS
ref[0..7] : -0.446  -0.879  -0.450   0.511   0.940   0.968  -0.947  -0.049
out[0..7] : -0.446  -0.879  -0.450   0.511   0.940   0.968  -0.947  -0.049

Bit-exact. Every element matches.

The 0.81 is the intrinsic precision cost of MXFP4 at `scale_vec::1X` granularity. FP4 E2M1 has only eight representable magnitudes. With one scale covering 32 elements, a single outlier in a block sets the scale for all 32 values and the remaining ones lose resolution. For inputs uniform in [-1, 1], roughly a quarter of values round to zero after block scaling. The CPU reference operates on the original float32 values, so the comparison is unfair. The kernel is correct. The 0.81 is an architectural constraint, not a bug.

---

*The kernel now correctly computes fused FP4 attention for a single Q tile against a single K tile, with the full score matrix living in registers throughout. What remains is extending K loading across multiple sequence tiles, adding asynchronous prefetching with cp.async, and extending V to use FP4 quantization for the second GEMM. That is the next section.*

*Code: [github.com/florianmattana/fp4-fused-attention-sm120](https://github.com/florianmattana/fp4-fused-attention-sm120)*
