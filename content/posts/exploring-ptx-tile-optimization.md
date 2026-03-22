---
title: "Exploring PTX: A Close Look at Tile Optimization in CUDA"
date: 2026-01-15
draft: false
tags: ["cuda", "ptx", "optimization", "shared-memory"]
description: "Comparing naive and tiled matrix transpose kernels at the PTX level to understand shared memory, coalescing, and bank conflicts."
showToc: true
TocOpen: true
---

Originally, I suggested on LinkedIn that I would write an article to unabstract a PyTorch program and reveal some potential CUDA optimizations we could make by debugging it. I actually took another road, but I kept in mind the desire to dig deeper from a given point of abstraction.

For context, I am currently writing a small program to compare some kernels with different optimization choices or intentionally unoptimized kernels to highlight the differences between various implementation options.

Before reaching the conclusion by uploading a terminal screenshot from which we already know the results in advance, I intended to explore the trick at a lower level.

As you might know, a CUDA program is compiled in two stages. First, the NVCC compiler splits your code into two parts: CPU code and GPU code. The CPU code is compiled in the traditional way, while the GPU code is compiled into PTX, which is a sort of intermediate representation between your source code and what the GPU actually runs. The second compilation step turns PTX into SASS, the actual machine code executed by the GPU. This can happen either during compilation (AOT) or at runtime (JIT). For now, let's focus on PTX itself.

So there I was, and I thought it would be interesting to catch the difference between a naive kernel implementation and a Matrix Transpose Optimized Kernel with tiling.

In CUDA, a tile is a small block of data that fits into shared memory. This tiling strategy reduces slow global memory accesses by reusing data from the fast shared memory multiple times. At least, that's what we are going to assess.

If you want to experiment by yourself I am dropping the generic code here and you can use [godbolt.org](https://godbolt.org/) to get the PTX result:

```cuda
// KERNEL A : Matrix Transpose Naive
__global__ void matrix_transpose_naive(
    const float* input,
    float* output,
    int width,
    int height)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col < width && row < height)
    {
        int input_index  = row * width + col;
        int output_index = col * height + row;
        output[output_index] = input[input_index];
    }
}
```

```cuda
// KERNEL B : Matrix Transpose Optimized
__global__ void matrix_transpose_optimized(
    const float* input,
    float* output,
    int width,
    int height)
{
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];

    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    if (x < width && y < height)
    {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }

    __syncthreads();

    int x_out = blockIdx.y * TILE_SIZE + threadIdx.x;
    int y_out = blockIdx.x * TILE_SIZE + threadIdx.y;

    if (x_out < height && y_out < width)
    {
        output[y_out * height + x_out] = tile[threadIdx.x][threadIdx.y];
    }
}
```

And here are the PTX results we are going to work with:

```asm
// Kernel A PTX
.visible .entry matrix_transpose_naive(...)
{
    ld.param.u64    %rd1, [_param_0];
    ld.param.u64    %rd2, [_param_1];
    ld.param.u32    %r3,  [_param_2];
    ld.param.u32    %r4,  [_param_3];
    mov.u32         %r5, %ntid.x;
    mov.u32         %r6, %ctaid.x;
    mov.u32         %r7, %tid.x;
    mad.lo.s32      %r1, %r5, %r6, %r7;
    mov.u32         %r8, %ctaid.y;
    mov.u32         %r9, %ntid.y;
    mov.u32         %r10, %tid.y;
    mad.lo.s32      %r2, %r9, %r8, %r10;
    setp.ge.s32     %p1, %r1, %r3;
    setp.ge.s32     %p2, %r2, %r4;
    or.pred         %p3, %p1, %p2;
    @%p3 bra        $L__BB0_2;
    cvta.to.global.u64  %rd3, %rd1;
    mad.lo.s32      %r11, %r2, %r3, %r1;
    mad.lo.s32      %r12, %r1, %r4, %r2;
    mul.wide.s32    %rd4, %r11, 4;
    add.s64         %rd5, %rd3, %rd4;
    ld.global.f32   %f1, [%rd5];
    cvta.to.global.u64  %rd6, %rd2;
    mul.wide.s32    %rd7, %r12, 4;
    add.s64         %rd8, %rd6, %rd7;
    st.global.f32   [%rd8], %f1;
$L__BB0_2:
    ret;
}
```

```asm
// Kernel B PTX
.visible .entry matrix_transpose_optimized(...)
{
    ld.param.u64    %rd1, [_param_0];
    ld.param.u64    %rd2, [_param_1];
    ld.param.u32    %r9,  [_param_2];
    ld.param.u32    %r10, [_param_3];
    mov.u32         %r11, %ctaid.x;
    shl.b32         %r1, %r11, 5;
    mov.u32         %r2, %tid.x;
    add.s32         %r3, %r1, %r2;
    mov.u32         %r12, %ctaid.y;
    shl.b32         %r4, %r12, 5;
    mov.u32         %r5, %tid.y;
    add.s32         %r6, %r4, %r5;
    setp.ge.s32     %p1, %r3, %r9;
    setp.ge.s32     %p2, %r6, %r10;
    or.pred         %p3, %p1, %p2;
    @%p3 bra        $L__BB0_2;
    cvta.to.global.u64  %rd3, %rd1;
    mad.lo.s32      %r13, %r6, %r9, %r3;
    mul.wide.s32    %rd4, %r13, 4;
    add.s64         %rd5, %rd3, %rd4;
    ld.global.f32   %f1, [%rd5];
    mov.u32         %r14, tile;
    mad.lo.s32      %r15, %r5, 132, %r14;
    shl.b32         %r16, %r2, 2;
    add.s32         %r17, %r15, %r16;
    st.shared.f32   [%r17], %f1;
$L__BB0_2:
    bar.sync        0;
    add.s32         %r7, %r4, %r2;
    setp.ge.s32     %p4, %r7, %r10;
    add.s32         %r8, %r1, %r5;
    setp.ge.s32     %p5, %r8, %r9;
    or.pred         %p6, %p5, %p4;
    @%p6 bra        $L__BB0_4;
    mov.u32         %r18, tile;
    mad.lo.s32      %r19, %r2, 132, %r18;
    shl.b32         %r20, %r5, 2;
    add.s32         %r21, %r19, %r20;
    ld.shared.f32   %f2, [%r21];
    mad.lo.s32      %r22, %r8, %r10, %r7;
    cvta.to.global.u64  %rd6, %rd2;
    mul.wide.s32    %rd7, %r22, 4;
    add.s64         %rd8, %rd6, %rd7;
    st.global.f32   [%rd8], %f2;
$L__BB0_4:
    ret;
}
```

## 1. PTX, how nice you look

```asm
.visible .entry (rdm_kernel_name)
```

The common first line in both kernels is printed because of the `__global__` keyword. This makes the function visible outside the module, much like a public class in traditional programming.

## 2. Program is setting up

Another common point between both PTX kernels is that each element of the kernel signature is stored in a dedicated register: 64-bit registers for pointers (since they store memory addresses; modern GPUs have large memory, up to tens of gigabytes, and 64-bit addresses can reference up to 16 exabytes; a 32-bit pointer could only address 4GB), and 32-bit registers (4 bytes) for integers.

One fun fact you may have noticed is the difference in selected registers between the two kernels for the same operation. This difference in register usage is not arbitrary. It is a direct consequence of compiler-driven register scheduling to support shared memory, tiling, and memory coalescing.

When a kernel uses `__shared__` memory, the compiler knows that shared memory will be leveraged to reduce slow global memory accesses and applies additional memory-related optimizations. As a result, it reorganizes register usage to hold indices, offsets, and temporary values needed for shared memory loads and stores.

## 3. Thread indices calculation

This marks the first divergence between the two kernels.

```asm
// KERNEL A (not optimized)
mov.u32      %r5, %ntid.x;
mov.u32      %r6, %ctaid.x;
mov.u32      %r7, %tid.x;
mad.lo.s32   %r1, %r5, %r6, %r7;
mov.u32      %r8, %ctaid.y;
mov.u32      %r9, %ntid.y;
mov.u32      %r10, %tid.y;
mad.lo.s32   %r2, %r9, %r8, %r10;
```

```asm
// Kernel B (Optimized)
mov.u32      %r11, %ctaid.x;
shl.b32      %r1, %r11, 5;
mov.u32      %r2, %tid.x;
add.s32      %r3, %r1, %r2;
mov.u32      %r12, %ctaid.y;
shl.b32      %r4, %r12, 5;
mov.u32      %r5, %tid.y;
add.s32      %r6, %r4, %r5;
```

These two code snippets calculate thread indices in different ways, revealing an important assumption about block dimensions. The difference is:

The first version is flexible, works with any block dimensions, but requires reading `blockDim` and using a multiply-add instruction. The second version is faster (one less instruction, uses shift instead of multiply) but only works if blocks are 32x32.

**First kernel:** move the block dimension in x (number of threads per block) into register r5, move the block ID in x-dimension into register r6, move the thread ID within block into register r7, then multiply r5 by r6, add r7, and store result in r1. This uses the actual runtime block dimension instead of a hardcoded value. It translates the formula: `globalX = blockDim.x * blockIdx.x + threadIdx.x`.

**Second kernel:** move the block ID in x-dimension into register r11, shift r11 left by 5 bits and store in r1 (which multiplies by 32), move the thread ID within block into register r2, add r1 and r2 and store result in r3. The shift left by 5 bits multiplies by 2^5 = 32. This assumes the block size is exactly 32, hardcoded at compile time because of the tile size we defined to 32. The formula becomes: `globalX = blockIdx.x * 32 + threadIdx.x`.

A left shift is cheaper than a multiplication because it is implemented as simple bit rewiring with minimal hardware and latency, while multiplication requires complex arithmetic logic, deeper pipelines, and higher scheduling and energy costs, even when both appear as a single instruction.

## 4. Index boundaries

```asm
// Kernel A (naive)
setp.ge.s32   %p1, %r1, %r3;
setp.ge.s32   %p2, %r2, %r4;
or.pred       %p3, %p1, %p2;
@%p3 bra      $L__BB0_2;
```

```asm
// Kernel B (optimized)
setp.ge.s32   %p1, %r3, %r9;
setp.ge.s32   %p2, %r6, %r10;
or.pred       %p3, %p1, %p2;
@%p3 bra      $L__BB0_2;
```

The structure is identical when it comes to preventing indices from going out of bounds. We can just observe that different registers are used (r3 vs r9, r6 vs r10) in order to reduce register pressure, enable better instruction scheduling, and improve instruction-level parallelism.

## 5. Calculation time

### 5.1 Kernel A: coalesced read, uncoalesced write

If we want to identify the moment when performance diverges even further, we need to check whether the memory access is coalesced. As a quick reminder, memory access is coalesced when threads in a warp access consecutive addresses. For float data, each address should be 4 bytes apart.

```asm
// Read part (COALESCED)
cvta.to.global.u64  %rd3, %rd1;
mad.lo.s32          %r11, %r2, %r3, %r1;
mul.wide.s32        %rd4, %r11, 4;
add.s64             %rd5, %rd3, %rd4;
ld.global.f32       %f1, [%rd5];

// Write part (UNCOALESCED)
mad.lo.s32          %r12, %r1, %r4, %r2;
cvta.to.global.u64  %rd6, %rd2;
mul.wide.s32        %rd7, %r12, 4;
add.s64             %rd8, %rd6, %rd7;
st.global.f32       [%rd8], %f1;
```

You can tell whether memory access is coalesced just by looking at the PTX. Check how `threadIdx.x` changes the address: if each thread increments the address by the size of one element (4 bytes for a float), the access is coalesced. If each thread jumps by a much larger number, like the matrix height, the access is not coalesced.

**Read part:** `mad.lo.s32 %r11, %r2, %r3, %r1`. Here %r1 corresponds to `threadIdx.x` and %r2 * %r3 computes a constant base offset (row * width). `threadIdx.x` is added directly to the index, giving `index(thread i) = constant(r2 * r3) + i`. Each thread reads the next element in memory. Coalesced.

**Write part:** `mad.lo.s32 %r12, %r1, %r4, %r2`. Here %r1 = `threadIdx.x`, %r4 = height. This calculates `index(thread i) = threadIdx.x * height + row`. Each thread jumps by height elements instead of 1. The addresses are not consecutive, the write is uncoalesced.

### 5.2 Kernel B: the optimized write through shared memory

```asm
mov.u32         %r14, tile;
mad.lo.s32      %r15, %r5, 132, %r14;
shl.b32         %r16, %r2, 2;
add.s32         %r17, %r15, %r16;
st.shared.f32   [%r17], %f1;
```

My first surprise was to see `tile` as it was never declared in PTX. I then understood that `tile` is a `__shared__` memory variable declared in the original CUDA code and it doesn't appear explicitly as a standard variable in the PTX code.

But the most important part is the 132 on line 2. In our code, we have `__shared__ float tile[32][33]`. Considering that each float is 4 bytes, the second dimension is 33 (not 32), and the stride is the distance in bytes between the start of one row and the next:

`stride = 33 floats * 4 bytes/float = 132 bytes`

This extra float (33 instead of 32) is used to avoid shared memory bank conflicts.

## 6. Thread synchronization

An important divergent point is about thread synchronization, which is mandatory when we use shared memory.

```asm
$L__BB0_2:
    bar.sync  0;
```

When we use a tile in shared memory, all threads in a block write their part of the data into that tile. However, threads may execute at slightly different speeds. If a thread tries to read from the tile before another thread has finished writing, it could read wrong or incomplete data.

That's why we use a synchronization barrier (`__syncthreads()` in CUDA, `bar.sync 0` in PTX). It makes all threads wait until everyone has finished writing to the shared memory. After the barrier, it is safe for all threads to read from the tile.

## 7. Wrap up

Exploring a PTX file is a great way to better understand CUDA code. It is a good exercise to become a stronger engineer. That said, PTX is just a low-level translation of your higher-level code. Nothing will appear in PTX that you didn't already write.

*Originally published on [LinkedIn](https://www.linkedin.com/pulse/exploring-ptx-close-look-tile-optimization-cuda-florian-elio-mattana-9jjfe/).*
