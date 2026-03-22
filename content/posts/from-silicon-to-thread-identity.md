---
title: "From Silicon to Thread Identity: How CUDA Threads Know Who They Are"
date: 2026-02-13
draft: false
tags: ["cuda", "ptx", "gpu-architecture", "registers"]
description: "How threadIdx, blockIdx and blockDim are hardware registers, not software functions, and what that means for performance."
showToc: true
TocOpen: true
---

## The Natural Question

When you start learning CUDA, you use `threadIdx.x`, `blockIdx.x` and `blockDim.x` like magic variables that always contain the right value. At some point, you naturally start wondering: how are these values computed? Is there a function somewhere in the CUDA runtime that produces them? Can you see the source code behind them?

The answer is surprising: there is no code. These values are not the result of a software computation. They come directly from the hardware.

## Registers, Not Functions

### The Wrong Intuition

On a classic CPU, when you call a function, the processor executes a sequence of instructions, pushes values onto the stack, performs calculations, and returns a result. You might imagine that `threadIdx.x` works the same way, that some piece of code somewhere in the NVIDIA driver computes "you are thread number 42" and hands you that value.

### What Actually Happens

That is not at all what happens. `threadIdx`, `blockIdx` and `blockDim` correspond to special registers physically wired into the GPU's silicon.

Think of it like a maternity ward. When a baby is born, nobody asks them to go look up their name in an administrative file. A wristband is immediately attached to their wrist with all their information: name, time of birth, identification number. That wristband is physically there from the very first instant. The baby doesn't need to do anything to "compute" it. This is exactly how special registers work in CUDA. When a thread is "born" on the GPU, the hardware instantly attaches its coordinates (`threadIdx`, `blockIdx`, `blockDim`) to it. The thread just has to "look at its wristband" to know who it is.

A special register is a small piece of memory integrated directly into the processor, whose value is set by the hardware itself, with no software intervention. When your thread reads `threadIdx.x`, it simply accesses the contents of such a register. There is no function call, no call stack, no intermediate computation. It is a direct, instantaneous read from the hardware.

## How These Values End Up in the Registers

To understand the mechanism, you need to know what happens when a kernel is launched.

Think of it like a school principal on the first day of school. The principal (the hardware scheduler) has the complete list of students and the class assignments in front of them. Before classes begin, they send each student to their classroom, and at each desk they have already placed a label: "You are student number 3 in class B, and your class is in building 2." When the student sits down, they don't need to do anything. The information is already there, on their desk.

Here is what happens step by step.

**Step 1: the kernel launch.** When you write a kernel call with a grid and block configuration, the CUDA runtime transmits this configuration to the GPU. This is the moment when the "principal" receives their list of classes and students.

**Step 2: distribution across SMs.** The hardware scheduler distributes blocks of threads across the different Streaming Multiprocessors (SMs), the GPU's compute units. This is the moment when the principal sends each class to its room.

**Step 3: writing the registers.** When a block is assigned to an SM and its threads begin executing, the scheduler physically writes into each thread's special registers the values that correspond to it: its `threadIdx`, its block's `blockIdx`, and the `blockDim` dimensions. This is the moment when each student finds their label on their desk.

**Step 4: the kernel executes.** When your code starts running, the values are already there, ready to be read. The thread never needs to "discover" who it is through a computation. The hardware told it at the moment of its birth.

To make this concrete, suppose you launch a kernel with a grid of 4 blocks, each block containing 256 threads. At launch time, the scheduler assigns block 0 to SM number 5, block 1 to SM number 12, and so on. For each thread in block 0, it writes `blockIdx.x = 0` in the corresponding register. For thread number 147 of that block, it writes `threadIdx.x = 147`. All of this happens in hardware, before a single instruction of your kernel executes.

## What It Looks Like at the Assembly Level

In PTX, the built-in variables we know from CUDA C++ are exposed as special registers with predefined names.

The naming might look cryptic at first, but there is a logic to it. `%tid` simply stands for Thread ID, which maps directly to `threadIdx`. `%ctaid` stands for Cooperative Thread Array ID. Internally, NVIDIA doesn't call a block a "block." The real technical name is CTA, or Cooperative Thread Array. So `%ctaid` literally means "the ID of the CTA," in other words, which block you are in within the grid. The word "block" that we use in CUDA C++ is actually a pedagogical simplification. Similarly, `%ntid` stands for Number of Threads in the CTA (i.e., `blockDim`).

When the NVCC compiler transforms your CUDA code into PTX, an expression like `threadIdx.x` simply becomes an instruction to read the `%tid.x` register. In terms of cost, it is comparable to reading the `rax` register on an x86 processor: an operation that takes essentially zero extra cycles. There is no memory latency, no cache access, no computation.

To give you an order of magnitude: an access to GPU global memory costs between 200 and 800 cycles. An access to shared memory costs around 20 to 30 cycles. An access to a register costs 0 to 1 cycles. Reading `threadIdx.x` is therefore between 200 and 800 times cheaper than fetching a value from global memory.

## Seeing It for Real

You can verify all of this yourself on [godbolt.org](https://godbolt.org/). Take the simplest possible kernel. The key thing to notice is the `mov` instructions reading `%ctaid.x`, `%tid.x`, `%ctaid.y`, and `%tid.y`. No function call. No memory load. Just a direct read from a hardware register. Then the `mad` (Multiply-Add) instructions do the local-to-global mapping in a single operation each. That's it.

## What This Means for the Local to Global Mapping

Let's revisit a simple computation:

```cuda
int globalX = blockIdx.x * BN + threadIdx.x;
int globalY = blockIdx.y * BM + threadIdx.y;
```

In light of what we now know, we can better understand what is actually happening.

Imagine you live in a housing complex. You know two things: your building number (`blockIdx`) and your apartment number within that building (`threadIdx`). If each building contains 50 apartments (`blockDim`), then your global apartment number in the entire complex is `building_number * 50 + local_number`. Building 3, apartment 12, is global apartment number 3 * 50 + 12 = 162. Both starting pieces of information (building and local number) were already on your wristband. The only real computation is the multiplication and the addition.

Here's a full worked example. Consider a kernel launched with 16x16 thread blocks on a 64x64 matrix. The thread that has `blockIdx = (2, 1)` and `threadIdx = (5, 9)` computes its global coordinates as follows: `globalX = 2 * 16 + 5 = 37` (column 37) and `globalY = 1 * 16 + 9 = 25` (row 25). This thread now knows it is responsible for element `C[25][37]` of the matrix. The reads of `blockIdx` and `threadIdx` cost 0 cycles (hardware registers). The multiplication and addition cost 1 to 2 cycles. In total, the thread determined its unique global position in fewer than 5 cycles. Compare that to the hundreds of cycles that even a single access to global memory data will cost afterwards.

## One Step Further to Avoid Confusion

A common misconception worth clearing up: when we say `threadIdx` and `blockIdx` live in hardware registers, it does not mean there are millions of physical silicon slots for millions of threads. The GPU does not physically instantiate all threads at once.

What actually happens is that each SM has a fixed-size register file (for example, 65,536 registers on an A100). That is real silicon, a fixed physical resource that does not grow. If your kernel uses 32 registers per thread, one SM can hold at most 2,048 threads at a time. Across all 108 SMs of an A100, that gives you roughly 220,000 resident threads at any given moment, not the millions you may have launched.

The rest of the threads simply wait in line. When a block finishes on an SM, the scheduler assigns a new block to that same physical space. The special registers (`%tid`, `%ctaid`) are overwritten with the new thread's values, just like a hotel room being cleaned and prepared for the next guest.

So do not confuse threads launched (logical, potentially millions) with threads resident (physical, limited by the register file). The hardware registers are real silicon, but they are recycled across threads over time. And this is exactly why register pressure matters so much in CUDA: the more registers your kernel uses per thread, the fewer threads the SM can host simultaneously, and the lower your occupancy drops.

## Key Takeaways

CUDA's built-in variables (`threadIdx`, `blockIdx`, `blockDim`) are not software. They are hardware registers, filled by the GPU's hardware scheduler the moment each thread is launched. Reading them costs nothing. This is a deliberate architectural decision by NVIDIA to allow tens of thousands of threads to instantly know their identity, with zero overhead. In PTX assembly, they appear as `%tid` (Thread ID), `%ctaid` (Cooperative Thread Array ID, because NVIDIA internally calls a block a CTA), and `%ntid` (number of threads in the CTA). The local-to-global mapping that we build from these registers adds only a multiplication and an addition, a few cycles at most, making it one of the cheapest operations in any kernel.

*Originally published on [LinkedIn](https://www.linkedin.com/pulse/from-silicon-thread-identity-how-cuda-threads-know-who-mattana-p8jje/).*
