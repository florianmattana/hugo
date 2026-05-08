---
title: "SASS King, Part 2: Reading the Compiler's Mind"
date: 2026-05-08
draft: true
description: "How studying SASS taught me to identify ptxas heuristic decisions, understand their logic, and explore where the compiled binary diverges from source-level intent."
tags: ["cuda", "sass", "gpu", "nvidia", "reverse-engineering", "sm120", "blackwell", "ptxas"]
categories: ["SASS King"]
---

## 1. Why a Second Article

When I started SASS King I didn't know really the purpose behind it, except knowing better the compiler behaviour. Because the more I went deep into CUDA, the more I saw myself drawn by the click between software and hardware, that tiny frontier few people actually care about.

As a GPU programmer, when you write a kernel you still think at a level of abstraction that delegates a lot of superpower to the CUDA machinery. And nowadays NVIDIA tends to keep abstracting up the stack to give developers the ease to focus on high level challenges that are already numerous. As we will all agree that the principal use cases are LLM architectures, we can start to assess that the main challenges a developer faces are not really at the register level. The further forward we go, we can even start to conclude that the main bottlenecks are rarely at compute level and more at memory level, how we transfer data between H2D, D2D or D2H. A recent documented case makes this concrete. Anam, a company building real-time AI avatars, [found that their Cara-3 inference pipeline was not GPU-bound at all](https://zymtrace.com/blog/how-anam-achieved-250-faster-inference-using-zymtrace-continuous-gpu-profiling). The GPU would run a burst of kernels, then sit idle, then run again. The bottleneck was Python runtime overhead stalling CPU dispatch and leaving the GPU waiting. Fixing the dispatch path, not the kernels, gave them a 2.5x latency improvement on the same hardware. Considering this, trying to achieve a full understanding of deep machinery seems like being on the wrong side of the risk/reward ratio.

But I didn't care much. When I paused my job career to dedicate my time to the GPU open source community, I wanted to pursue my curiosity and let the future dictate whether I would one day value my knowledge or not.

The first SASS King article was about learning to read SASS from first principles, and I started to receive a lot of energy from the community. Like they were eager to know this machinery too but played the safe game. But the more deeply I went into this thing, the more I discovered we weren't just curious. We needed some answers here.

What kind of answers? The fundamental one. Do we leave gains on the table?

The usual workflow is to optimize at a higher level in C++ and try to find the algorithm sweet spot. Increase occupancy but nothing changes because the kernel was memory-bound, or add shared memory and it shifts pressure to registers. You unroll loops and ILP improves but then fewer warps are resident. Fuse kernels and launch overhead drops but memory contention rises. Same kernel. Same correctness. Different constraint still active. We always face the same limits: SM occupancy, warp scheduler efficiency, register pressure, bandwidth saturation, dependency chain depth, SIMT divergence.

So the question was: since we have strong reference implementations like [CUTLASS](https://github.com/NVIDIA/cutlass) or [FlashAttention](https://github.com/Dao-AILab/flash-attention), can we look at what they emit at the SASS level, extract the patterns that work, and use them to do better than what the compiler chose for our own kernels? But to do better than the compiler, you first need to understand what the compiler actually does. And what it does is not search for the optimal answer. It can't. The problem is too big.

The heuristic. The concept is simple.

When a problem is too big to solve perfectly, a rational agent stops searching for the best answer. They search for a good enough answer, fast.

ptxas faces exactly that. Register allocation is NP-complete. Instruction scheduling across pipelines is NP-hard. Solving these exactly on a real kernel would take forever. So the compiler uses heuristics. Spill the variable with the longest live range. Schedule by DAG depth. Inline functions below a size threshold.

As an example, chess engines do the same thing. 10^40 possible games. No way to enumerate them. So they approximate: centre control, passed pawns, king safety. Prune the bad looking branches.

You do it too. Lost your keys? You check three spots. Not the whole flat.

A heuristic is not a failure of rigour. It's the rational response to scarcity. A perfect answer at infinite cost is less useful than a 95% answer in a millisecond.

So when you read a SASS dump and something looks wrong, it's almost never a bug. It's a heuristic that matched the code in a way not expected. The first signal came from a dump that showed two tensor-core instructions where sixteen were expected. Not a compiler error. A decision I didn't yet understand.

That experience turned the project into two questions at once. First: can we identify those heuristic decisions, measure their cost, and correct them through source restructuring, PTX inline, or direct patching? Second: can we build the missing tooling to do this systematically? cuobjdump gives a text dump, enough for manual reading but not for analysis at scale. What's missing is a proper disassembler integrated into an analysis framework, and eventually an assembler that makes SM120 binaries writable, not just readable. I'll come back to why I believe that's not a fantasy.

So yes, performance was the first motivation. A second one emerged also through conversations about the project. GPU binaries are opaque in practice. Tools like cuobjdump and CUPTI exist, but there is no public framework for systematic static analysis of compiled GPU code: no control flow graph reconstruction, no cross-references, no scriptable pattern matching. The equivalent of what security researchers take for granted on x86 does not exist for cubins. A cubin runs on a device that has direct memory access to the host via PCIe BAR mappings. Not being able to inspect what that binary actually does is not just inconvenient for performance engineers. It is a gap in the security story. I'll try to go deeper on both.

But I want to be clear about where this started. SASS King is not built against NVIDIA. It's an engineering research project built on top of what's already available. I don't see only a black box to unravel. I see a journey to answer questions that maybe even NVIDIA hasn't fully answered.

> ⭐ *SASS King is an open project. If any of this resonates with you, the repo is at [github.com/florianmattana/sass-king](https://github.com/florianmattana/sass-king). Open issues describe what needs to be done next, [CONTRIBUTING.md](https://github.com/florianmattana/sass-king/blob/main/CONTRIBUTING.md) explains how to add a dump or a correction, and a star or a watch helps the project reach the people who can contribute hardware I don't have access to.*


## 2. Where SASS Sits in the Full Execution Chain

You can't study a mechanism without understanding what feeds it and what it feeds. SASS is no exception. To know what it means, you need to know where it comes from and where it goes.

Here is the broad picture:

```
kernel.cu → cicc → PTX → ptxas → SASS → fatbinary → runtime dispatch → PCIe → GPU
```

We write a kernel in CUDA C++. Maybe with CUTLASS, maybe from scratch. We compile with nvcc and get a cubin ready to execute. Simple story. Except nvcc is not a compiler. It's a driver that orchestrates several compilers, each making independent decisions on your behalf.

Three of them matter for SASS King.

cicc is NVIDIA's device frontend. Most people don't know it exists. It takes the preprocessed source, extracts device code, and compiles it to PTX, the architecture-independent virtual instruction set. cicc handles everything CUDA-specific: `__global__`, `__shared__`, `threadIdx`, template instantiation for device functions. It is a full optimizing compiler with its own IR and its own analysis passes.

ptxas takes that PTX and produces SASS, the native ISA for one specific architecture. sm_80, sm_90, sm_120. This is where register allocation happens, where instructions get reordered to hide latency, and where control codes are computed and embedded into every 128-bit instruction slot. The output is a cubin.

The host compiler handles the CPU side. gcc, clang, or MSVC. nvcc passes host code through with injected stubs for kernel launch. The host compiler has no idea it's part of a GPU build. After that, the fatbin is assembled, one or more cubins plus a PTX fallback, embedded as a binary blob inside the host object file and linked against libcudart.

Each compiler makes independent decisions. cicc decides how to lower C++ to PTX. ptxas decides how to schedule and allocate. If a kernel is slow, the cause could live in either one, and they leave different traces. But ptxas is the one I decided to investigate. It is the last stage that touches your code before it becomes the bits the hardware runs. Every other stage's decisions are filtered through it. Its heuristic choices are the ones most directly visible in the SASS and most directly responsible for the performance you measure.

That's the upstream. The downstream matters too.

Your binary is on disk. You launch it. Before a single GPU thread runs, the runtime does real work. Libcudart loads, registers your kernels and the embedded fatbin with the driver, initializes streams and contexts. The CUDA driver detects available GPUs, creates a context, allocates resources. Then it reads the fatbin and queries the GPU's compute capability.

Three outcomes are possible. If an exact cubin match exists: direct load via PCIe, fast, no compilation, the happy path. If no cubin but PTX is available: the driver JIT-compiles the PTX to SASS using its own internal ptxas, caches the result, slow first launch but fast afterwards. This is why a binary built in 2020 can run on a 2026 GPU. If nothing matches: launch failure.

One detail matters here. The build-time ptxas that created your cubin and the runtime ptxas inside the driver are not necessarily the same version. NVIDIA says the JIT cache is wiped on driver upgrade, so SASS output is driver-specific. Same PTX, different driver, potentially different SASS. If you're fingerprinting binaries for analysis or security, this creates a real divergence between what you analyzed and what actually runs.

Every stage in this chain shapes the SASS that reaches the hardware. Understanding the chain is not optional. It is the prerequisite for knowing where to look when something goes wrong.

This is where SASS King focuses: the gap between source-level intent and machine-level reality.


## 3. Prior Art and the Remaining Gap

Five bodies of work matter here. Each one taught me something. None of them covers what SASS King is trying to do.

[Jia et al. 2018](https://arxiv.org/abs/1804.06826) built the methodological foundation for empirical GPU microbenchmarking on Volta and Turing. Latency per instruction, throughput per pipeline, producer-consumer dependencies. Rigorous, reproducible, eight years behind current hardware. But more than a paper, Jia is a methodological reminder. You cannot claim to beat a heuristic if you have no baseline to measure against. This shaped a core principle in SASS King: the project works in two phases. Static observation first, from the dump, without needing the hardware. Dynamic validation second, when the hardware is available to support or contradict what was observed. An observation without measurement is explicitly marked as such. That discipline comes directly from Jia.

[Kuterdinel's nv_isa](https://kuterdinel.com/nv_isa/) went after a different problem. Rather than measuring performance, he reverse-engineered the SM90a ISA encoding by fuzzing nvdisasm. Fuzzing here means generating systematic binary patterns, feeding them to NVIDIA's own disassembler, and observing what mnemonics and operands come out. Vary bits one by one, watch how the output changes, and you reconstruct the encoding rules without ever seeing NVIDIA's internal specification. You are reverse-engineering the decoder itself. The result is a machine-readable ISA description derived purely from observation. This is exactly what you need if you want to build a proper disassembler integrated into a framework like Ghidra: a formal description of the instruction encoding. Without this kind of work, building static analysis tooling means guessing. What Kuterdinel does not give you is whether an instruction costs 29 cycles or 35, or whether the compiler's choice to emit it was the right one. Encoding coverage, no performance meaning.

[Redplait](https://github.com/redplait/denvdis) went further than anyone on the tooling side. He built [ced](https://redplait.blogspot.com/2025/07/ced-sed-like-cubin-editor.html), a cubin editor for inline patching of existing binaries. He published [Perl bindings](https://redplait.blogspot.com/2025/10/sass-disasm-on-perl.html) for SASS disassembly. He produced [latency analysis work](https://redplait.blogspot.com/2026/04/sass-latency-analysis.html) for instruction reordering. And he did something very cool: he extracted the [MD files](https://github.com/redplait/denvdis/tree/master/data12) directly from the ptxas binary.

Since ptxas is a closed binary, you cannot read its source code. But like any binary, it contains internal data structures, and in the case of ptxas, those structures are tables that describe every SASS instruction. The mnemonic, the binary encoding, the valid operands, the modifier constraints. NVIDIA uses these tables so ptxas knows how to assemble each instruction. Redplait opened the binary, found the tables, and published them. Not by accessing source code. Just by reading what was already there, inside the binary that ships with every toolkit.

This matters because it means the formal encoding description for SM120 instructions already exists, extracted from ptxas itself. What SASS King adds is the empirical layer: what each instruction actually costs, how the compiler chooses between alternatives, and what the performance implications are. Redplait gives the structure. SASS King measures the behaviour.

But one constraint needs to be named. nvasm_internal, NVIDIA's internal SASS assembler, [is not public](https://redplait.blogspot.com/2025/12/libcudaso-internals.html). Today, if you want to modify SM120 SASS, ced lets you patch bytes directly inside an existing cubin. If you want to write SASS from scratch and assemble it into a binary, you can't. Nobody has built an SM120 assembler yet. Maxwell got [MaxAS](https://github.com/NervanaSystems/maxas). Turing got [TuringAS](https://github.com/daadaada/turingas). Blackwell has nothing. Building one requires exactly the kind of encoding documentation that SASS King and redplait's extracted tables provide. This is not a blocker for SASS King. It is the reason SASS King is necessary. The encoding knowledge the project documents is precisely what an assembler would need to exist.

[Huerta et al. 2025](https://arxiv.org/abs/2503.20481) changed how I think about control codes.

Here is the simplest way to explain what they found. When the GPU is about to execute an instruction, it needs to know: is it safe to go now, or do I have to wait for something? On a CPU, the hardware figures this out by itself, tracking which registers are being written and which are being read, in real time. That mechanism is called a scoreboard. It works, but it costs transistors and power.

Modern NVIDIA GPUs do something different. They offload that job to the compiler. ptxas analyses the code at compile time, figures out exactly how long each instruction will take, and embeds the answer directly into the binary. Every 128-bit instruction slot carries a set of control bits that tell the hardware: wait this many cycles before issuing the next instruction from this warp. Which dependency counter to increment. Which counters to check before proceeding. Whether to let another warp run next cycle.

The hardware does not verify any of this. It trusts the compiler completely. If ptxas writes the wrong stall count, the GPU does not slow down gracefully. It executes with incorrect data. This is not a performance hint. It is a correctness contract.

In practice, this means each warp carries a small stall counter that counts down every cycle, and six dependency counters, SB0 through SB5, for instructions whose latency is not fixed, like memory loads. The producer instruction increments a counter when it issues. The counter decrements when the result is written or when the source registers are read, depending on the type of hazard. The consumer instruction checks a wait mask before it is allowed to issue.

They also confirmed that the .reuse bit on instruction operands controls a small hardware cache in front of the register file. The compiler decides which operands to cache. And a yield bit forces the warp scheduler to switch to another warp on the next cycle. They call the resulting scheduling policy CGGTY: Compiler Guided Greedy Then Youngest.

The whole mechanism uses 0.09% of register file area. A traditional scoreboard doing the same job would cost 5.32%. That is why NVIDIA chose this design.

Why does this matter for SASS King? Because the control codes I am decoding on SM120 are the same mechanism. Stall counter, dependency counters, wait mask, yield bit, reuse bits. Huerta reverse-engineered how the hardware consumes them. SASS King reverse-engineers how ptxas decides to emit them, and whether those decisions are optimal. Huerta's work covers Ampere. SM120 is Blackwell. The mechanism is likely the same, but the specifics need to be remeasured: bit positions, latencies, instruction families, MMA paths. Nobody has done that for SM120. That is part of what SASS King does.

[Yan et al. 2026](https://arxiv.org/abs/2604.26889) went in the other direction entirely. Below SASS, into the driver.

When you launch a CUDA kernel, you probably think: my code goes to the GPU and runs. But there is a whole layer in between that nobody sees. The closed-source NVIDIA driver takes your kernel launch, translates it into a sequence of low-level hardware commands, and submits them to the GPU through a specific protocol. Think of it like writing a letter. You write the content (your kernel), but someone else puts it in an envelope, addresses it, and drops it in the mailbox. That someone is the driver, and nobody knows exactly what it writes on the envelope.

Concretely: the driver writes commands into a buffer in host memory called a pushbuffer, then posts a notification to the GPU through a PCIe register called a doorbell. The GPU picks up the commands and executes them. This protocol is entirely undocumented.

Yan et al. found a way to intercept that moment. They modified the open-source kernel driver to install a hardware watchpoint on the doorbell register. Every time the driver rings the doorbell, the system freezes and they can read the exact commands that were just submitted. They see what the driver sends to the GPU, byte by byte.

Two findings from their work put SASS-level analysis in perspective.

First, for data transfers on an NVIDIA A40, the driver silently picks between two mechanisms depending on the transfer size. For small transfers, it embeds the data directly in the command stream, about 24 nanoseconds of raw hardware latency. For larger transfers, it uses a dedicated copy engine, about 500 nanoseconds of startup. You cannot control this choice from CUDA. You cannot see it from SASS. And what Nsight reports as "CUDA HW" duration for an 8-byte transfer is about 94% driver overhead, not hardware execution time. The gap shrinks as transfers get larger, but for small messages the driver cost dominates.

Second, between CUDA 11.8 and CUDA 13.0, on the exact same kernel graph of 2000 nodes on the same hardware, launch overhead dropped from 209 microseconds to 5.9 microseconds. Nothing changed in the SASS. Nothing changed in the hardware. The entire gain came from how the driver packaged its commands: fewer notifications to the GPU, a more compact command stream, one submission instead of many.

This contextualizes SASS King. The SASS you read is necessary but not sufficient. Below it, the driver makes invisible decisions. Above it, cicc shapes what ptxas receives. SASS King focuses on ptxas because that is where the heuristic decisions most directly visible in the machine code live. But it is one layer in a stack that has opacity at every level.

Do you know Bernard de Chartres? He is credited with the idea that further progress comes from building on what predecessors discovered rather than pretending the work does not exist.

These are my giants. Jia gives the measurement methodology. Kuterdinel gives the encoding surface. Redplait gives the patching tools and the instruction tables from ptxas. Huerta gives the microarchitectural mechanism that the control codes drive. Yan gives the driver layer below. Each one covers a different piece. What none of them covers is the combination: a systematic study of ptxas heuristic decisions, their SASS-level signatures, their measurable performance implications, across architectures, with the tooling to analyze them at scale.

That is the gap. And closing it starts with the tooling that currently does not exist.

> ⭐ *SASS King is an open project. If any of this resonates with you, the repo is at [github.com/florianmattana/sass-king](https://github.com/florianmattana/sass-king). Open issues describe what needs to be done next, [CONTRIBUTING.md](https://github.com/florianmattana/sass-king/blob/main/CONTRIBUTING.md) explains how to add a dump or a correction, and a star or a watch helps the project reach the people who can contribute hardware I don't have access to.*
