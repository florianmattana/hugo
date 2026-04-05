---
title: "I Wrote an MXFP4 Quantization Kernel and Ranked #1 on Tensara"
date: 2026-04-05
draft: false
tags: ["CUDA", "quantization", "FP4", "GPU", "Tensara"]
---

## Why I Did This

I'm building an FP4 fused attention kernel for consumer Blackwell GPUs (SM120). That means I spend my days thinking about how to squeeze 32-bit numbers into 4 bits without losing too much information.

[Tensara](https://tensara.org) is a platform where you submit GPU kernels and compete on real hardware. They had an MXFP4 quantization problem with almost no submissions. I figured: I already know this format inside out on SM120, how hard can it be to write a standalone quantization kernel?

Turns out the kernel itself was straightforward. What surprised me were the subtle details that took hours to get right. This post walks through everything, step by step, assuming you've never heard of FP4 or quantization before.

---

## 1. The Problem: Numbers Are Too Big

A modern AI model like Llama has billions of parameters. Each parameter is a 32-bit floating point number, that's 4 bytes. A 7-billion parameter model takes 28 GB just to store the weights. That's more than most GPUs can hold.

We need to make these numbers smaller. Not fewer numbers, smaller numbers. Instead of 32 bits per number, what if we used 4 bits? That's 8 times less memory. A 28 GB model becomes 3.5 GB.

The catch: with 4 bits, you can only represent 16 different values. With 32 bits, you can represent about 4 billion different values. So we're going from 4 billion choices down to 16. We're going to lose information. The question is: how do we lose as little as possible?

This is what quantization is. The full process has three steps:

1. **Scale**: for each group of 32 values, compute a scale factor that brings those values into a range that 4 bits can represent.
2. **Encode**: divide each value by its group's scale, then round to the nearest of the 16 representable FP4 values. This is the encoder.
3. **Pack**: two 4-bit values fit in a single 8-bit byte. We pack them together to save space.

Each step has its own pitfalls. This post covers all three, plus the GPU kernel that runs them in parallel on over a million groups simultaneously.

---

## 2. FP4 E2M1: The Format

The format we're using is called FP4 E2M1. The name tells you the bit layout: 1 sign bit, 2 exponent bits, 1 mantissa bit. Total: 4 bits.

With these 4 bits, you can represent exactly these magnitudes:

```text
0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
```

And their negative versions. That's it. Your number 0.8734 has to become one of these. Your number -147.3 also has to become one of these.

The spacing isn't uniform. The gap between 0 and 0.5 is 0.5, but the gap between 4.0 and 6.0 is 2.0. This is by design. Floating point formats have more precision near zero and less precision for large values.

### Other FP4 formats

E2M1 is not the only way to split 4 bits. There are other formats, each with different trade-offs:

**FP4 E3M0** uses 3 exponent bits and 0 mantissa bits. No mantissa means you can only represent exact powers of 2: 0.25, 0.5, 1, 2, 4, 8, 16. Much wider range (up to 16 instead of 6) but no precision between powers of 2. The value 1.5 simply doesn't exist in this format. Rarely used in practice because the gaps are too large.

**NF4** (NormalFloat4) takes a completely different approach. Instead of using the sign/exponent/mantissa structure, it picks 16 values that match the statistical distribution of neural network weights, which tend to follow a bell curve. The values are irregularly spaced, clustered where weights are most likely to appear. Used by QLoRA for model fine-tuning. No hardware acceleration on current GPUs, it's purely a software format.

E2M1 has become the standard for hardware-accelerated FP4 because it offers the best balance between range and precision for typical AI workloads. Both the industry-wide open standard (MXFP4, which we'll cover in section 4) and NVIDIA's proprietary format (NVFP4) use E2M1 for the data itself. They differ in how they handle the scaling, which we'll get to next.

---

## 3. The Range Problem

Look at the E2M1 values again. The biggest one is 6.0. What do you do with a number like 150?

You can't represent it. 6.0 is the max. Everything above 6.0 gets clamped to 6.0.

This is terrible. If your data ranges from -200 to +200, everything gets crushed into [-6, 6] and you lose almost all information.

---

## 4. MXFP4 and Block Scaling

The first step of quantization is to divide your numbers by a scale factor before encoding them. If your data goes up to 150, a scale of 32 brings it down to `150 / 32 = 4.69`, which fits in the FP4 range. You store the scale alongside the data so you can reverse the division later and recover an approximation of the original values.

But one scale for the entire matrix is too coarse. Imagine one region of your matrix has values around 0.001 and another region has values around 100. A scale that works for 100 will crush the 0.001 values to zero.

The solution: split the matrix into groups of 32 consecutive elements along each row. Each group gets its own scale, calculated from the largest absolute value in that group. The groups with small values get a small scale (preserving their precision), the groups with large values get a large scale (preventing overflow).

For a matrix of M rows and K columns, you get `M x (K/32)` scale values. For example, a matrix of 1024 rows and 4096 columns produces `1024 x (4096/32) = 1024 x 128 = 131,072` scale values. That's 131,072 bytes (128 KB) of overhead, compared to 16 MB for the original matrix in FP32. Less than 1% overhead.

### MXFP4: the standard

This block scaling scheme, combined with the E2M1 data format and specific rules for rounding and scale computation, forms a complete specification called **MXFP4** (Microscaling FP4).

MXFP4 is published by the **OCP (Open Compute Project)**. The OCP is a consortium of tech companies, including Meta, Microsoft, AMD, Intel, and NVIDIA, that defines open hardware and software standards. "Open" means anyone can implement them without licensing fees. The goal is interoperability: a model quantized in MXFP4 can run on any hardware that supports the standard, not just one vendor's chips.

Why not just use NVIDIA's format on NVIDIA hardware? NVIDIA does have their own proprietary format called **NVFP4**. The data is still E2M1, but the scaling works differently: NVFP4 uses two levels of scale factors (a coarse per-tensor scale and a fine per-block scale with blocks of 16 elements) instead of MXFP4's single scale per block of 32. This gives NVFP4 better dynamic range, but it's tied to NVIDIA hardware.

NVIDIA's Blackwell GPUs (the B200 that Tensara uses for benchmarking) support both MXFP4 and NVFP4. The Tensara problem asks for MXFP4 specifically, because it uses **TorchAO** (PyTorch's quantization library) as the reference implementation, and TorchAO implements the OCP standard.

This matters for verification: Tensara takes my output, dequantizes it using TorchAO's code, does the same with TorchAO's own quantization, and compares the two results. If my scale computation or rounding doesn't match TorchAO's behavior exactly, the test fails. So understanding how TorchAO implements the spec is just as important as understanding the spec itself.

---

## 5. E8M0: Why the Scale Must Be a Power of 2

The scale is stored in a format called E8M0: 8 exponent bits, 0 mantissa bits. No mantissa means the scale is always an exact power of 2: 1, 2, 4, 8, or going the other way, 0.5, 0.25, 0.125, and so on.

Why force powers of 2? Because GPUs can multiply by a power of 2 by just shifting the exponent. It's essentially free in hardware. A scale of 3.7 would require a real floating-point multiplication. A scale of 4.0 = 2^2 is a bit shift. When your Tensor Core is doing billions of these operations per second, "free" matters.

### The bias: a thermometer trick

The exponent stored in E8M0 can be negative. A scale of 0.125 is 2^(-3), so the exponent is -3. But we're storing this in a single byte (values 0 to 255). How do you fit a negative number in there?

Think of a thermometer. Temperature can go below zero (-30°C, -10°C), but the markings on the physical tube start at the bottom and go up. A thermometer that reads from -127°C to +128°C could just shift everything up by 127: the bottom of the tube is labeled 0 (meaning -127°C), the middle is labeled 127 (meaning 0°C), and the top is labeled 254 (meaning +127°C). To read the real temperature, subtract 127 from the tube reading.

E8M0 does exactly this. The real exponent can range from -127 to +128, but instead of storing negative numbers, we add 127 to everything. This offset is called the **bias**.

```text
stored_value = real_exponent + 127
real_exponent = stored_value - 127
scale = 2^(stored_value - 127)
```

Some examples:

| Stored value | Calculation | Real exponent | Scale |
|:---:|:---:|:---:|:---:|
| 120 | 120 - 127 | -7 | 2^(-7) = 0.0078 |
| 124 | 124 - 127 | -3 | 2^(-3) = 0.125 |
| 127 | 127 - 127 | 0 | 2^0 = 1.0 |
| 130 | 130 - 127 | 3 | 2^3 = 8.0 |
| 137 | 137 - 127 | 10 | 2^10 = 1024.0 |

The regular 32-bit floats that every computer uses also store their exponent with a bias of 127. It's the same trick, and we'll use that fact later to extract the exponent cheaply from a float's binary representation.

---

## 6. Computing the Scale: The Part That Took Hours

This is where the engineering got interesting. Here's the logic:

1. Find the biggest absolute value in your block of 32 numbers. Call it `amax`.
2. You want `amax / scale` to fit inside the FP4 range, which maxes out at 6.0.
3. So you want `scale ≈ amax / 6.0`.
4. But scale must be a power of 2, so you pick the nearest power of 2.

Let's work through a concrete example. Say your block of 32 values has a max absolute value of `amax = 25.0`. You want `scale ≈ 25.0 / 6.0 ≈ 4.17`. The nearest powers of 2 are 4.0 (below) and 8.0 (above). Which do you pick?

### Floor vs ceil: a real trade-off

**If you pick 4.0 (rounding down, called "floor"):** `25.0 / 4.0 = 6.25`. That exceeds the FP4 max of 6.0, so the value 25.0 gets clamped to 6.0. When you dequantize later, you recover `6.0 x 4.0 = 24.0` instead of 25.0. Error on this value: 1.0.

Now take a smaller value in the same block, say 1.0. With scale 4.0: `1.0 / 4.0 = 0.25`. That rounds to FP4 value 0.5 (the nearest representable value). Dequantized: `0.5 x 4.0 = 2.0`. Error: 1.0.

**If you pick 8.0 (rounding up, called "ceil"):** `25.0 / 8.0 = 3.125`, which rounds to FP4 value 3.0. Dequantized: `3.0 x 8.0 = 24.0`. Error: 1.0. Same as floor for the max value.

But now the value 1.0 becomes `1.0 / 8.0 = 0.125`. That rounds to FP4 value 0.0 (since 0.125 is below the midpoint 0.25 between 0.0 and 0.5). Dequantized: `0.0 x 8.0 = 0.0`. Error: 1.0. The value 1.0 was completely erased to zero.

With the floor scale (4.0), that same value 1.0 became 2.0. Not perfect, but the information is preserved. With the ceil scale (8.0), it became 0.0. Gone.

The pattern: with a larger scale, small values get pushed toward zero more aggressively. The OCP standard uses floor. Sacrifice accuracy on the one or two extreme values in the block (they get clamped to 6.0), but give every other value in the block the best precision possible.

### Getting floor(log2) for free from the float bits

In code, "floor of log2" is surprisingly easy to compute. You don't need a logarithm function at all.

Every 32-bit float is stored in memory as 32 bits: 1 sign bit, 8 exponent bits, 23 mantissa bits. The computer represents every float as `mantissa x 2^exponent`, where the mantissa is always between 1.0 and 2.0. The exponent tells you the "order of magnitude" in powers of 2.

This means `log2(value) = exponent + log2(mantissa)`. Since the mantissa is between 1.0 and 2.0, `log2(mantissa)` is between 0 and 1. So `floor(log2(value))` is simply the exponent. It's already sitting there in the float's bits.

The exponent is stored with a bias of 127 (the same thermometer trick from section 5), so to extract it: read the exponent bits, subtract 127.

Two examples:

**amax = 0.945**

```text
0.945 in memory = 1.890 x 2^(-1)
The mantissa is 1.890 (between 1.0 and 2.0, good)
The exponent is -1, stored as -1 + 127 = 126
We extract 126, subtract bias: 126 - 127 = -1
floor(log2(0.945)) = -1
Check: 2^(-1) = 0.5 ≤ 0.945 < 1.0 = 2^0. Correct.
```

**amax = 25.0**

```text
25.0 in memory = 1.5625 x 2^4
The mantissa is 1.5625 (between 1.0 and 2.0, good)
The exponent is 4, stored as 4 + 127 = 131
We extract 131, subtract bias: 131 - 127 = 4
floor(log2(25.0)) = 4
Check: 2^4 = 16 ≤ 25.0 < 32 = 2^5. Correct.
```

### The full scale calculation

```cpp
unsigned int bits = __float_as_uint(max_abs);
int max_exp = (int)((bits >> 23) & 0xFF) - 127;
int scale_exp = max_exp - 2;
int biased = scale_exp + 127;
```

Line by line:

**Line 1:** `__float_as_uint(max_abs)` reads the raw 32 bits of the float as an unsigned integer. We're not converting the value, we're reading the same bits with a different interpretation. Like reading a French word as if it were English: the letters are the same, but you interpret them differently.

**Line 2:** `(bits >> 23) & 0xFF` shifts right by 23 positions to move the exponent bits (bits 23-30) down to bits 0-7, then masks with `0xFF` to keep only those 8 bits. Subtracting 127 removes the bias. Result: `floor(log2(max_abs))`.

**Line 3:** Subtract 2. This is the key step. We want `scale ≈ amax / 6.0`, and we're working in powers of 2. The FP4 E2M1 format can represent values up to 6.0. How many powers of 2 does it take to reach 6? `2^2 = 4` is the largest power of 2 at or below 6. So the FP4 format "covers" 2 powers of 2 on its own.

If your input needs 4 powers of 2 to represent (say `amax ≈ 25`, so `floor(log2(25)) = 4`), and FP4 already covers 2, then the scale needs to cover the remaining `4 - 2 = 2` powers of 2. So `scale = 2^2 = 4`.

**Line 4:** Add 127 to put the result back into the biased E8M0 format for storage.

Full trace for `amax = 0.945`:

```text
max_exp = floor(log2(0.945)) = -1
scale_exp = -1 - 2 = -3
biased = -3 + 127 = 124
scale = 2^(124 - 127) = 2^(-3) = 0.125
Check: 0.945 / 0.125 = 7.56, clamped to FP4 max (6.0)
Dequantized: 6.0 x 0.125 = 0.75 (error: 0.195 on the max value)
```

Full trace for `amax = 25.0`:

```text
max_exp = floor(log2(25.0)) = 4
scale_exp = 4 - 2 = 2
biased = 2 + 127 = 129
scale = 2^(129 - 127) = 2^2 = 4.0
Check: 25.0 / 4.0 = 6.25, clamped to FP4 max (6.0)
Dequantized: 6.0 x 4.0 = 24.0 (error: 1.0 on the max value)
```

### The wrong turns

Getting the scale right took four attempts. Each failure taught me something about how the spec actually works.

#### Attempt 1: frexpf

The C standard library provides a function called `frexpf`. It takes a float and splits it into a mantissa and an exponent, similar to what the float's bits encode internally. For example, `frexpf(0.1576)` returns `m = 0.6304` and `exp = -2`, such that `0.6304 x 2^(-2) = 0.1576`.

I used it to compute `floor(log2(amax / 6.0))` directly. The idea seemed clean: call frexpf, get the exponent, done.

The problem: frexpf defines its mantissa as being between 0.5 and 1.0, not between 1.0 and 2.0 like the float's internal representation. This means frexpf's exponent is always one higher than `floor(log2(x))`. For `0.1576`: frexpf gives `exp = -2`, but `floor(log2(0.1576)) = -3` (because `2^(-3) = 0.125 ≤ 0.1576 < 0.25 = 2^(-2)`). You need to subtract 1 from frexpf's exponent to get the floor.

I did that, but juggling two different conventions (frexpf's [0.5, 1.0) vs the float's internal [1.0, 2.0)) made the code confusing. And that confusion led directly to the next mistake.

*Result: correct output, but fragile code.*

#### Attempt 2: accidental ceil

While modifying the frexpf version, I accidentally removed the `exp -= 1` adjustment. This changed the scale from floor to ceil, making it one power of 2 too large.

With the correct scale 0.125 (floor): `0.945 / 0.125 = 7.56`, clamped to 6.0. With the wrong scale 0.25 (ceil): `0.945 / 0.25 = 3.78`, rounds to FP4 value 4.0.

Every value in the block was divided by a scale twice as large as needed, so all the FP4 nibbles came out wrong. The test failed completely. Every single byte was different from the expected output.

*Result: complete failure. One power of 2 off changes everything.*

#### Attempt 3: the safety check

I went back to the floor version (with `exp -= 1`) but added a safety net. After computing the scale, I checked: does `amax / scale` exceed 6.0? If so, bump the scale up by one power of 2.

For `amax = 0.945`, scale = 0.125: `0.945 / 0.125 = 7.56 > 6.0`. So my code bumped the scale to 0.25. Stored value: 125 instead of 124.

But TorchAO (the reference that Tensara verifies against, as explained in section 4) expects 124. It uses floor and accepts the clamping. My "safety" produced a different scale, which changed every output byte.

This was the most frustrating attempt because preventing overflow feels like the right engineering instinct. But the spec deliberately allows overflow. The clamping is a feature, not a bug.

*Result: wrong answer. Scale was 125 everywhere, expected was 124.*

#### Attempt 4: direct bit extraction

I abandoned frexpf entirely and extracted the exponent directly from the float's binary representation:

```cpp
unsigned int bits = __float_as_uint(max_abs);
int max_exp = (int)((bits >> 23) & 0xFF) - 127;
```

No ambiguity about mantissa conventions. No off-by-one adjustments. The float's exponent bits are literally `floor(log2)` with a bias. Extract and subtract.

*Result: correct scale on all test cases. Time to move on to the encoder.*

---

## 7. The Encoder: Turning a Float into 4 Bits

The scale is computed and validated. We divide each value by the scale. The result is a float somewhere in the range that FP4 can represent. For example, if the original value is 0.6 and the scale is 0.125, we get `0.6 / 0.125 = 4.8`.

Now we need to choose which of the 8 FP4 magnitudes (0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0) is the closest to 4.8. In this case it's 4.0 (nibble 6) or 6.0 (nibble 7). The midpoint between 4.0 and 6.0 is 5.0. Since 4.8 < 5.0, we pick 4.0. The nibble is 6.

That's the encoder: for each scaled value, find the nearest FP4 value.

The midpoints between consecutive FP4 values are the decision boundaries:

```text
FP4 values:  0    0.5    1.0    1.5    2.0    3.0    4.0    6.0
Nibble:      0     1      2      3      4      5      6      7
                 |      |      |      |      |      |      |
Midpoints:     0.25   0.75   1.25   1.75   2.5    3.5    5.0
```

If a value falls between two midpoints, the nearest FP4 value is unambiguous. The code checks each midpoint from top to bottom:

```cpp
if      (abs_val >  5.0)  nibble = 7;  // represents 6.0
else if (abs_val >= 3.5)  nibble = 6;  // represents 4.0
else if (abs_val >  2.5)  nibble = 5;  // represents 3.0
else if (abs_val >= 1.75) nibble = 4;  // represents 2.0
else if (abs_val >  1.25) nibble = 3;  // represents 1.5
else if (abs_val >= 0.75) nibble = 2;  // represents 1.0
else if (abs_val >  0.25) nibble = 1;  // represents 0.5
else                      nibble = 0;  // represents 0.0
```

### The >= vs > puzzle

Look carefully: some thresholds use `>` (strictly greater) and some use `>=` (greater or equal). For most values, it doesn't matter. If your value is 2.7, it's between midpoints 2.5 and 3.5, and both `>` and `>=` give the same answer: nibble 5 (represents 3.0).

The difference only shows up when a value lands exactly on a midpoint. When that happens, the value is equidistant from two FP4 values. You have to pick one.

The MXFP4 spec uses **round-to-nearest-even**: when there's a tie, pick the nibble with an even index. This is the same rule that every computer uses for regular floating point arithmetic. It exists to prevent statistical bias: always rounding up (or always down) would systematically shift your values in one direction over large datasets.

Let's work through one midpoint in detail to see how this translates to code.

Take the midpoint **0.75**. It's the point exactly between the FP4 value 0.5 (nibble 1) and the FP4 value 1.0 (nibble 2). If your scaled value is 0.74, it's closer to 0.5, you pick nibble 1. If it's 0.76, it's closer to 1.0, you pick nibble 2. No problem.

But if it's exactly 0.75, it's at equal distance from both. Nibble 1 is odd, nibble 2 is even. Round-to-nearest-even says: pick nibble 2 (the even one). In the code, this means the threshold 0.75 must use `>=` so that the exact value 0.75 enters the `nibble = 2` branch.

Now take the midpoint **1.25**. It's between nibble 2 (even) and nibble 3 (odd). Round-to-nearest-even says: pick nibble 2 (the even one). This time we want the exact value 1.25 to NOT enter the `nibble = 3` branch, so we use `>` instead of `>=`.

The rule for each midpoint:

| Midpoint | Below | Above | Even pick | Operator | Example: exact midpoint becomes |
|:---:|:---:|:---:|:---:|:---:|:---|
| 0.25 | nibble 0 (even) | nibble 1 (odd) | 0 | `>` | 0.25 becomes nibble 0 (value 0.0) |
| 0.75 | nibble 1 (odd) | nibble 2 (even) | 2 | `>=` | 0.75 becomes nibble 2 (value 1.0) |
| 1.25 | nibble 2 (even) | nibble 3 (odd) | 2 | `>` | 1.25 becomes nibble 2 (value 1.0) |
| 1.75 | nibble 3 (odd) | nibble 4 (even) | 4 | `>=` | 1.75 becomes nibble 4 (value 2.0) |
| 2.5 | nibble 4 (even) | nibble 5 (odd) | 4 | `>` | 2.5 becomes nibble 4 (value 2.0) |
| 3.5 | nibble 5 (odd) | nibble 6 (even) | 6 | `>=` | 3.5 becomes nibble 6 (value 4.0) |
| 5.0 | nibble 6 (even) | nibble 7 (odd) | 6 | `>` | 5.0 becomes nibble 6 (value 4.0) |

The pattern: `>=` when the upper nibble is even, `>` when it's odd.

### The debugging journey

This detail took the longest to get right. The progression of errors shows how small the gap is between "almost correct" and "correct".

#### Encoder v1: >= everywhere

My first instinct: use `>=` at every threshold. Clean, uniform, easy to read.

Tensara result: **wrong answer, maximum difference 0.375.** At this stage I was still debugging the scale at the same time, so this error came from both issues combined. But it told me the rounding was wrong on at least some boundary values.

#### Encoder v2: lookup table with strict <

I replaced the threshold checks with a brute-force approach: store all 8 FP4 values in an array, compute the distance from the input to each one, and pick the closest.

```cpp
const float fp4_vals[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
uint8_t best = 0;
float best_dist = x;
for (int i = 1; i < 8; i++) {
    float dist = fabsf(x - fp4_vals[i]);
    if (dist < best_dist) {
        best_dist = dist;
        best = i;
    }
}
```

When two values are equally close (a tie), the `<` means the earlier candidate (lower nibble) wins.

Tensara result: **wrong answer, maximum difference 0.0625.** Much better, from 0.375 down to 0.0625. The remaining error was exactly one FP4 step (0.5) times the scale (0.125). One nibble was off by one position on a tie. The `<` tie-break always picks the lower nibble, but round-to-nearest-even sometimes wants the upper nibble (when the upper one has an even index).

#### Encoder v3: round-to-nearest-even

I went back to the threshold approach but analyzed each midpoint individually using the table above. Every tie goes to the even nibble: some thresholds use `>=`, others use `>`.

Tensara result: **accepted, maximum difference 0.0.** Seven characters changed from v1 (some `>=` became `>` and vice versa). But those seven characters are the difference between matching the spec and not.

---

## 8. Packing: Two Values per Byte

The third and last step. A nibble is 4 bits. A byte is 8 bits. We pack two FP4 values into one byte:

```cpp
byte = (nibble_odd << 4) | nibble_even;
```

The even-indexed element goes in the low 4 bits (bits 0-3), the odd-indexed element goes in the high 4 bits (bits 4-7). For a block of 32 FP4 values, that's 16 packed bytes.

The packing order matters. If you swap even and odd, the dequantization produces garbage. On my SM120 kernel, I use a different packing format (one FP4 in an 8-bit container with 4 bits of padding). For MXFP4 OCP standard, it's two FP4 per byte with no padding.

---

## 9. The First Kernel: One Thread per Block

Now we need to write the actual GPU code. The three steps above (scale, encode, pack) describe what to do for one block of 32 elements. But a real matrix has over a million blocks. We need to run them all in parallel.

I knew from the start that the first kernel would not be the fastest. The goal of a first implementation is never performance, it's correctness. A simple kernel where each thread does everything alone is easier to debug: if the output is wrong, the bug is in the algorithm (scale or encoder), not in the parallelism. Once all tests pass, we know the logic is correct and we can optimize the thread-to-data mapping without worrying about confusing an algorithm bug with a parallelism bug.

### Mapping the problem to the GPU

The matrix has M rows and K columns. Each row is split into blocks of 32 elements. The total number of blocks is `M x (K / 32)`. For a 4096 x 8192 matrix, that's `4096 x 256 = 1,048,576` blocks. Each block is completely independent: it reads its own 32 values, computes its own scale, and produces its own 16 output bytes (32 FP4 values packed 2 per byte). No block needs to communicate with any other.

This maps naturally to a GPU. We launch one thread per block:

```cpp
int total_blocks = m * (k / 32);        // 1,048,576 for a 4096x8192 matrix
int threads_per_cuda_block = 256;
int grid = (total_blocks + 255) / 256;
kernel<<<grid, threads_per_cuda_block>>>(...);
```

A note on naming: "block" means two different things here. In the quantization context, a "block" is 32 data elements that share a scale. In the CUDA context, a "block" is a group of threads that are assigned to the same processor (SM) on the GPU, share a fast local memory (shared memory), and can synchronize with each other. I'll say "data block" for the 32 elements and "CUDA block" for the thread group.

Inside the kernel, each thread figures out which data block it's responsible for:

```cpp
int bid = blockIdx.x * blockDim.x + threadIdx.x;
int row       = bid / num_blocks_per_row;
int col_block = bid % num_blocks_per_row;
int col_start = col_block * 32;
```

Then the thread runs all three steps of the pipeline alone:

1. Reads 32 float values from global memory (128 bytes, one load at a time)
2. Loops through all 32 to find the max absolute value, computes the scale
3. Loops through all 32 again, divides each by the scale, encodes to FP4
4. Packs pairs into bytes and writes 16 bytes of output + 1 byte of scale

This works. All four test cases passed. But the benchmarks told a clear story:

| Matrix size | My kernel | #1 (Triton) |
|:---:|:---:|:---:|
| 1024 x 1024 | 23 μs | 65 μs |
| 2048 x 2048 | 49 μs | 55 μs |
| 4096 x 8192 | 285 μs | 89 μs |
| 8192 x 4096 | 282 μs | 98 μs |

On small matrices, I was faster. On large matrices, 3x slower. The arithmetic is the same regardless of size. What changes is the memory access pattern.

### Why it's slow on large matrices

The bottleneck on large matrices is memory access. GPUs execute threads in groups of 32 called **warps**. When all 32 threads in a warp read from consecutive addresses, the memory controller merges everything into a single 128-byte transaction. This is called **coalesced access**.

In my naive kernel, each thread handles a different data block. Thread 0 reads from column 0, thread 1 reads from column 32, thread 2 from column 64. These threads are in the same warp, but their addresses are 128 bytes apart (32 floats x 4 bytes). The memory controller can't merge those into one transaction. Each read is served separately.

On small matrices, the cache absorbs the penalty. On large matrices, every scattered read hits global memory at full latency (hundreds of cycles), and performance collapses.
---

## 10. The Optimized Kernel: One Warp per Block

The algorithm is correct. All tests pass. Now we change nothing about the logic. We only change how the work is distributed across threads so that memory access matches what the hardware can do efficiently.

The fix comes from noticing that a warp has exactly 32 threads, and a data block has exactly 32 elements. What if we assign one warp to one data block, where each thread handles exactly one element?

```cpp
int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
int lane    = threadIdx.x & 31;

float val = a[row * k + col_start + lane];
```

Thread 0 reads element 0, thread 1 reads element 1, thread 2 reads element 2, all the way to thread 31 reading element 31. All 32 addresses are consecutive in memory (4 bytes apart). The memory controller combines them into a single 128-byte transaction. One memory request instead of 32.

### Warp shuffle for the max

In the naive kernel, one thread loops over 32 values to find the max. In the warp version, each thread holds one value. We need to find the max across all 32 threads without using shared memory.

Warp shuffles let threads exchange values directly through their registers. The instruction `__shfl_down_sync(mask, val, offset)` sends each thread's value to the thread `offset` positions below:

```cpp
float max_abs = fabsf(val);
for (int offset = 16; offset > 0; offset >>= 1)
    max_abs = fmaxf(max_abs, __shfl_down_sync(0xFFFFFFFF, max_abs, offset));
max_abs = __shfl_sync(0xFFFFFFFF, max_abs, 0);
```

First iteration (offset = 16): thread 0 receives thread 16's value and keeps the max of the two. Thread 1 receives thread 17's value. After this step, threads 0-15 each hold the max of a pair.

Second iteration (offset = 8): thread 0 receives thread 8's value (which is already the max of a pair). Now threads 0-7 each hold the max of four values.

After 5 iterations (log2(32) = 5), thread 0 holds the max of all 32 values. The final `__shfl_sync` broadcasts thread 0's result to all 32 threads so they all know the scale.

Total cost: 5 register-to-register operations. No memory access, no synchronization barriers.

### Warp shuffle for packing

Each even-numbered thread needs its odd neighbor's nibble to pack two FP4 values into one byte:

```cpp
uint8_t partner_nibble = __shfl_xor_sync(0xFFFFFFFF, nibble, 1);
```

This swaps values between adjacent pairs: thread 0 gets thread 1's nibble, thread 2 gets thread 3's. Then only even threads write the packed byte:

```cpp
if ((lane & 1) == 0) {
    uint8_t byte = (partner_nibble << 4) | nibble;
    q[warp_id * 16 + lane / 2] = byte;
}
```

16 threads write 16 consecutive bytes. Coalesced again.

### The result

| Matrix size | Naive | Warp | #1 (Triton) |
|:---:|:---:|:---:|:---:|
| 1024 x 1024 | 23 μs | 32 μs | 65 μs |
| 2048 x 2048 | 49 μs | 39 μs | 55 μs |
| 4096 x 8192 | 285 μs | 201 μs | 89 μs |
| 8192 x 4096 | 282 μs | 290 μs | 98 μs |

The warp version is slower on 1024x1024. The naive kernel launches 1,024 threads (one per data block). The warp kernel launches 32,768 threads (32 per data block). The extra threads have overhead: register allocation, warp scheduling. When the computation per data block is tiny, this overhead matters. On large matrices, the coalesced memory access more than compensates.

The final optimization was replacing the lookup-table encoder (a loop of 8 distance comparisons, each calling `fabsf`) with the direct threshold encoder (7 branches, no loop). This reduced the instruction count per thread and brought the overall average down.

Final result across all test cases: **72.48 μs. First place, ahead of the Triton kernel at 75.12 μs.**

---

## 11. What I Learned

**The spec matters more than the code.** I spent more time understanding how TorchAO computes the scale and rounds the values than writing the actual CUDA kernel. A one-bit difference in the scale exponent changes every single output value.

**Floor vs ceil is a design choice, not a bug.** The OCP spec deliberately uses floor for the scale, accepting that some values overflow and get clamped. This maximizes precision for the majority of values at the cost of clipping the extremes. My instinct was to prevent overflow, which was wrong.

**Round-to-nearest-even is everywhere.** I knew this rule from standard floating point arithmetic but didn't expect it in a 4-bit format with only 16 values. The difference between `>=` and `>` at each threshold is invisible in 99.999% of cases, but Tensara's verification caught it.

**Coalesced memory access is as often the single biggest optimization.** Going from "one thread reads 32 values" to "32 threads each read one value" was the difference between 285 μs and 72 μs on large matrices. The arithmetic was identical. Only the memory access pattern changed.

**Correctness first, always.** The naive kernel was not the fastest, but it was the easiest to debug. Every optimization I applied afterward changed zero lines of algorithm code. The scale, encoder, and packing logic stayed identical. Only the thread-to-data mapping changed.

There is still room to go faster. Vectorized loads (reading 4 floats at once with `float4`), better occupancy tuning, and register pressure optimization are all on the table. And the same approach applies to the next problems on the Tensara leaderboard: MXFP4 GEMM, NVFP4 quantization, and MXFP8 quantization.
