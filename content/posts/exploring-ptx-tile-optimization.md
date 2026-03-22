Originally, I suggested on LinkedIn that I would write an article to unabstract a PyTorch program and reveal some potential CUDA optimizations we could make by debugging it. I actually took another road, but I kept in mind the desire to dig deeper from a given point of abstraction.

Hironically, the first lesson I learned was to talk about what I did rather than what I am up to. A famous French rapper once said, “We never talk about what we did; we only do what we said,” and I felt like I hadn’t really followed his motto. I’ll keep that in mind for the future, even though I still expect you to grab some knowledge from the experience I’m going to share. At the very least, let’s try to have a fun time together.

For context, I am currently writing a small program to compare some kernels with different optimization choices or intentionally unoptimized kernels to highlight the differences between various implementation options.

Article content
Little proof that I am really on it. I don't intend to disappoint the rapper again
Before reaching the conclusion by uploading a terminal screenshot from which we already know the results in advance, I intended to explore the trick at a lower level.

As you might know, a CUDA program is compiled in two stages. First, the NVCC compiler splits your code into two parts: CPU code and GPU code. The CPU code is compiled in the traditional way, while the GPU code is compiled into PTX, which is a sort of intermediate representation between your source code and what the GPU actually runs. The second compilation step turns PTX into SASS, the actual machine code executed by the GPU. This can happen either during compilation (AOT) or at runtime (JIT), but I will try to make another article about those approaches in detail later, if you enjoy this one. For now, let’s focus on PTX itself.

So there I was, and I thought it would be interesting to catch, for example, the difference between a naive kernel implementation and a Matrix Transpose Optimized Kernel with tiling.

In CUDA, a tile is a small block of data that fits into shared memory. This tiling strategy reduces slow global memory accesses by reusing data from the fast shared memory multiple times. At least, that’s what we are going to assess.

If you want to experiment by yourself I am dropping the generic code here and you can use https://godbolt.org/ to get the PTX result:

//KERNEL A : Matrix Transpose Naive

__global__ void matrix_transpose_naive(
    const float* input,
    float* output,
    int width,
    int height
)
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
// KERNEL B : Matrix Transpose Optimized

__global__ void matrix_transpose_optimized(
    const float* input,
    float* output,
    int width,
    int height
)
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
And here are the PTX results we are going to work with :

//Kernel A PTX

.visible .entry matrix_transpose_naive(float const*, float*, int, int)(
	.param .u64 matrix_transpose_naive(float const*, float*, int, int)_param_0,
	.param .u64 matrix_transpose_naive(float const*, float*, int, int)_param_1,
	.param .u32 matrix_transpose_naive(float const*, float*, int, int)_param_2,
	.param .u32 matrix_transpose_naive(float const*, float*, int, int)_param_3
)
{
	ld.param.u64 	%rd1, [matrix_transpose_naive(float const*, float*, int, int)_param_0];
	ld.param.u64 	%rd2, [matrix_transpose_naive(float const*, float*, int, int)_param_1];
	ld.param.u32 	%r3, [matrix_transpose_naive(float const*, float*, int, int)_param_2];
	ld.param.u32 	%r4, [matrix_transpose_naive(float const*, float*, int, int)_param_3];
	mov.u32 	%r5, %ntid.x;
	mov.u32 	%r6, %ctaid.x;
	mov.u32 	%r7, %tid.x;
	mad.lo.s32 	%r1, %r5, %r6, %r7;
	mov.u32 	%r8, %ctaid.y;
	mov.u32 	%r9, %ntid.y;
	mov.u32 	%r10, %tid.y;
	mad.lo.s32 	%r2, %r9, %r8, %r10;
	setp.ge.s32 	%p1, %r1, %r3;
	setp.ge.s32 	%p2, %r2, %r4;
	or.pred  	%p3, %p1, %p2;
	@%p3 bra 	$L__BB0_2;
	cvta.to.global.u64 	%rd3, %rd1;
	mad.lo.s32 	%r11, %r2, %r3, %r1;
	mad.lo.s32 	%r12, %r1, %r4, %r2;
	mul.wide.s32 	%rd4, %r11, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.f32 	%f1, [%rd5];
	cvta.to.global.u64 	%rd6, %rd2;
	mul.wide.s32 	%rd7, %r12, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;
$L__BB0_2:
	ret;
}
//Kernel B PTX

.visible .entry matrix_transpose_optimized(float const*, float*, int, int)(
	.param .u64 matrix_transpose_optimized(float const*, float*, int, int)_param_0,
	.param .u64 matrix_transpose_optimized(float const*, float*, int, int)_param_1,
	.param .u32 matrix_transpose_optimized(float const*, float*, int, int)_param_2,
	.param .u32 matrix_transpose_optimized(float const*, float*, int, int)_param_3
)
{
	ld.param.u64 	%rd1, [matrix_transpose_optimized(float const*, float*, int, int)_param_0];
	ld.param.u64 	%rd2, [matrix_transpose_optimized(float const*, float*, int, int)_param_1];
	ld.param.u32 	%r9, [matrix_transpose_optimized(float const*, float*, int, int)_param_2];
	ld.param.u32 	%r10, [matrix_transpose_optimized(float const*, float*, int, int)_param_3];
	mov.u32 	%r11, %ctaid.x;
	shl.b32 	%r1, %r11, 5;
	mov.u32 	%r2, %tid.x;
	add.s32 	%r3, %r1, %r2;
	mov.u32 	%r12, %ctaid.y;
	shl.b32 	%r4, %r12, 5;
	mov.u32 	%r5, %tid.y;
	add.s32 	%r6, %r4, %r5;
	setp.ge.s32 	%p1, %r3, %r9;
	setp.ge.s32 	%p2, %r6, %r10;
	or.pred  	%p3, %p1, %p2;
	@%p3 bra 	$L__BB0_2;
	cvta.to.global.u64 	%rd3, %rd1;
	mad.lo.s32 	%r13, %r6, %r9, %r3;
	mul.wide.s32 	%rd4, %r13, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.f32 	%f1, [%rd5];
	mov.u32 	%r14, matrix_transpose_optimized(float const*, float*, int, int)::tile;
	mad.lo.s32 	%r15, %r5, 132, %r14;
	shl.b32 	%r16, %r2, 2;
	add.s32 	%r17, %r15, %r16;
	st.shared.f32 	[%r17], %f1;
$L__BB0_2:
	bar.sync 	0;
	add.s32 	%r7, %r4, %r2;
	setp.ge.s32 	%p4, %r7, %r10;
	add.s32 	%r8, %r1, %r5;
	setp.ge.s32 	%p5, %r8, %r9;
	or.pred  	%p6, %p5, %p4;
	@%p6 bra 	$L__BB0_4;
	mov.u32 	%r18, matrix_transpose_optimized(float const*, float*, int, int)::tile;
	mad.lo.s32 	%r19, %r2, 132, %r18;
	shl.b32 	%r20, %r5, 2;
	add.s32 	%r21, %r19, %r20;
	ld.shared.f32 	%f2, [%r21];
	mad.lo.s32 	%r22, %r8, %r10, %r7;
	cvta.to.global.u64 	%rd6, %rd2;
	mul.wide.s32 	%rd7, %r22, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f2;
$L__BB0_4:
	ret;
}
1. PTX, how nice you look.
.visible .entry (rdm_kernel_name)
I’ll try to focus on the most important differences, but as an appetizer, let’s consider that the common first line in both kernels is printed because of the __global__ keyword. This makes the function visible outside the module, much like a public class in traditional programming.

2. Program is setting up...
 .param .u64 matrix_transpose_optimized(float const*, float*, int, int)_param_0,
	.param .u64 matrix_transpose_optimized(float const*, float*, int, int)_param_1,
	.param .u32 matrix_transpose_optimized(float const*, float*, int, int)_param_2,
	.param .u32 matrix_transpose_optimized(float const*, float*, int, int)_param_3
)
{
	ld.param.u64 	%rd1, [matrix_transpose_optimized(float const*, float*, int, int)_param_0];
	ld.param.u64 	%rd2, [matrix_transpose_optimized(float const*, float*, int, int)_param_1];
	ld.param.u32 	%r9, [matrix_transpose_optimized(float const*, float*, int, int)_param_2];
	ld.param.u32 	%r10, [matrix_transpose_optimized(float const*, float*, int, int)_param_3];
Another common point between both PTX kernels which I am sure, shouldn’t surprise anyone is that each element of the kernel signature is stored in a dedicated register: 64-bit registers for pointers, since they store memory addresses (modern GPUs have large memory, up to tens of gigabytes, and 64-bit addresses can reference up to 16 exabytes; a 32-bit pointer could only address 4GB, which is far too limited), and 32-bit registers (4 bytes) for integers, which is what we’ve always learned.

Kernal A

mad.lo.s32 	%r1, %r5, %r6, %r7;
(...)
mad.lo.s32 	%r2, %r9, %r8, %r10;

===
Kernal N 
Before moving on, one last fun fact you may have noticed is the difference in selected registers between the two kernels for the same operation. This difference in register usage between the naïve and optimized kernels is not arbitrary—it’s a direct consequence of compiler-driven register scheduling to support shared memory, tiling, and memory coalescing.

When a kernel uses __shared__ memory, the compiler knows that shared memory will be leveraged to reduce slow global memory accesses and applies additional memory-related optimizations. As a result, it reorganizes register usage to hold indices, offsets, and temporary values needed for shared memory loads and stores.

3. Thread indices calculation
For now, that’s the PTX… half-interesting, right? You could have been watching an exciting tutorial during this time, but I kind of kidnapped you on my not-so-exciting PTX journey. I promise to raise your interest from the very next line, as it marks the first divergence between the two kernels.

KERNEL A (not optimized)	

        mov.u32 	%r5, %ntid.x; 
	mov.u32 	%r6, %ctaid.x;
	mov.u32 	%r7, %tid.x;
	mad.lo.s32 	%r1, %r5, %r6, %r7;
	mov.u32 	%r8, %ctaid.y;
	mov.u32 	%r9, %ntid.y;
	mov.u32 	%r10, %tid.y;
	mad.lo.s32 	%r2, %r9, %r8, %r10;

Kernel B (Optimized)

	mov.u32 	%r11, %ctaid.x;
	shl.b32 	%r1, %r11, 5;
	mov.u32 	%r2, %tid.x;
	add.s32 	%r3, %r1, %r2;
	mov.u32 	%r12, %ctaid.y;
	shl.b32 	%r4, %r12, 5;
	mov.u32 	%r5, %tid.y;
	add.s32 	%r6, %r4, %r5;
These two code snippets calculate thread indices in different ways, revealing an important assumption about block dimensions. I mean, the difference is that :

First version: Flexible, works with any block dimensions, but requires reading blockDim and using a multiply-add instruction.
Second version: Faster (one less instruction, uses shift instead of multiply) but only works if blocks are 32×32.

Why is that ?
First kernel :

Move the block dimension in x (number of threads per block) into register r5
Move the block ID in x-dimension into register r6
Move the thread ID within block into register r7
Multiply r5 by r6, add r7, and store result in r1 (multiply-add operation). 

Key point to remember : This uses the actual runtime block dimension instead of a hardcoded value. This works with any block size. and it translate the very famous formula you might know :

globalX = blockDim.x × blockIdx.x + threadIdx.x
Second kernel :

Move the block ID in x-dimension into register r11
Shift r11 left by 5 bits and store in r1, which multiplies by 32
Move the thread ID within block into register r2
Add r1 and r2, store result in r3, giving the global thread index

Key point to remember : The shift left by 5 bits multiplies by 2⁵ = 32. This assumes the block size is exactly 32, hardcoded at compile time. Hardcoded, why ? Because of the size of the tile we define to 32. Some how of you have noticed that the formula is :

globalX = blockIdx.x × 32 + threadIdx.x
The first conclusion we can draw is that both kernels compute the global thread index, but they do so in different ways, leading to subtle differences at the instruction level.

The first kernel uses the generic formula that requires loading blockDim.x at runtime and performing a multiplication followed by an addition. In practice, this translates to roughly five instructions, typically including a multiply-add. This approach is fully flexible and works with any block size, but it limits certain compile-time optimizations because blockDim.x is not a constant.

The second kernel assumes a fixed block size of 32 and computes. Here, the multiplication is replaced by a left shift of 5 bits, which is simpler and cheaper. This version uses about four instructions, avoids loading blockDim.x, and allows the compiler to apply more aggressive optimizations. However, it is rigid and only valid when the block size is exactly 32.

A left shift is cheaper than a multiplication because it is implemented as simple bit rewiring with minimal hardware and latency, while multiplication requires complex arithmetic logic, deeper pipelines, and higher scheduling and energy costs, even when both appear as a single instruction.(chatGPT).

4. Index bounderies
Kernel A(naive) 

	setp.ge.s32 	%p1, %r1, %r3;
	setp.ge.s32 	%p2, %r2, %r4;
	or.pred  	%p3, %p1, %p2;
	@%p3 bra 	$L__BB0_2;

----

Kernel B(optimized)

	setp.ge.s32 	%p1, %r3, %r9;
	setp.ge.s32 	%p2, %r6, %r10;
	or.pred  	%p3, %p1, %p2;
	@%p3 bra 	$L__BB0_2;
Not too much to say here, I believe, but because I like to talk (or write), I just want you to notice that the structure is, of course, identical when it comes to preventing indices from going out of bounds. We can just observe that different registers are used (r3 vs r9, r6 vs r10), as previously mentioned, in order to:

Reduce register pressure (avoid running out of registers)
Enable better instruction scheduling (hide memory latency)
Improve instruction-level parallelism

5.1 Calculation time for Kernel A
Now we are all set up. It is calculation time ... as a french rapper said. No I am kidding.

If we want to identify the moment when performance diverges even further, we need to check whether the memory access is coalesced. That’s the gold standard of our work. As a quick reminder, memory access is coalesced when threads in a warp access consecutive addresses. For float or int data, for example, each address should be 4 bytes apart.

Let’s first investigate the naïve kernel to see whether its memory accesses are coalesced.

cvta.to.global.u64  %rd3, %rd1;
mad.lo.s32  %r11, %r2, %r3, %r1;
mul.wide.s32  %rd4, %r11, 4;
add.s64  %rd5, %rd3, %rd4;
ld.global.f32  %f1, [%rd5];  // COALESCED READ

====

mad.lo.s32  %r12, %r1, %r4, %r2;
cvta.to.global.u64  %rd6, %rd2;
mul.wide.s32  %rd7, %r12, 4;
add.s64  %rd8, %rd6, %rd7;
st.global.f32  [%rd8], %f1;  // UNCOALESCED WRITE
Keep in mind that we can tell whether memory access is coalesced just by looking at the PTX. Check how threadIdx.x changes the address:

If each thread increments the address by the size of one element (for example, 4 bytes for a float), the access is coalesced.
If each thread jumps by a much larger number, like the matrix height, the access is not coalesced.

Read part for not optimized kernel : 
mad.lo.s32  %r11, %r2, %r3, %r1;
In this instruction:

%r1 corresponds to threadIdx.x
%r2 * %r3 computes a constant base offset (row * width)
threadIdx.x is added directly to the index

This gives an access pattern like:

index(thread i) = constant(r2 * r3) + i(r1)
This means each thread reads the next element in memory.

Write part for optimized kernel :
mad.lo.s32  %r12, %r1, %r4, %r2;
%r1 = threadIdx.x
%r4 = height
%r2 = row

This calculates:

index(thread i) = threadIdx.x * height + row
Each thread jumps by height elements instead of 1.
After multiplying by the element size (4 bytes for float), consecutive threads access memory addresses that are far apart.

Because the addresses are not consecutive, the write is uncoalesced.

5.2.a Calculation time for Kernel B
I won't explain the read part as I am not sure it brings any value. But because you can be curious I have translated some line for you to better understand PTX.

cvta.to.global.u64  %rd3, %rd1;              
// Convert the pointer %rd1 to a global memory address and store it in %rd3

mad.lo.s32  %r13, %r6, %r9, %r3;            
// Compute the element index: r13 = r6 * r9 + r3 (multiply-add)

mul.wide.s32  %rd4, %r13, 4;                 
// Convert the index to a byte offset: offset = index * 4 (since float = 4 bytes)

add.s64  %rd5, %rd3, %rd4;                   
// Compute the final address: base address + offset

ld.global.f32  %f1, [%rd5];                  
// Load the float value from global memory at that address into %f1
5.2b The writting part 
mov.u32  %r14, tile;                       
mad.lo.s32  %r15, %r5, 132, %r14;           
shl.b32  %r16, %r2, 2;                      
add.s32  %r17, %r15, %r16;                 
st.shared.f32  [%r17], %f1; 
My first surprise was to see TILE as it was never declared. I then understood that tile is a __shared__ memory variable declared in the original CUDA code (before compilation to PTX) and it doesn’t appear explicitly as a standard variable in the PTX code. 

But still the most important part at line 2 with the 132.

Why is that ?

In our code , we have : 

__shared__ float tile[32][33];
Considering than : 

Each float is 4 bytes.
The second dimension is 33, not 32.
The stride is the distance in bytes between the start of one row and the start of the next row in memory)

We understand that :

stride in bytes = number of floats per row × size of each float
stride          = 33 floats                × 4 bytes/float     
                =132 bytes
So here is the key. This extra float (33 instead of 32) is used to avoid shared memory bank conflicts.

6. Thread sync
A important divergent point is about thread synchronisation which is mandatory when we use share_memory. Which is what tile is about, actually.

$L__BB0_2:
bar.sync  0; 
When we use a tile in shared memory, all threads in a block write their part of the data into that tile. However, threads may execute at slightly different speeds.

If a thread tries to read from the tile before another thread has finished writing, it could read wrong or incomplete data.

That’s why we use a synchronization barrier (__syncthreads() in CUDA, bar.sync 0 in PTX).

It makes all threads wait until everyone has finished writing to the shared memory.
After the barrier, it’s safe for all threads to read from the tile.

7. A quick wrap up
I won’t go further into this PTX code, as I believe the main takeaways have already been discussed.

Still, I can confidently say that exploring a PTX file is a great way to better understand CUDA code. I’m sure I’ll use this approach more in future debugging sessions it’s a good exercise to become a stronger engineer.

That said, I wouldn’t recommend putting too much focus on PTX alone. Remember, it’s just a low-level translation of your higher-level code. Nothing will appear in PTX that you didn’t already write.

As a quick wrap-up: feel free to improve this article with your comments. I did a lot of research and refined it with the help of LLM, but I could still be wrong. I hope you learned something and enjoyed the journey. If you did, let me know and I’d be happy to write more articles like this.

Thanks for reading GPU team and as a french rapper said...