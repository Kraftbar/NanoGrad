/*
 * NanoGrad's GPU kernels: the twin of kernels.c for NVIDIA GPUs.
 *
 * kernels_cuda.py compiles this file at run time with NVRTC (for this exact
 * GPU, cached on disk) and launches the kernels through the CUDA driver.
 * Every function here has a twin of the same name in kernels.c, which
 * computes the same thing on the CPU; test_kernels_cuda.py checks that the
 * two agree.
 *
 * Conventions, as in kernels.c: arrays are flat row-major `real` numbers
 * (double, or float with -DREAL=float); index tables are `long`; scalar
 * settings (eps, factors, learning rates) are double; counts are long.
 *
 * How a GPU kernel differs from its C twin. The C loop
 *
 *     for (long i = 0; i < count; i++) out[i] = x[i] + y[i];
 *
 * becomes one GPU *thread* per i. Threads come in blocks (256 here), and
 * kernels_cuda.py launches enough blocks to cover `count`; each thread
 * works out its own i from its block's index and its index in the block,
 * and the last block's spare threads do nothing. Thousands of threads run
 * at once, so a kernel is fast when every thread has independent work and
 * neighbouring threads read neighbouring memory (then a group of 32
 * threads, a *warp*, loads one contiguous run in a single transaction).
 * Sums across threads (a row's maximum, a weight gradient over the batch)
 * need cooperation: the threads of a block share fast on-chip *shared
 * memory* and can wait for each other (__syncthreads), but different
 * blocks cannot, so a sum over everything is split into per-block sums.
 */

#ifndef REAL
#define REAL double
#endif
typedef REAL real;

#define KERNEL extern "C" __global__ void

/* This thread's index among all the launched threads. */
__device__ __forceinline__ long thread_index()
{
    return (long)blockIdx.x * blockDim.x + threadIdx.x;
}

/* ------------------------------------------------------------------------
 * Elementwise: one thread per value.
 * --------------------------------------------------------------------- */

KERNEL add_vectors(const real *x, const real *y, real *out, long count)
{
    long i = thread_index();
    if (i < count) out[i] = x[i] + y[i];
}

KERNEL multiply_vectors(const real *x, const real *y, real *out, long count)
{
    long i = thread_index();
    if (i < count) out[i] = x[i] * y[i];
}

KERNEL scale_vector(const real *x, double factor, real *out, long count)
{
    long i = thread_index();
    if (i < count) out[i] = (real)(x[i] * factor);
}

KERNEL add_scaled(const real *x, const real *y, double factor, real *out, long count)
{
    long i = thread_index();
    if (i < count) out[i] = (real)(x[i] + factor * y[i]);
}

KERNEL reciprocal_vector(const real *x, real *out, long count)
{
    long i = thread_index();
    if (i < count) out[i] = (real)1 / x[i];
}

KERNEL exp_vector(const real *x, real *out, long count)
{
    long i = thread_index();
    if (i < count) out[i] = exp(x[i]);
}

KERNEL log_vector(const real *x, real *out, long count)
{
    long i = thread_index();
    if (i < count) out[i] = log(x[i]);
}

KERNEL sqrt_vector(const real *x, real *out, long count)
{
    long i = thread_index();
    if (i < count) out[i] = sqrt(x[i]);
}

KERNEL relu_forward(const real *x, real *out, long count)
{
    long i = thread_index();
    if (i < count) out[i] = x[i] > 0 ? x[i] : (real)0;
}

KERNEL relu_backward(const real *x, const real *grad, real *dx, long count)
{
    long i = thread_index();
    if (i < count) dx[i] = x[i] > 0 ? grad[i] : (real)0;
}

KERNEL tanh_forward(const real *x, real *out, long count)
{
    long i = thread_index();
    if (i < count) out[i] = tanh(x[i]);
}

KERNEL tanh_backward(const real *y, const real *grad, real *dx, long count)
{
    long i = thread_index();
    if (i < count) dx[i] = grad[i] * ((real)1 - y[i] * y[i]);
}

KERNEL sigmoid_forward(const real *x, real *out, long count)
{
    long i = thread_index();
    if (i < count) out[i] = (real)1 / ((real)1 + exp(-x[i]));
}

KERNEL sigmoid_backward(const real *y, const real *grad, real *dx, long count)
{
    long i = thread_index();
    if (i < count) dx[i] = grad[i] * y[i] * ((real)1 - y[i]);
}

/* sum_repeated's forward (a block broadcast): `outer` copies of [each of
 * the `size` values of x repeated `inner` times], `total` values in all. */
KERNEL expand_repeated(const real *x, real *out, long size, long inner, long total)
{
    long i = thread_index();
    if (i < total) out[i] = x[(i / inner) % size];
}

/* `times` copies of x one after another (DeviceArray's x * times, like a list's). */
KERNEL repeat_values(const real *x, long count, real *out, long total)
{
    long i = thread_index();
    if (i < total) out[i] = x[i % count];
}

/* ------------------------------------------------------------------------
 * Matrix products (gemm).
 * --------------------------------------------------------------------- */

/* (end of gemm) */

/* ------------------------------------------------------------------------
 * Convolutions, pooling, gathers and scatters.
 * --------------------------------------------------------------------- */

/* (end of conv) */

/* ------------------------------------------------------------------------
 * Rows and channels: softmax, norms, losses, GELU, the optimizer.
 * --------------------------------------------------------------------- */

/* (end of rows) */

/* ------------------------------------------------------------------------
 * Recurrent: the LSTM cell and layer, the CRF.
 * --------------------------------------------------------------------- */

/* (end of recurrent) */
