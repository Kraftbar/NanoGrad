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

/*
 * Two ideas carry this section.
 *
 * 1. A reduction over a row. A row's maximum or sum needs every value of
 *    the row, so a team of threads shares one row: each thread strides over
 *    it (thread t takes values t, t + team, t + 2 team, ..., so neighbouring
 *    threads read neighbouring values) and keeps its own partial result,
 *    in double, as kernels.c does. Then the team combines its partials.
 *    Within a warp (32 threads) __shfl_xor_sync swaps values directly
 *    between registers: after offsets 16, 8, 4, 2, 1 every thread holds the
 *    warp's total (a tree: 5 steps instead of 31 additions in a chain).
 *
 *    The team is a warp for rows of up to 1024 values: blocks of 32 x 8
 *    threads take 8 rows, threadIdx.y picks the row and threadIdx.x the
 *    thread within it, and the shuffles are the whole reduction. Wider rows
 *    get a whole block each (256 x 1 threads): each warp's total goes
 *    through shared memory, and every thread adds up the 8 warps' totals.
 *    (One big block per narrow row would leave the GPU idle: a block
 *    spends much of its time waiting, at each reduction, for its slowest
 *    warp, and an SM holds only a few big blocks to switch between.)
 *
 * 2. Sums down the columns (a LayerNorm's dweight, summed over every row)
 *    without atomics. atomicAdd would let each row add into dweight as it
 *    goes, but the order of the additions, and so the rounding, would
 *    change from run to run. Instead, two passes: thread (chunk c, column
 *    s) sums column s over one chunk of rows into a partial sum; then
 *    thread s adds column s's partials in chunk order. Neighbouring threads
 *    take neighbouring columns, so reads are coalesced, and the result is
 *    the same every run.
 *
 * The arithmetic on each value is done in the stored precision, `real`,
 * and only the sums are accumulated in double (as in kernels.c). In the
 * float32 build that matters: a gaming GPU runs double arithmetic, and
 * even float <-> double conversions, at 1/64 of its float speed, so a
 * float kernel that computed every value in double would be limited by
 * that rather than by memory.
 */

/* a + b, or the larger of the two */
__device__ __forceinline__ double rows_combine(double a, double b, bool take_max)
{
    return take_max ? fmax(a, b) : a + b;
}

/*
 * The sum (or maximum) of `value` over the threads that share a row (those
 * with this threadIdx.y), returned to all of them. They must all call it.
 * blockDim.x is 32 (a warp per row) or a multiple of 32 with blockDim.y = 1
 * (a block per row, at most 1024 threads: then every thread must call it).
 */
__device__ double rows_reduce(double value, bool take_max)
{
    for (int offset = 16; offset > 0; offset /= 2)            /* within the warp */
        value = rows_combine(value, __shfl_xor_sync(0xffffffffu, value, offset), take_max);
    if (blockDim.x == 32) return value;                       /* the warp is the row's team */

    __shared__ double warp_results[32];                       /* a block per row: combine its warps */
    int warps = blockDim.x / 32;
    if (threadIdx.x % 32 == 0) warp_results[threadIdx.x / 32] = value;
    __syncthreads();
    value = warp_results[0];
    for (int w = 1; w < warps; w++) value = rows_combine(value, warp_results[w], take_max);
    __syncthreads();                                          /* everyone has read them: reusable */
    return value;
}

__device__ __forceinline__ double rows_sum(double value) { return rows_reduce(value, false); }
__device__ __forceinline__ double rows_max(double value) { return rows_reduce(value, true); }

/* The row this thread works on (blockDim.y rows per block, see idea 1). */
__device__ __forceinline__ long row_index()
{
    return (long)blockIdx.x * blockDim.y + threadIdx.y;
}

/* The column sums of idea 2, step 1: partial[c, s] = the sum over the rows r of chunk c of
 *     a[r, s] * (b[r, s] if with_b) * (scale[r] if with_scale)
 * with one thread per (chunk c = blockIdx.y, column s). */
KERNEL rows_column_partials(const real *a, const real *b, const real *scale, long with_b, long with_scale,
                            double *partial, long rows, long width, long chunk)
{
    long s = thread_index(), c = blockIdx.y;
    if (s >= width) return;
    long r1 = (c + 1) * chunk < rows ? (c + 1) * chunk : rows;
    double total = 0.0;
    for (long r = c * chunk; r < r1; r++) {
        real term = a[r * width + s];
        if (with_b) term *= b[r * width + s];
        if (with_scale) term *= scale[r];
        total += term;
    }
    partial[c * width + s] = total;
}

/* Step 2: out[s] += the chunks' partial sums of column s, in chunk order. */
KERNEL rows_column_finish(const double *partial, real *out, long chunks, long width)
{
    long s = thread_index();
    if (s >= width) return;
    double total = 0.0;
    for (long c = 0; c < chunks; c++) total += partial[c * width + s];
    out[s] = (real)(out[s] + total);
}

/*
 * A sum over a whole array works like idea 1 over many blocks: each block
 * sums its share (a grid-stride loop: thread i takes values i, i + the
 * number of threads launched, ...) into partial[block], and then one block
 * adds up the partials (this kernel).
 */
KERNEL rows_sum_partials(const double *partial, double *total, long count)
{
    double sum = 0.0;
    for (long i = threadIdx.x; i < count; i += blockDim.x) sum += partial[i];
    sum = rows_sum(sum);
    if (threadIdx.x == 0) total[0] = sum;
}

/* GELU (exact erf form), one thread per value:  y = x Phi(x),  cdf = Phi(x),
 *     dx = dy * (Phi(x) + x phi(x)),   phi(x) = e^(-x^2 / 2) / sqrt(2 pi) */
KERNEL gelu_forward(const real *x, real *y, real *cdf, long count)
{
    long i = thread_index();
    if (i >= count) return;
    real phi = (real)0.5 * ((real)1 + erf(x[i] * (real)0.7071067811865476));
    cdf[i] = phi;
    y[i] = x[i] * phi;
}

KERNEL gelu_backward(const real *x, const real *cdf, const real *grad, real *dx, long count)
{
    long i = thread_index();
    if (i >= count) return;
    real density = exp((real)-0.5 * x[i] * x[i]) * (real)0.3989422804014327;   /* 1 / sqrt(2 pi) */
    dx[i] = grad[i] * (cdf[i] + x[i] * density);
}

/* Softmax of each row, with a team of threads per row (idea 1):
 *     y_i = e^(x_i - max) / sum_j e^(x_j - max) */
KERNEL softmax_rows(const real *x, real *y, long rows, long width)
{
    long r = row_index();
    if (r >= rows) return;                                    /* the last block's spare rows */
    const real *in = x + r * width;
    real *out = y + r * width;
    real row_max = in[0];
    for (long i = threadIdx.x; i < width; i += blockDim.x) row_max = in[i] > row_max ? in[i] : row_max;
    row_max = (real)rows_max(row_max);
    double total = 0.0;
    for (long i = threadIdx.x; i < width; i += blockDim.x) {
        real e = exp(in[i] - row_max);
        out[i] = e;
        total += e;
    }
    real inverse = (real)1 / (real)rows_sum(total);
    for (long i = threadIdx.x; i < width; i += blockDim.x) out[i] *= inverse;
}

/* dx_i = y_i * (dy_i - sum_j dy_j y_j), a team per row */
KERNEL softmax_rows_backward(const real *y, const real *grad, real *dx, long rows, long width)
{
    long r = row_index();
    if (r >= rows) return;
    long offset = r * width;
    double weighted = 0.0;
    for (long i = threadIdx.x; i < width; i += blockDim.x) weighted += grad[offset + i] * y[offset + i];
    real total = (real)rows_sum(weighted);
    for (long i = threadIdx.x; i < width; i += blockDim.x)
        dx[offset + i] = y[offset + i] * (grad[offset + i] - total);
}

/*
 * Softmax cross entropy, in place (see kernels.c): row r of logits becomes
 * factor * (softmax - one_hot(target)), and nll[r] = -log p_target. Only
 * the thread whose stride passes the target ever touches x[target], so
 * that thread alone reads the target's logit and probability, with no
 * extra waiting.
 */
KERNEL softmax_xent_rows(real *x, const long *targets, double factor, double *nll, long rows, long width)
{
    long r = row_index();
    if (r >= rows) return;
    real *row = x + r * width;
    long target = targets[r];
    real x_target = threadIdx.x == target % blockDim.x ? row[target] : (real)0;
    real row_max = row[0];
    for (long i = threadIdx.x; i < width; i += blockDim.x) row_max = row[i] > row_max ? row[i] : row_max;
    row_max = (real)rows_max(row_max);
    double total = 0.0;
    for (long i = threadIdx.x; i < width; i += blockDim.x) {
        real e = exp(row[i] - row_max);
        row[i] = e;
        total += e;
    }
    total = rows_sum(total);
    real inverse = (real)1 / (real)total, scale = (real)factor;
    for (long i = threadIdx.x; i < width; i += blockDim.x) {
        real p = row[i] * inverse;
        row[i] = p * scale;
        if (i == target) {
            nll[r] = p > 0 ? -log((double)p) : row_max + log(total) - x_target;
            row[i] = (real)((double)row[i] - factor);
        }
    }
}

/*
 * LayerNorm of each row, a team per row:
 *     x_hat = (x - mean) / sqrt(var + eps),   y = x_hat * weight + bias
 * The mean, and then the variance, are sums over the row.
 */
KERNEL layer_norm_rows(const real *x, const real *weight, const real *bias,
                       real *y, real *x_hat, real *inv_std, long rows, long width, double eps)
{
    long r = row_index();
    if (r >= rows) return;
    long offset = r * width;
    double sum = 0.0, squares = 0.0;
    for (long i = threadIdx.x; i < width; i += blockDim.x) sum += x[offset + i];
    real mean = (real)rows_sum(sum) / width;
    for (long i = threadIdx.x; i < width; i += blockDim.x) {
        real d = x[offset + i] - mean;
        squares += d * d;
    }
    real scale = (real)1 / sqrt((real)rows_sum(squares) / width + (real)eps);
    if (threadIdx.x == 0) inv_std[r] = scale;
    for (long i = threadIdx.x; i < width; i += blockDim.x) {
        real h = (x[offset + i] - mean) * scale;
        x_hat[offset + i] = h;
        y[offset + i] = h * weight[i] + bias[i];
    }
}

/* Its dx, a team per row, with g = dy * weight:
 *     dx = inv_std * (g - mean(g) - x_hat * mean(g * x_hat))
 * (dweight and dbias, sums over the rows, are column sums: idea 2). */
KERNEL layer_norm_rows_backward(const real *grad, const real *x_hat, const real *inv_std,
                                const real *weight, real *dx, long rows, long width)
{
    long r = row_index();
    if (r >= rows) return;
    long offset = r * width;
    double sum_g = 0.0, sum_g_h = 0.0;
    for (long i = threadIdx.x; i < width; i += blockDim.x) {
        real g = grad[offset + i] * weight[i];
        sum_g += g;
        sum_g_h += g * x_hat[offset + i];
    }
    real mean_g = (real)rows_sum(sum_g) / width, mean_g_h = (real)rows_sum(sum_g_h) / width;
    real scale = inv_std[r];
    for (long i = threadIdx.x; i < width; i += blockDim.x) {
        real g = grad[offset + i] * weight[i];
        dx[offset + i] = scale * (g - mean_g - x_hat[offset + i] * mean_g_h);
    }
}

/* RMSNorm of each row, a team per row:
 *     inv_rms = 1 / sqrt(mean(x^2) + eps),   y = x * inv_rms * weight */
KERNEL rms_norm_rows(const real *x, const real *weight, real *y, real *inv_rms,
                     long rows, long width, double eps)
{
    long r = row_index();
    if (r >= rows) return;
    long offset = r * width;
    double squares = 0.0;
    for (long i = threadIdx.x; i < width; i += blockDim.x) squares += x[offset + i] * x[offset + i];
    real scale = (real)1 / sqrt((real)rows_sum(squares) / width + (real)eps);
    if (threadIdx.x == 0) inv_rms[r] = scale;
    for (long i = threadIdx.x; i < width; i += blockDim.x) y[offset + i] = x[offset + i] * scale * weight[i];
}

/* Its dx, with s = inv_rms and g = dy * weight:  dx = s * (g - x * s^2 * mean(g * x))
 * (dweight, the sum over rows of dy * x * s, is a column sum: idea 2). */
KERNEL rms_norm_rows_backward(const real *grad, const real *x, const real *inv_rms,
                              const real *weight, real *dx, long rows, long width)
{
    long r = row_index();
    if (r >= rows) return;
    long offset = r * width;
    real scale = inv_rms[r];
    double sum_g_x = 0.0;
    for (long i = threadIdx.x; i < width; i += blockDim.x) sum_g_x += grad[offset + i] * weight[i] * x[offset + i];
    real mean_g_x = (real)rows_sum(sum_g_x) / width;
    for (long i = threadIdx.x; i < width; i += blockDim.x)
        dx[offset + i] = scale * (grad[offset + i] * weight[i] - x[offset + i] * scale * scale * mean_g_x);
}

/*
 * Batch norm in training mode, for (batch, channels, plane) data: one block
 * per channel, reducing over the channel's batch * plane values (value j
 * of channel c is pixel j % plane of image j / plane). The formulas are
 * LayerNorm's, per channel instead of per row.
 */
__device__ __forceinline__ long channel_offset(long j, long c, long channels, long plane)
{
    return ((j / plane) * channels + c) * plane + j % plane;
}

KERNEL batch_norm_channels(const real *x, const real *weight, const real *bias,
                           real *y, real *x_hat, real *inv_std, real *mean, real *variance,
                           long batch, long channels, long plane, double eps)
{
    long c = blockIdx.x, n = batch * plane;
    double sum = 0.0, squares = 0.0;
    for (long j = threadIdx.x; j < n; j += blockDim.x) sum += x[channel_offset(j, c, channels, plane)];
    real m = (real)(rows_sum(sum) / n);
    for (long j = threadIdx.x; j < n; j += blockDim.x) {
        real d = x[channel_offset(j, c, channels, plane)] - m;
        squares += d * d;
    }
    double v = rows_sum(squares) / n;
    real scale = (real)(1.0 / sqrt(v + eps)), w = weight[c], b = bias[c];
    if (threadIdx.x == 0) {
        mean[c] = m;
        variance[c] = (real)v;
        inv_std[c] = scale;
    }
    for (long j = threadIdx.x; j < n; j += blockDim.x) {
        long k = channel_offset(j, c, channels, plane);
        real h = (x[k] - m) * scale;
        x_hat[k] = h;
        y[k] = h * w + b;
    }
}

/* dweight_c = sum dy * x_hat,  dbias_c = sum dy  (block sums: a block is a channel),
 * dx = inv_std_c * (dy * weight_c - mean(g) - x_hat * mean(g * x_hat)) */
KERNEL batch_norm_channels_backward(const real *grad, const real *x_hat, const real *inv_std,
                                    const real *weight, real *dx, real *dweight, real *dbias,
                                    long batch, long channels, long plane)
{
    long c = blockIdx.x, n = batch * plane;
    double sum_dy = 0.0, sum_dy_h = 0.0;
    for (long j = threadIdx.x; j < n; j += blockDim.x) {
        long k = channel_offset(j, c, channels, plane);
        sum_dy += grad[k];
        sum_dy_h += grad[k] * x_hat[k];
    }
    sum_dy = rows_sum(sum_dy);
    sum_dy_h = rows_sum(sum_dy_h);
    if (threadIdx.x == 0) {
        dweight[c] = (real)sum_dy_h;
        dbias[c] = (real)sum_dy;
    }
    real w = weight[c], scale = inv_std[c];
    real mean_g = (real)(w * sum_dy / n), mean_g_h = (real)(w * sum_dy_h / n);
    for (long j = threadIdx.x; j < n; j += blockDim.x) {
        long k = channel_offset(j, c, channels, plane);
        dx[k] = scale * (grad[k] * w - mean_g - x_hat[k] * mean_g_h);
    }
}

/* The sum of x[i]^2, as per-block partial sums (finished by rows_sum_partials). */
KERNEL sum_of_squares(const real *x, double *partial, long count)
{
    double sum = 0.0;
    for (long i = thread_index(); i < count; i += (long)gridDim.x * blockDim.x) sum += (double)x[i] * x[i];
    sum = rows_sum(sum);
    if (threadIdx.x == 0) partial[blockIdx.x] = sum;
}

/* Binary cross entropy on logits, the sum of max(x, 0) - x t + log(1 + e^-|x|),
 * as per-block partial sums like sum_of_squares. */
KERNEL bce_with_logits_sum(const real *x, const real *t, double *partial, long count)
{
    double sum = 0.0;
    for (long i = thread_index(); i < count; i += (long)gridDim.x * blockDim.x) {
        real value = x[i];
        sum += (value > 0 ? value : (real)0) - value * t[i] + log1p(exp(-fabs(value)));
    }
    sum = rows_sum(sum);
    if (threadIdx.x == 0) partial[blockIdx.x] = sum;
}

/* Its gradient, scaled by g:  dx = g * (sigmoid(x) - t), with sigmoid in the form that cannot overflow */
KERNEL bce_with_logits_backward(const real *x, const real *t, double g, real *dx, long count)
{
    long i = thread_index();
    if (i >= count) return;
    real e = exp(-fabs(x[i])), s = (real)1 / ((real)1 + e);
    real sigmoid = x[i] >= 0 ? s : e * s;
    dx[i] = (real)(g * (sigmoid - t[i]));
}

/*
 * Dropout mask: entry i is 0 with probability p, else 1 / (1 - p), from
 * the i-th splitmix64 output after `seed`. splitmix64 is counter based:
 * output i is a function of seed + i alone, so each thread computes its
 * own entry independently, and the mask is the same bit for bit as
 * kernels.c's. (`seed` holds the 64 bits of the C version's unsigned seed.)
 */
KERNEL dropout_mask(real *mask, long count, double p, long seed)
{
    long i = thread_index();
    if (i >= count) return;
    unsigned long long z = (unsigned long long)seed + (unsigned long long)(i + 1) * 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    double u = (double)(z >> 11) * 0x1.0p-53;
    mask[i] = u < p ? (real)0 : (real)(1.0 / (1.0 - p));
}

/* One step of a sign-gradient attack, one thread per value:
 *     d = clip(delta + alpha * sign(grad), -eps, eps),  then clip(d, -x, 1 - x) */
KERNEL projected_sign_step(const real *delta, const real *grad, const real *x, double alpha, double eps,
                           real *out, long count)
{
    long i = thread_index();
    if (i >= count) return;
    real a = (real)alpha, e = (real)eps, low = -x[i], high = (real)1 - x[i];
    real d = delta[i] + (grad[i] > 0 ? a : (grad[i] < 0 ? -a : (real)0));
    d = d < -e ? -e : (d > e ? e : d);
    d = d < low ? low : d;
    out[i] = d > high ? high : d;
}

/* One Adam step, one thread per weight, m and v updated in place (see kernels.c):
 *     g = grad * grad_scale,  m = beta1 m + (1 - beta1) g,  v = beta2 v + (1 - beta2) g^2
 *     new_w = w * decay - step_size * m / (sqrt(v) / sqrt_correction + eps) */
KERNEL adam_update(const real *w, const real *grad, real *m, real *v, real *new_w,
                   long count, double grad_scale, double beta1, double beta2, double decay,
                   double step_size, double sqrt_correction, double eps)
{
    long i = thread_index();
    if (i >= count) return;
    real g = grad[i] * (real)grad_scale;
    real m_i = (real)beta1 * m[i] + (real)(1.0 - beta1) * g;
    real v_i = (real)beta2 * v[i] + (real)(1.0 - beta2) * g * g;
    m[i] = m_i;
    v[i] = v_i;
    new_w[i] = w[i] * (real)decay - (real)step_size * m_i / (sqrt(v_i) / (real)sqrt_correction + (real)eps);
}

/* (end of rows) */

/* ------------------------------------------------------------------------
 * Recurrent: the LSTM cell and layer, the CRF.
 * --------------------------------------------------------------------- */

/* (end of recurrent) */
