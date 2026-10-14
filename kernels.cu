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
 *
 * Scatter-adds (several values landing on the same pixel or row) can't
 * simply give each value a thread that adds it on: two threads adding to
 * the same number at once would lose one of the adds. There are two ways
 * out, both used here:
 *
 * - atomicAdd: the hardware adds each value in turn, in whatever order
 *   the threads arrive. Right, but the order changes from run to run, and
 *   float addition is not associative, so the last bits do too.
 * - Turn the scatter into a gather: one thread per *target*, adding up
 *   the values that land on it, in a fixed order. That needs the index
 *   table inverted (for each target, which entries point at it: `starts`
 *   and `sources`, built once per table by kernels_cuda.py), and gives
 *   the same bits every run (the same as the C loop, in fact).
 *
 * The scatters through fixed index tables (sums over an axis, pooling and
 * depthwise gradients) are gathers; those through indices that change
 * every call (max-pool winners, embedding ids, the strided convolution's
 * col2im) use atomics. The kernel gradients, which sum over the whole
 * batch, add in a fixed order too.
 * --------------------------------------------------------------------- */

/*
 * The three convolution products ("implicit GEMM"). Each is a matrix
 * product C = A @ B in which one operand is the patch matrix
 *
 *     patches_b[j][q] = image_b[index[j * K + q]]        (K = patch_size)
 *
 * which is never built: the tile loads below read it straight from the
 * image through the index table. As in kernels.c, every entry of the
 * table is a position's base plus a patch offset,
 *
 *     index[j * K + q] = index[j * K] + index[q],
 *
 * so a load needs the position's base (index[j * K]) and the offset of
 * patch value q (index[q], the table's first row).
 *
 * Tiling. A block of 256 threads computes one 64 x 64 tile of C. It walks
 * the inner dimension 16 at a time: the threads together copy a 64 x 16
 * slice of A and a 16 x 64 slice of B into shared memory, wait, and then
 * each thread multiplies out its own 4 x 4 piece of the tile from shared
 * memory. Every number loaded from GPU memory is then used 64 times (by
 * the 64 rows or columns of the tile) instead of once, which is what turns
 * a memory-bound triple loop into an arithmetic-bound one.
 *
 * Thread (ty, tx) of the 16 x 16 threads owns rows ty, ty + 16, ty + 32,
 * ty + 48 and the same columns from tx: neighbouring threads then read
 * neighbouring shared-memory words (no bank conflicts) and write
 * neighbouring numbers of C (coalesced).
 */
#define CONV_TILE 64                 /* rows and columns of C per block */
#define CONV_STEP 16                 /* inner-dimension slice per step */
#define CONV_THREADS 256

/*
 * acc (this thread's 4 x 4 piece) += A[i0.., k_lo..k_hi] @ B[k_lo..k_hi, j0..]
 * for an m x n product. load_a(i, p) and load_b(p, j) read one entry;
 * a_along_k / b_along_k say which index neighbouring threads should take
 * while loading, so that they read neighbouring addresses: the inner one
 * (p) if the operand is contiguous along the inner dimension, else the
 * row (i) or column (j).
 */
template <bool a_along_k, bool b_along_k, class LoadA, class LoadB>
__device__ __forceinline__ void conv_tile_product(LoadA load_a, LoadB load_b, long m, long n,
                                                  long k_lo, long k_hi, long i0, long j0, real acc[4][4])
{
    /* One spare column keeps threads that store down a column on different banks. */
    __shared__ real a_tile[CONV_STEP][CONV_TILE + 1];   /* a_tile[p][i] = A(i0 + i, p0 + p) */
    __shared__ real b_tile[CONV_STEP][CONV_TILE + 1];   /* b_tile[p][j] = B(p0 + p, j0 + j) */
    const int t = threadIdx.x, tx = t % 16, ty = t / 16;
    for (long p0 = k_lo; p0 < k_hi; p0 += CONV_STEP) {
        for (int e = t; e < CONV_STEP * CONV_TILE; e += CONV_THREADS) {
            int p = a_along_k ? e % CONV_STEP : e / CONV_TILE;
            int i = a_along_k ? e / CONV_STEP : e % CONV_TILE;
            a_tile[p][i] = i0 + i < m && p0 + p < k_hi ? load_a(i0 + i, p0 + p) : (real)0;
            p = b_along_k ? e % CONV_STEP : e / CONV_TILE;
            int j = b_along_k ? e / CONV_STEP : e % CONV_TILE;
            b_tile[p][j] = j0 + j < n && p0 + p < k_hi ? load_b(p0 + p, j0 + j) : (real)0;
        }
        __syncthreads();                             /* the tiles are complete */
        for (int p = 0; p < CONV_STEP; p++) {
            real a[4], b[4];
            for (int r = 0; r < 4; r++) a[r] = a_tile[p][ty + 16 * r];
            for (int c = 0; c < 4; c++) b[c] = b_tile[p][tx + 16 * c];
            for (int r = 0; r < 4; r++)
                for (int c = 0; c < 4; c++) acc[r][c] += a[r] * b[c];
        }
        __syncthreads();                             /* everyone is done before they are overwritten */
    }
}

/*
 *     out_b (filters x positions) = kernel (filters x K) @ patches_b^T
 * Block (x, y, z) computes the tile at positions 64x.., filters 64y.. of image z.
 */
KERNEL conv_forward(const real *image, const long *index, const real *kernel, real *out,
                    long batches, long image_size, long filters, long patch_size, long positions)
{
    const long K = patch_size, P = positions;
    const real *img = image + blockIdx.z * image_size;
    long i0 = (long)blockIdx.y * CONV_TILE, j0 = (long)blockIdx.x * CONV_TILE;
    real acc[4][4] = {};
    conv_tile_product<true, false>(
        [&](long i, long p) { return kernel[i * K + p]; },               /* contiguous along p */
        [&](long p, long j) { return img[index[j * K] + index[p]]; },    /* along j: neighbouring pixels */
        filters, P, 0, K, i0, j0, acc);
    real *o = out + blockIdx.z * filters * P;
    int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) {
            long i = i0 + ty + 16 * r, j = j0 + tx + 16 * c;
            if (i < filters && j < P) o[i * P + j] = acc[r][c];
        }
}

/*
 *     dkernel (filters x K) = sum over images of grad_b (filters x positions) @ patches_b
 *
 * dkernel is small and the inner dimension (every position of every image)
 * is long, so there are too few tiles to keep the GPU busy. The inner
 * dimension is therefore split into slices (block z takes slice z: a run
 * of whole images, or part of one image's positions), each block writes
 * its slice's sum to partial[z], and conv_sum_slices adds the slices up.
 * Every number is summed by one thread in a fixed order, so unlike with
 * atomics the result is the same every run.
 */
KERNEL conv_kernel_grad(const real *image, const long *index, const real *grad, real *partial,
                        long batches, long image_size, long filters, long patch_size, long positions,
                        long images_per_slice, long slices_per_image, long chunk)
{
    const long K = patch_size, P = positions, F = filters;
    long i0 = (long)blockIdx.y * CONV_TILE, j0 = (long)blockIdx.x * CONV_TILE, slice = blockIdx.z;
    long b_lo = slice / slices_per_image * images_per_slice;
    long b_hi = b_lo + images_per_slice < batches ? b_lo + images_per_slice : batches;
    long p_lo = slice % slices_per_image * chunk, p_hi = p_lo + chunk < P ? p_lo + chunk : P;
    real acc[4][4] = {};
    for (long b = b_lo; b < b_hi; b++) {
        const real *img = image + b * image_size, *g = grad + b * F * P;
        conv_tile_product<true, true>(
            [&](long i, long p) { return g[i * P + p]; },                   /* grad_b, contiguous along p */
            [&](long p, long q) { return img[index[p * K] + index[q]]; },   /* patch p, value q */
            F, K, p_lo, p_hi, i0, j0, acc);
    }
    real *out = partial + slice * F * K;
    int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) {
            long i = i0 + ty + 16 * r, q = j0 + tx + 16 * c;
            if (i < F && q < K) out[i * K + q] = acc[r][c];
        }
}

/* out[i] = sum over s of partial[s * count + i], s in order (one thread per i). */
KERNEL conv_sum_slices(const real *partial, real *out, long slices, long count)
{
    long i = thread_index();
    if (i >= count) return;
    real total = 0;
    for (long s = 0; s < slices; s++) total += partial[s * count + i];
    out[i] = total;
}

/*
 * The image gradient ("col2im"): for each image,
 *     dpatches (positions x K) = grad_b^T (positions x filters) @ kernel (filters x K)
 *     image_grad_b[index[j * K + q]] += dpatches[j][q]
 * Each block computes a tile of dpatches and adds it straight onto the
 * pixels. Neighbouring patches overlap, and their tiles may be in
 * different blocks, so the adds are atomic. image_grad starts as zeros.
 */
KERNEL conv_input_grad(const real *grad, const real *kernel, const long *index, real *image_grad,
                       long batches, long image_size, long filters, long patch_size, long positions)
{
    const long K = patch_size, P = positions, F = filters;
    const real *g = grad + blockIdx.z * F * P;
    long i0 = (long)blockIdx.y * CONV_TILE, j0 = (long)blockIdx.x * CONV_TILE;
    real acc[4][4] = {};
    conv_tile_product<false, false>(
        [&](long j, long f) { return g[f * P + j]; },                    /* grad_b^T: contiguous along j */
        [&](long f, long q) { return kernel[f * K + q]; },               /* contiguous along q */
        P, K, 0, F, i0, j0, acc);
    real *dst = image_grad + blockIdx.z * image_size;
    int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) {
            long j = i0 + ty + 16 * r, q = j0 + tx + 16 * c;
            if (j < P && q < K) atomicAdd(dst + index[j * K] + index[q], acc[r][c]);
        }
}

/*
 * Direct convolution, stride 1 (the few-filter path): one thread per
 * output pixel, x fastest, so a warp reads a run of neighbouring pixels
 * and all its threads read the same weight (a broadcast).
 * image (B, C, H, W), kernel (F, C, kh, kw), out (B, F, H-kh+1, W-kw+1).
 */
KERNEL conv_direct(const real *image, const real *kernel, real *out, long batches,
                   long channels, long height, long width, long filters, long kh, long kw)
{
    long oh = height - kh + 1, ow = width - kw + 1;
    long o = thread_index();
    if (o >= batches * filters * oh * ow) return;
    long x = o % ow, y = o / ow % oh, f = o / (ow * oh) % filters, b = o / (ow * oh * filters);
    const real *img = image + b * channels * height * width + y * width + x;
    const real *w = kernel + f * channels * kh * kw;
    real total = 0;
    for (long c = 0; c < channels; c++)
        for (long i = 0; i < kh; i++) {
            const real *row = img + (c * height + i) * width;
            for (long j = 0; j < kw; j++) total += *w++ * row[j];
        }
    out[o] = total;
}

/*
 * Zero padding of 2D planes: one thread per padded value, which is either
 * copied from the plane or zero (so out needs no clearing first).
 * crop_planes, its backward pass, reads the middle back out.
 */
KERNEL pad_planes(const real *x, real *out, long planes, long rows, long cols, long pad_rows, long pad_cols)
{
    long padded_rows = rows + 2 * pad_rows, padded_cols = cols + 2 * pad_cols;
    long o = thread_index();
    if (o >= planes * padded_rows * padded_cols) return;
    long c = o % padded_cols - pad_cols, r = o / padded_cols % padded_rows - pad_rows;
    long p = o / (padded_cols * padded_rows);
    out[o] = r >= 0 && r < rows && c >= 0 && c < cols ? x[(p * rows + r) * cols + c] : (real)0;
}

KERNEL crop_planes(const real *padded, real *out, long planes, long rows, long cols, long pad_rows, long pad_cols)
{
    long padded_rows = rows + 2 * pad_rows, padded_cols = cols + 2 * pad_cols;
    long o = thread_index();
    if (o >= planes * rows * cols) return;
    long c = o % cols, r = o / cols % rows, p = o / (cols * rows);
    out[o] = padded[(p * padded_rows + r + pad_rows) * padded_cols + c + pad_cols];
}

/*
 * im2col / col2im with an index table:
 *     gather:  out[b, i] = image[b, index[i]]
 *     scatter: image_grad[b, index[i]] += grad[b, i]   (overlapping patches add up)
 * The gather has one thread per value read.
 */
KERNEL gather_indexed(const real *image, const long *index, real *out,
                      long batches, long image_size, long count)
{
    long o = thread_index();
    if (o < batches * count) out[o] = image[o / count * image_size + index[o % count]];
}

/* The scatter as a gather, one thread per pixel: the entries pointing at
 * pixel t are sources[starts[t] .. starts[t + 1] - 1], ascending. */
KERNEL scatter_add_indexed(const real *grad, const long *starts, const long *sources, real *image_grad,
                           long batches, long image_size, long count)
{
    long o = thread_index();
    if (o >= batches * image_size) return;
    const real *g = grad + o / image_size * count;
    long t = o % image_size;
    if (starts[t] == starts[t + 1]) return;          /* nothing lands here */
    real total = image_grad[o];
    for (long e = starts[t]; e < starts[t + 1]; e++) total += g[sources[e]];
    image_grad[o] = total;
}

/* The scatter with atomics, one thread per value, for indices without an inverse. */
KERNEL scatter_add_indexed_atomic(const real *grad, const long *index, real *image_grad,
                                  long batches, long image_size, long count)
{
    long o = thread_index();
    if (o < batches * count) atomicAdd(image_grad + o / count * image_size + index[o % count], grad[o]);
}

/*
 * Depthwise convolution: every channel filtered by its own kernel; index
 * holds the Q = kh * kw offsets of each output position's patch in one
 * channel image (see kernels.c).
 *     out[b, c, p]       = sum_q image[b, c, index[p, q]] * kernel[c, q]
 *     dkernel[c, q]     += sum_b,p grad[b, c, p] * image[b, c, index[p, q]]
 *     dimage[b, c, index[p, q]] += grad[b, c, p] * kernel[c, q]
 * Forward: one thread per output value.
 */
KERNEL depthwise_forward(const real *image, const long *index, const real *kernel, real *out,
                         long batches, long channels, long channel_size, long positions, long q_size)
{
    long o = thread_index();
    if (o >= batches * channels * positions) return;
    long p = o % positions, bc = o / positions;
    const real *img = image + bc * channel_size, *k = kernel + bc % channels * q_size;
    const long *cells = index + p * q_size;
    double total = 0.0;
    for (long q = 0; q < q_size; q++) total += img[cells[q]] * k[q];
    out[o] = total;
}

/*
 * The sum of one value per thread over a block of CONV_THREADS threads,
 * in shared memory: halve the number of active threads each step, each
 * adding its partner's value (a tree, always in the same order). Every
 * thread must call it; thread 0 gets the total.
 */
__device__ double conv_block_sum(double value)
{
    __shared__ double sums[CONV_THREADS];
    sums[threadIdx.x] = value;
    __syncthreads();
    for (int half = CONV_THREADS / 2; half > 0; half /= 2) {
        if (threadIdx.x < half) sums[threadIdx.x] += sums[threadIdx.x + half];
        __syncthreads();
    }
    return sums[0];
}

/*
 * depthwise_backward's kernel gradient: one block per (c, q), its threads
 * striding over every (image, position) and then adding up their sums (no
 * atomics, so reproducible). Launch with CONV_THREADS threads.
 */
KERNEL depthwise_kernel_grad(const real *image, const long *index, const real *grad, real *dkernel,
                             long batches, long channels, long channel_size, long positions, long q_size)
{
    long c = blockIdx.x / q_size, q = blockIdx.x % q_size;
    double total = 0.0;
    for (long e = threadIdx.x; e < batches * positions; e += CONV_THREADS) {
        long bc = e / positions * channels + c, p = e % positions;
        total += grad[bc * positions + p] * image[bc * channel_size + index[p * q_size + q]];
    }
    total = conv_block_sum(total);
    if (threadIdx.x == 0) dkernel[blockIdx.x] = total;
}

/* Its image gradient, as a gather: one thread per pixel, adding the
 * (position p, patch value q) pairs that read it, from the inverted index
 * (entry e = p * Q + q). */
KERNEL depthwise_image_grad(const long *starts, const long *sources, const real *kernel, const real *grad,
                            real *dimage, long batches, long channels, long channel_size, long positions,
                            long q_size)
{
    long o = thread_index();
    if (o >= batches * channels * channel_size) return;
    long bc = o / channel_size, t = o % channel_size;
    const real *k = kernel + bc % channels * q_size, *g = grad + bc * positions;
    real total = 0;
    for (long e = starts[t]; e < starts[t + 1]; e++) total += g[sources[e] / q_size] * k[sources[e] % q_size];
    dimage[o] = total;
}

/*
 * Sparse (CSR) times dense: one thread per output value out[r, j], adding
 * the nonzeros of row r in order (neighbouring threads take neighbouring j,
 * so they read neighbouring numbers of each dense row).
 */
KERNEL csr_matmul(const long *row_start, const long *cols, const real *values,
                  const real *dense, real *out, long rows, long n)
{
    long o = thread_index();
    if (o >= rows * n) return;
    long r = o / n, j = o % n;
    real total = 0;
    for (long e = row_start[r]; e < row_start[r + 1]; e++) total += values[e] * dense[cols[e] * n + j];
    out[o] = total;
}

/*
 * Embedding lookup, one thread per number:
 *     gather:  out[i, :] = table[ids[i], :]
 *     scatter: table_grad[ids[i], :] += grad[i, :]   (atomic: ids repeat)
 */
KERNEL gather_rows(const real *table, const long *ids, real *out, long count, long width)
{
    long o = thread_index();
    if (o < count * width) out[o] = table[ids[o / width] * width + o % width];
}

KERNEL scatter_add_rows(const real *grad, const long *ids, real *table_grad, long count, long width)
{
    long o = thread_index();
    if (o < count * width) atomicAdd(table_grad + ids[o / width] * width + o % width, grad[o]);
}

/*
 * Pooling over windows given by an index table, one thread per window:
 * window w of a plane reads the plane offsets index[w * window_size ..].
 *     mean:  out[p, w] = (1 / window_size) * sum_j plane_p[index[w, j]]
 *     max:   out[p, w] = max_j plane_p[index[w, j]], and argmax[p, w] is the
 *            winning pixel's offset in the whole input (first one on ties)
 * The mean's backward pass shares each window's gradient among its pixels;
 * windows can overlap, so it is a gather over the inverted index, one
 * thread per pixel (entry e belongs to window e / window_size).
 */
KERNEL window_mean(const real *image, const long *index, real *out,
                   long planes, long plane_size, long windows, long window_size)
{
    long o = thread_index();
    if (o >= planes * windows) return;
    const real *in = image + o / windows * plane_size;
    const long *cells = index + o % windows * window_size;
    double total = 0.0;
    for (long j = 0; j < window_size; j++) total += in[cells[j]];
    out[o] = total / window_size;
}

KERNEL window_mean_backward(const real *grad, const long *starts, const long *sources, real *image_grad,
                            long planes, long plane_size, long windows, long window_size)
{
    long o = thread_index();
    if (o >= planes * plane_size) return;
    const real *g = grad + o / plane_size * windows;
    long t = o % plane_size;
    real total = 0;
    for (long e = starts[t]; e < starts[t + 1]; e++) total += (real)((double)g[sources[e] / window_size] / window_size);
    image_grad[o] = total;
}

KERNEL window_max(const real *image, const long *index, real *out, long *argmax,
                  long planes, long plane_size, long windows, long window_size)
{
    long o = thread_index();
    if (o >= planes * windows) return;
    long p = o / windows;
    const real *in = image + p * plane_size;
    const long *cells = index + o % windows * window_size;
    long best = cells[0];
    for (long j = 1; j < window_size; j++) if (in[cells[j]] > in[best]) best = cells[j];
    out[o] = in[best];
    argmax[o] = p * plane_size + best;
}

/*
 * The backward pass of broadcasting a block: x holds `outer` copies of
 * [each of `size` source values repeated `inner` times]; out[s] is the sum
 * of every copy of value s. One block per s, its threads striding over the
 * copies (neighbouring threads read neighbouring numbers when inner > 1),
 * then a block sum. Launch with CONV_THREADS threads.
 */
KERNEL sum_repeated(const real *x, real *out, long outer, long size, long inner)
{
    long s = blockIdx.x;
    double total = 0.0;
    for (long e = threadIdx.x; e < outer * inner; e += CONV_THREADS)
        total += x[(e / inner * size + s) * inner + e % inner];
    total = conv_block_sum(total);
    if (threadIdx.x == 0) out[s] = total;
}

/* sum_repeated with inner == 1 and many values (a bias over rows): one
 * thread per s, adding the rows in order, so neighbouring threads read
 * neighbouring numbers of each row. */
KERNEL sum_repeated_rows(const real *x, real *out, long outer, long size)
{
    long s = thread_index();
    if (s >= size) return;
    double total = 0.0;
    for (long o = 0; o < outer; o++) total += x[o * size + s];
    out[s] = total;
}

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

/*
 * The LSTM cell (rnn.py), fused into two steps as in kernels.c. gates is
 * (batch x 4H) in PyTorch's order i, f, g, o:
 *     memory:  c' = sigmoid(f) * c + sigmoid(i) * tanh(g)
 *     output:  h  = sigmoid(o) * tanh(c')
 * One thread per state value (b, j). Neighbouring threads take neighbouring
 * j, so each of the four gate columns is read as one contiguous run. The
 * backward passes recompute the activations from the gates, and write the
 * gate columns they do not use as zero, like their C twins.
 *
 * The cell's arithmetic is written once, as __device__ functions on one
 * sentence's gate row (4H values) and its gradient row, and used both by
 * the single-step kernels and by the whole-layer ones further down.
 */
__device__ __forceinline__ real lstm_sigmoid(real x)
{
    return (real)1 / ((real)1 + exp(-x));
}

__device__ __forceinline__ real lstm_memory_value(const real *row, real c, long j, long hidden)
{
    real i = lstm_sigmoid(row[j]), f = lstm_sigmoid(row[hidden + j]), g = tanh(row[2 * hidden + j]);
    return f * c + i * g;
}

/* d is c''s gradient. Writes the i, f, g columns of drow (o is lstm_output's)
 * and returns c's gradient. */
__device__ __forceinline__ real lstm_memory_grad(const real *row, real c, real d, real *drow, long j, long hidden)
{
    real i = lstm_sigmoid(row[j]), f = lstm_sigmoid(row[hidden + j]), g = tanh(row[2 * hidden + j]);
    drow[j] = d * g * i * ((real)1 - i);
    drow[hidden + j] = d * c * f * ((real)1 - f);
    drow[2 * hidden + j] = d * i * ((real)1 - g * g);
    return d * f;
}

__device__ __forceinline__ real lstm_output_value(const real *row, real c_new, long j, long hidden)
{
    return lstm_sigmoid(row[3 * hidden + j]) * tanh(c_new);
}

/* d is h's gradient. Writes the o column of drow and returns c''s gradient. */
__device__ __forceinline__ real lstm_output_grad(const real *row, real c_new, real d, real *drow, long j, long hidden)
{
    real o = lstm_sigmoid(row[3 * hidden + j]), t = tanh(c_new);
    drow[3 * hidden + j] = d * t * o * ((real)1 - o);
    return d * o * ((real)1 - t * t);
}

KERNEL lstm_memory(const real *gates, const real *c, real *c_new, long batch, long hidden)
{
    long index = thread_index(), b = index / hidden, j = index % hidden;
    if (index < batch * hidden) c_new[index] = lstm_memory_value(gates + b * 4 * hidden, c[index], j, hidden);
}

KERNEL lstm_memory_backward(const real *gates, const real *c, const real *grad,
                            real *dgates, real *dc, long batch, long hidden)
{
    long index = thread_index(), b = index / hidden, j = index % hidden;
    if (index >= batch * hidden) return;
    real *drow = dgates + b * 4 * hidden;
    dc[index] = lstm_memory_grad(gates + b * 4 * hidden, c[index], grad[index], drow, j, hidden);
    drow[3 * hidden + j] = 0;
}

KERNEL lstm_output(const real *gates, const real *c_new, real *h, long batch, long hidden)
{
    long index = thread_index(), b = index / hidden, j = index % hidden;
    if (index < batch * hidden) h[index] = lstm_output_value(gates + b * 4 * hidden, c_new[index], j, hidden);
}

KERNEL lstm_output_backward(const real *gates, const real *c_new, const real *grad,
                            real *dgates, real *dc_new, long batch, long hidden)
{
    long index = thread_index(), b = index / hidden, j = index % hidden;
    if (index >= batch * hidden) return;
    real *drow = dgates + b * 4 * hidden;
    drow[j] = drow[hidden + j] = drow[2 * hidden + j] = 0;
    dc_new[index] = lstm_output_grad(gates + b * 4 * hidden, c_new[index], grad[index], drow, j, hidden);
}

/*
 * A whole LSTM layer (rnn.py's lstm_sequence), with kernels.c's layouts:
 * input_gates (batch, steps, 4H), weight (4H, H); gates (steps, batch, 4H)
 * and cs, hs (steps + 1, batch, H; entry 0 the starting state) are kept for
 * the backward pass; packed (batch, steps + 1, H) is h_1 .. h_T, then the
 * final c.
 *
 * Each step needs the previous step's h, so the time loop cannot run in
 * parallel: kernels_cuda.py launches two kernels per step. Kernels run in
 * launch order, so each one sees everything the one before it wrote:
 *
 *     lstm_step_gates:  gates_t = input_gates_t + h_t-1 W^T   (a tiled product)
 *     lstm_step_cell:   c_t and h_t from gates_t and c_t-1    (lstm_memory, lstm_output)
 *
 * The parallelism is within a step: batch x 4H gate values, batch x H states.
 *
 * Shared-memory tiling. Each gate value is a dot product of length H; every
 * h row is used by all 4H gates of its sentence, and every W row by all
 * sentences. Rather than have each thread read its whole h row and W row
 * from global memory, a block of LSTM_TILE x LSTM_TILE threads computes a
 * tile of the output together: it walks along k in steps of LSTM_TILE,
 * each thread loading one value of the h tile and one of the W tile into
 * shared memory, and after a __syncthreads every thread uses the whole
 * stretch of both. So each value comes from global memory once per tile
 * instead of once per output. Threads with consecutive threadIdx.x load
 * consecutive k, so the loads coalesce. (The extra column pads each row so
 * that threads reading down a column of a tile hit different memory banks
 * instead of queueing on the same one.)
 */
#define LSTM_TILE 16

KERNEL lstm_step_gates(const real *input_gates, const real *weight, const real *hs, real *gates,
                       long t, long batch, long steps, long hidden)
{
    __shared__ real h_tile[LSTM_TILE][LSTM_TILE + 1];      /* h_t-1[b][k] */
    __shared__ real w_tile[LSTM_TILE][LSTM_TILE + 1];      /* W[n][k] */
    long width = 4 * hidden;
    long tile_b = (long)blockIdx.y * LSTM_TILE, tile_n = (long)blockIdx.x * LSTM_TILE;
    long b = tile_b + threadIdx.y, n = tile_n + threadIdx.x;   /* this thread's gate value */
    const real *h = hs + t * batch * hidden;
    real sum = 0;
    for (long k0 = 0; k0 < hidden; k0 += LSTM_TILE) {
        long k = k0 + threadIdx.x, w_row = tile_n + threadIdx.y;
        h_tile[threadIdx.y][threadIdx.x] = b < batch && k < hidden ? h[b * hidden + k] : (real)0;
        w_tile[threadIdx.y][threadIdx.x] = w_row < width && k < hidden ? weight[w_row * hidden + k] : (real)0;
        __syncthreads();
        for (int kk = 0; kk < LSTM_TILE; kk++) sum += h_tile[threadIdx.y][kk] * w_tile[threadIdx.x][kk];
        __syncthreads();                                   /* before the next tile overwrites them */
    }
    if (b < batch && n < width)
        gates[(t * batch + b) * width + n] = input_gates[(b * steps + t) * width + n] + sum;
}

KERNEL lstm_step_cell(const real *gates, real *cs, real *hs, long t, long batch, long hidden)
{
    long index = thread_index(), b = index / hidden, j = index % hidden, state = batch * hidden;
    if (index >= state) return;
    const real *row = gates + (t * batch + b) * 4 * hidden;
    real c_new = lstm_memory_value(row, cs[t * state + index], j, hidden);
    cs[(t + 1) * state + index] = c_new;
    hs[(t + 1) * state + index] = lstm_output_value(row, c_new, j, hidden);
}

/* packed[b][t] = h_t+1[b] for t < steps, then the final c: one thread per value. */
KERNEL lstm_pack(const real *cs, const real *hs, real *packed, long batch, long steps, long hidden)
{
    long index = thread_index();
    if (index >= batch * (steps + 1) * hidden) return;
    long j = index % hidden, t = index / hidden % (steps + 1), b = index / hidden / (steps + 1);
    long state = batch * hidden;
    packed[index] = t < steps ? hs[(t + 1) * state + b * hidden + j] : cs[steps * state + b * hidden + j];
}

/*
 * Its backward pass, through time from the last step. grad is (batch,
 * steps + 1, H) like packed: slot t < steps is h_t+1's output gradient,
 * slot steps the final c's. Per step, again two kernels in order:
 *
 *     lstm_step_cell_backward:  dgates_t from dh_t and the carried c gradient
 *                               (lstm_output's backward, then lstm_memory's)
 *     lstm_step_recurrent:      dh_t-1 = grad slot t-1 + dgates_t W
 *
 * dgates_t goes straight into dinput_gates (it is the input gates'
 * gradient). The weight's gradient, the sum over steps of dgates_t^T h_t-1,
 * is one product after the loop (lstm_weight_grad), whose inner dimension
 * runs over every (b, t) pair at once. One thread per weight sums its own
 * products, with no atomics, so the result is the same on every run.
 */

/* dh = grad slot steps - 1, carry (c's gradient) = grad slot steps. */
KERNEL lstm_grad_start(const real *grad, real *dh, real *carry, long batch, long steps, long hidden)
{
    long index = thread_index(), b = index / hidden, j = index % hidden;
    if (index >= batch * hidden) return;
    dh[index] = grad[(b * (steps + 1) + steps - 1) * hidden + j];
    carry[index] = grad[(b * (steps + 1) + steps) * hidden + j];
}

/* One thread per state value, kernels.c's order: lstm_output_backward,
 * dc = dc_new + carry, lstm_memory_backward. Each thread reads and then
 * overwrites its own carry value, so no other thread is in the way. */
KERNEL lstm_step_cell_backward(const real *gates, const real *cs, const real *dh, real *carry,
                               real *dinput_gates, long t, long batch, long steps, long hidden)
{
    long index = thread_index(), b = index / hidden, j = index % hidden, state = batch * hidden;
    if (index >= state) return;
    long width = 4 * hidden;
    const real *row = gates + (t * batch + b) * width;
    real *drow = dinput_gates + (b * steps + t) * width;
    real dc = lstm_output_grad(row, cs[(t + 1) * state + index], dh[index], drow, j, hidden) + carry[index];
    carry[index] = lstm_memory_grad(row, cs[t * state + index], dc, drow, j, hidden);
}

/* out[b][j] = (grad slot t-1, if t > 0) + sum over n of dgates_t[b][n] W[n][j],
 * tiled like lstm_step_gates; dgates_t is dinput_gates' rows (b, t). */
KERNEL lstm_step_recurrent(const real *dinput_gates, const real *weight, const real *grad, real *out,
                           long t, long batch, long steps, long hidden)
{
    __shared__ real d_tile[LSTM_TILE][LSTM_TILE + 1];      /* dgates_t[b][n] */
    __shared__ real w_tile[LSTM_TILE][LSTM_TILE + 1];      /* W[n][j] */
    long width = 4 * hidden;
    long b = (long)blockIdx.y * LSTM_TILE + threadIdx.y, j = (long)blockIdx.x * LSTM_TILE + threadIdx.x;
    real sum = 0;
    for (long n0 = 0; n0 < width; n0 += LSTM_TILE) {
        long n = n0 + threadIdx.x, w_row = n0 + threadIdx.y;
        d_tile[threadIdx.y][threadIdx.x] = b < batch && n < width ? dinput_gates[(b * steps + t) * width + n] : (real)0;
        w_tile[threadIdx.y][threadIdx.x] = w_row < width && j < hidden ? weight[w_row * hidden + j] : (real)0;
        __syncthreads();
        for (int nn = 0; nn < LSTM_TILE; nn++) sum += d_tile[threadIdx.y][nn] * w_tile[nn][threadIdx.x];
        __syncthreads();
    }
    if (b < batch && j < hidden)
        out[b * hidden + j] = t > 0 ? grad[(b * (steps + 1) + t - 1) * hidden + j] + sum : sum;
}

/* dW[n][k] = sum over (b, t) of dgates[b, t][n] * h_t-1[b][k]: a tiled
 * product whose inner dimension runs over all batch x steps rows. */
KERNEL lstm_weight_grad(const real *dinput_gates, const real *hs, real *dweight, long batch, long steps, long hidden)
{
    __shared__ real d_tile[LSTM_TILE][LSTM_TILE + 1];      /* dgates[r][n], r = b * steps + t */
    __shared__ real h_tile[LSTM_TILE][LSTM_TILE + 1];      /* h_t-1[b][k] for the same r */
    long width = 4 * hidden, rows = batch * steps;
    long tile_n = (long)blockIdx.y * LSTM_TILE;
    long n = tile_n + threadIdx.y, k = (long)blockIdx.x * LSTM_TILE + threadIdx.x;
    real sum = 0;
    for (long r0 = 0; r0 < rows; r0 += LSTM_TILE) {
        long r = r0 + threadIdx.y, b = r / steps, t = r % steps, d_column = tile_n + threadIdx.x;
        d_tile[threadIdx.y][threadIdx.x] = r < rows && d_column < width ? dinput_gates[r * width + d_column] : (real)0;
        h_tile[threadIdx.y][threadIdx.x] = r < rows && k < hidden ? hs[(t * batch + b) * hidden + k] : (real)0;
        __syncthreads();
        for (int rr = 0; rr < LSTM_TILE; rr++) sum += d_tile[rr][threadIdx.y] * h_tile[rr][threadIdx.x];
        __syncthreads();
    }
    if (n < width && k < hidden) dweight[n * hidden + k] = sum;
}

/*
 * The CRF's forward algorithm (tagger.py's crf_forward): alpha_0 = e_0 +
 * start, then for each later word
 *
 *     m = the row max of alpha,   p = e^(alpha - m),   q = p @ weights,
 *     alpha = log(q) + m + e_t
 *
 * e is (batch, steps, tags), weights (tags, tags); ps and qs (steps - 1,
 * batch, tags) are kept for the backward pass; alpha (batch, tags) is the
 * last word's.
 *
 * Sentences are independent, so one block runs one sentence through all
 * its words, one thread per tag (looping if there are more tags than
 * threads), in a single launch: the threads of a block can wait for each
 * other between words (__syncthreads), which threads in different blocks
 * could not. The sentence's alpha and p live in shared memory (2 x tags
 * values, the launch's dynamic shared memory).
 */
KERNEL crf_forward(const real *e, const real *start, const real *weights, real *ps, real *qs, real *alpha_out,
                   long batch, long steps, long tags)
{
    extern __shared__ unsigned char crf_shared[];
    real *alpha = (real *)crf_shared, *p = alpha + tags;
    long b = blockIdx.x, n = batch * tags;
    const real *word = e + b * steps * tags;

    for (long j = threadIdx.x; j < tags; j += blockDim.x) alpha[j] = word[j] + start[j];
    __syncthreads();
    for (long t = 1; t < steps; t++) {
        real best = alpha[0];                              /* every thread finds the max: tags is small */
        for (long i = 1; i < tags; i++)
            if (alpha[i] > best) best = alpha[i];
        for (long i = threadIdx.x; i < tags; i += blockDim.x) {
            p[i] = exp(alpha[i] - best);
            ps[(t - 1) * n + b * tags + i] = p[i];
        }
        __syncthreads();                                   /* p complete, alpha read by all */
        for (long j = threadIdx.x; j < tags; j += blockDim.x) {
            real q = 0;
            for (long i = 0; i < tags; i++) q += p[i] * weights[i * tags + j];
            qs[(t - 1) * n + b * tags + j] = q;
            alpha[j] = log(q) + best + word[t * tags + j];
        }
        __syncthreads();                                   /* the new alpha complete, p read by all */
    }
    for (long j = threadIdx.x; j < tags; j += blockDim.x) alpha_out[b * tags + j] = alpha[j];
}

/*
 * Its backward pass, again one block per sentence, from the last word:
 * e_t's gradient is dalpha, then dq = dalpha / q, dp = dq @ weights^T and
 * the previous dalpha = dp * p. The weights' gradient sums p^T dq over
 * every sentence and word, across blocks, so this kernel keeps each dq (in
 * dqs, laid out like qs) and crf_weights_grad sums the products after it,
 * one thread per weight. start's gradient sums e_0's over the batch
 * (crf_start_grad).
 */
KERNEL crf_forward_backward(const real *weights, const real *ps, const real *qs, const real *grad,
                            real *de, real *dqs, long batch, long steps, long tags)
{
    extern __shared__ unsigned char crf_shared[];
    real *dalpha = (real *)crf_shared, *dq = dalpha + tags;
    long b = blockIdx.x, n = batch * tags;

    for (long j = threadIdx.x; j < tags; j += blockDim.x) dalpha[j] = grad[b * tags + j];
    __syncthreads();
    for (long t = steps - 1; t >= 1; t--) {
        const real *p = ps + (t - 1) * n + b * tags, *q = qs + (t - 1) * n + b * tags;
        for (long j = threadIdx.x; j < tags; j += blockDim.x) {
            de[(b * steps + t) * tags + j] = dalpha[j];
            dq[j] = dalpha[j] * ((real)1 / q[j]);
            dqs[(t - 1) * n + b * tags + j] = dq[j];
        }
        __syncthreads();                                   /* dq complete, dalpha read by all */
        for (long i = threadIdx.x; i < tags; i += blockDim.x) {
            real dp = 0;
            for (long j = 0; j < tags; j++) dp += dq[j] * weights[i * tags + j];
            dalpha[i] = dp * p[i];
        }
        __syncthreads();
    }
    for (long j = threadIdx.x; j < tags; j += blockDim.x) de[b * steps * tags + j] = dalpha[j];
}

/* dweights[i][j] = sum over the rows (words x sentences) of p[i] * dq[j]. */
KERNEL crf_weights_grad(const real *ps, const real *dqs, real *dweights, long rows, long tags)
{
    long index = thread_index(), i = index / tags, j = index % tags;
    if (index >= tags * tags) return;
    real sum = 0;
    for (long r = 0; r < rows; r++) sum += ps[r * tags + i] * dqs[r * tags + j];
    dweights[index] = sum;
}

/* dstart[j] = sum over sentences of e_0's gradient de[b][0][j]. */
KERNEL crf_start_grad(const real *de, real *dstart, long batch, long steps, long tags)
{
    long j = thread_index();
    if (j >= tags) return;
    real sum = 0;
    for (long b = 0; b < batch; b++) sum += de[b * steps * tags + j];
    dstart[j] = sum;
}

/* (end of recurrent) */
