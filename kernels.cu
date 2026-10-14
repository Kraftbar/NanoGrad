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

/* (end of rows) */

/* ------------------------------------------------------------------------
 * Recurrent: the LSTM cell and layer, the CRF.
 * --------------------------------------------------------------------- */

/* (end of recurrent) */
