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

/* C = op(A) @ op(B), batched, as in kernels.c's gemm: op(A) is m x k (A is
 * stored k x m when trans_a), op(B) is k x n (B is stored n x k when
 * trans_b), and batch entry z reads A + z * a_step, B + z * b_step.
 *
 * Why tiles. The plain kernel, one thread per output reading a row of A and
 * a column of B straight from global memory, reads 2k numbers for k
 * multiply-adds, and global memory is far too slow for that: an RTX 3060 Ti
 * does about 70 multiply-adds in the time it fetches one number. So a block
 * of threads computes a BM x BN *tile* of C together. It walks along k in
 * steps of BK: the block copies a BM x BK slab of op(A) and a BK x BN slab
 * of op(B) into shared memory (on-chip, about as fast as registers), and
 * every thread then reads its operands from there. Each number fetched from
 * global memory is now used BN (or BM) times instead of once.
 *
 * Why each thread computes several outputs (register blocking). Shared
 * memory is fast but not free: a thread computing one output still loads
 * two numbers per multiply-add. A thread that owns a TM x TN patch of the
 * tile loads TM numbers of A's column and TN of B's row, and combines them
 * into TM * TN multiply-adds on values held in registers: with 8 x 8, 64
 * multiply-adds for 16 loads. The patch's rows are taken in runs of 4 (for
 * 8 rows: rows 4ty..4ty+3 and the same 64 rows lower), so the threads of a
 * warp read neighbouring runs of shared memory, which costs no bank
 * conflicts and lets each run load as one 16-byte vector.
 *
 * Why two shared buffers. While the threads multiply step s from one
 * buffer, the numbers for step s + 1 are already on their way from global
 * memory into registers, and are then stored into the other buffer; so the
 * slow loads overlap the arithmetic, and one __syncthreads per step is
 * enough.
 *
 * Slabs are stored k-major (slab[p][i]) whatever the layout of A or B, so
 * the transposes only change how a slab is loaded: each load walks along
 * the matrix's contiguous direction, so neighbouring threads read
 * neighbouring numbers (a coalesced load). Outside the matrix (edges of
 * sizes that are not tile multiples) a load gives 0 and a store is skipped.
 *
 * Batches. With separate outputs (slices == 0) grid z runs over the batch
 * entries, entry z writing C + z * c_step. With one summed output (c_step 0
 * in kernels.c) the batch entries' k-steps are one long sum: a block walks
 * through entry 0's steps, then entry 1's, and so on, adding into the same
 * registers (no atomics, and a fixed order). When there are too few tiles
 * to fill the GPU, that long sum is split into `slices` parts (grid z),
 * each written to its own m x n partial result, and gemm_sum_slices adds
 * them up in order ("split-k").
 *
 * Matrix-vector products (one row or column of C) are done by gemm_dots,
 * one dot product per output, since a tile would be mostly empty. */

#define GEMM_THREADS 256
#define GEMM_PAD 4          /* slab rows padded: k-major stores of row-major data hit different banks */

/* A slab is ROWS x BK numbers of a matrix seen as rows x depth: op(A) as
 * m x k, or op(B) turned on its side as n x k. ALONG_ROWS says which way
 * it is stored: contiguous along the rows (element (row, p) at
 * p * rows + row), or along p (at row * depth + p). For op(A) that is
 * trans_a; for op(B) it is !trans_b (B stored k x n is contiguous along
 * op(B)'s columns, which are the rows here). */

/* The slab element that load number `r` of this thread covers: neighbouring
 * threads take neighbouring elements of the matrix's contiguous direction. */
template <int ROWS, int BK, bool ALONG_ROWS>
__device__ __forceinline__ void gemm_slab_cell(int r, int &row, int &p)
{
    int e = threadIdx.x + r * GEMM_THREADS;
    if (ALONG_ROWS) { row = e % ROWS; p = e / ROWS; }
    else            { row = e / BK;   p = e % BK;   }
}

/* This thread's share of the slab at (row0, p0) of X, into registers.
 * (Index arithmetic in int, as in kernels.c, whose sizes are int: 64-bit
 * multiplies are slow on a GPU. Only the final offset is widened.) */
template <int ROWS, int BK, bool ALONG_ROWS>
__device__ __forceinline__ void gemm_fetch(const real *X, int rows, int depth, int row0, int p0, real *out)
{
    #pragma unroll
    for (int r = 0; r < ROWS * BK / GEMM_THREADS; r++) {
        int row, p;
        gemm_slab_cell<ROWS, BK, ALONG_ROWS>(r, row, p);
        row += row0;
        p += p0;
        bool inside = row < rows && p < depth;
        out[r] = inside ? X[ALONG_ROWS ? (long)p * rows + row : (long)row * depth + p] : (real)0;
    }
}

/* ...and from registers into the slab in shared memory, k-major. */
template <int ROWS, int BK, bool ALONG_ROWS>
__device__ __forceinline__ void gemm_store(real (*slab)[ROWS + GEMM_PAD], const real *in)
{
    #pragma unroll
    for (int r = 0; r < ROWS * BK / GEMM_THREADS; r++) {
        int row, p;
        gemm_slab_cell<ROWS, BK, ALONG_ROWS>(r, row, p);
        slab[p][row] = in[r];
    }
}

/* The block's two pairs of slabs, in shared memory. The kernel declares
 * them and passes them down (a __shared__ array declared inside a template
 * function would get its own memory in every instantiation). */
template <int BM, int BN, int BK>
struct GemmSlabs {
    real a[2][BK][BM + GEMM_PAD];
    real b[2][BK][BN + GEMM_PAD];
};

/* Row (or column) `t` of this thread's TM-row patch, inside the tile: runs
 * of 4 (or TM, if smaller), the threads' runs side by side, then repeated
 * further down the tile. `lane` is the thread's row (ty) or column (tx). */
template <int BM, int TM>
__device__ __forceinline__ int gemm_patch_row(int lane, int t)
{
    const int run = TM < 4 ? TM : 4;
    return (t / run) * (BM / TM) * run + lane * run + t % run;
}

/* One block's tile of C (m x n): the products of steps first..last-1,
 * where step = batch * ktiles + (k tile), into C, or added to C if
 * accumulate. */
template <int BM, int BN, int BK, int TM, int TN, bool TRANS_A, bool TRANS_B>
__device__ __forceinline__ void gemm_block(GemmSlabs<BM, BN, BK> &slabs, const real *a, const real *b, real *C,
                                           int m, int k, int n, long a_step, long b_step,
                                           long first, long last, bool accumulate)
{
    auto &a_slab = slabs.a;
    auto &b_slab = slabs.b;
    /* This thread's patch is column tx, row ty of the (BN / TN) x (BM / TM)
     * patches. Each warp takes a 4-row by 8-column block of patches (not one
     * long row of them), so its 32 threads read only 4 different runs of
     * a_slab and 8 of b_slab per p: shared memory serves a run to every
     * thread that wants it at once, so fewer different runs mean fewer
     * cycles. */
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, warps_across = BN / TN / 8;
    const int tx = warp % warps_across * 8 + lane % 8, ty = warp / warps_across * 4 + lane / 8;
    const int i0 = blockIdx.y * BM, j0 = blockIdx.x * BN;
    const int ktiles = (k + BK - 1) / BK;

    real acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0;

    if (first < last) {
        const real *A = a + first / ktiles * a_step, *B = b + first / ktiles * b_step;
        int p0 = first % ktiles * BK;
        real a_next[BM * BK / GEMM_THREADS], b_next[BN * BK / GEMM_THREADS];
        gemm_fetch<BM, BK, TRANS_A>(A, m, k, i0, p0, a_next);
        gemm_fetch<BN, BK, !TRANS_B>(B, n, k, j0, p0, b_next);
        gemm_store<BM, BK, TRANS_A>(a_slab[0], a_next);
        gemm_store<BN, BK, !TRANS_B>(b_slab[0], b_next);
        __syncthreads();

        for (long step = first; step < last; step++) {
            const int now = (step - first) & 1;
            const bool more = step + 1 < last;
            if (more) {                                   /* start loading the next step */
                p0 += BK;
                if (p0 >= k) { p0 = 0; A += a_step; B += b_step; }   /* on to the next batch entry */
                gemm_fetch<BM, BK, TRANS_A>(A, m, k, i0, p0, a_next);
                gemm_fetch<BN, BK, !TRANS_B>(B, n, k, j0, p0, b_next);
            }
            #pragma unroll 2
            for (int p = 0; p < BK; p++) {                /* the TM x TN multiply-adds of each p */
                real a_part[TM], b_part[TN];
                #pragma unroll
                for (int i = 0; i < TM; i++) a_part[i] = a_slab[now][p][gemm_patch_row<BM, TM>(ty, i)];
                #pragma unroll
                for (int j = 0; j < TN; j++) b_part[j] = b_slab[now][p][gemm_patch_row<BN, TN>(tx, j)];
                #pragma unroll
                for (int i = 0; i < TM; i++)
                    #pragma unroll
                    for (int j = 0; j < TN; j++) acc[i][j] += a_part[i] * b_part[j];
            }
            if (more) {
                gemm_store<BM, BK, TRANS_A>(a_slab[now ^ 1], a_next);
                gemm_store<BN, BK, !TRANS_B>(b_slab[now ^ 1], b_next);
            }
            __syncthreads();
        }
    }

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int row = i0 + gemm_patch_row<BM, TM>(ty, i);
        if (row >= m) continue;
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int col = j0 + gemm_patch_row<BN, TN>(tx, j);
            if (col < n) {
                real *out = C + (long)row * n + col;
                *out = accumulate ? *out + acc[i][j] : acc[i][j];
            }
        }
    }
}

/* One launch: grid x, y over the tiles of C, z over batch entries
 * (slices == 0) or over slices of the summed batches (slices >= 1). */
template <int BM, int BN, int BK, int TM, int TN, bool TRANS_A, bool TRANS_B>
__device__ __forceinline__ void gemm_tiles(GemmSlabs<BM, BN, BK> &slabs, const real *a, const real *b, real *c, long batches,
                                           long m, long k, long n, long a_step, long b_step, long c_step,
                                           long accumulate, long slices)
{
    const long ktiles = (k + BK - 1) / BK, steps = batches * ktiles;
    const long share = slices ? (steps + slices - 1) / slices : ktiles;
    for (long z = blockIdx.z; z < (slices ? slices : batches); z += gridDim.z) {
        long first = z * share, last = first + share < steps ? first + share : steps;
        real *C = c + (slices ? z * m * n : z * c_step);
        gemm_block<BM, BN, BK, TM, TN, TRANS_A, TRANS_B>(slabs, a, b, C, m, k, n, a_step, b_step,
                                                         first, last, accumulate);
    }
}

/* The kernels: three tile sizes, all 256 threads, each compiled for the
 * four layouts (gemm_128x128_nt: A as it is, B transposed), so the loads
 * know their layout at compile time. kernels_cuda.gemm picks the tile per
 * shape and precision: the big tile reuses each load most; the small ones
 * give small products enough blocks to keep every multiprocessor busy.
 * BLOCKS is how many blocks should fit on one multiprocessor at once: the
 * compiler then keeps each thread within 65536 / (256 * BLOCKS) registers. */
#define GEMM_KERNEL(name, TRANS_A, TRANS_B, BM, BN, BK, TM, TN, BLOCKS)                              \
    KERNEL __launch_bounds__(GEMM_THREADS, BLOCKS)                                                  \
    name(const real *a, const real *b, real *c, long batches, long m, long k, long n,               \
         long a_step, long b_step, long c_step, long accumulate, long slices)                       \
    {                                                                                               \
        __shared__ __align__(16) GemmSlabs<BM, BN, BK> slabs;                                       \
        gemm_tiles<BM, BN, BK, TM, TN, TRANS_A, TRANS_B>(slabs, a, b, c, batches, m, k, n,          \
                                                         a_step, b_step, c_step, accumulate, slices); \
    }
#define GEMM_KERNELS(tile, BM, BN, BK, TM, TN, BLOCKS)                                              \
    GEMM_KERNEL(gemm_##tile##_nn, false, false, BM, BN, BK, TM, TN, BLOCKS)                         \
    GEMM_KERNEL(gemm_##tile##_nt, false, true, BM, BN, BK, TM, TN, BLOCKS)                          \
    GEMM_KERNEL(gemm_##tile##_tn, true, false, BM, BN, BK, TM, TN, BLOCKS)                          \
    GEMM_KERNEL(gemm_##tile##_tt, true, true, BM, BN, BK, TM, TN, BLOCKS)

GEMM_KERNELS(128x128, 128, 128, 8, 8, 8, 2)
GEMM_KERNELS(64x64, 64, 64, 16, 4, 4, 2)
GEMM_KERNELS(32x32, 32, 32, 16, 2, 2, 2)

/* Matrix-vector products (C has one column or one row), when the matrix
 * is stored along p: a tile would be one column wide, wasting nearly all
 * its work, so instead each output number is one dot product, done by one
 * warp. Its 32 threads take every 32nd p (neighbouring threads read
 * neighbouring numbers), and then add up their 32 partial sums by passing
 * values between registers (__shfl_down_sync: thread t gets thread
 * t + offset's value), halving the count each time. Output o's row of
 * op(A) starts at o * a_row and its column of op(B) at o * b_row (one of
 * them is 0: the vector, shared by every output). */
KERNEL gemm_dots(const real *a, const real *b, real *c, long batches, long outputs, long k,
                 long a_step, long b_step, long c_step, long a_row, long b_row, long summed, long accumulate)
{
    long warp = thread_index() / 32;
    int lane = threadIdx.x % 32;
    if (warp >= (summed ? outputs : batches * outputs)) return;      /* the whole warp leaves together */
    long o = warp % outputs, first = summed ? 0 : warp / outputs, last = summed ? batches : first + 1;
    real total = 0;
    for (long batch = first; batch < last; batch++) {
        const real *x = a + batch * a_step + o * a_row, *y = b + batch * b_step + o * b_row;
        for (long p = lane; p < k; p += 32) total += x[p] * y[p];
    }
    for (int offset = 16; offset > 0; offset /= 2) total += __shfl_down_sync(0xffffffff, total, offset);
    if (lane == 0) {
        real *out = c + first * c_step + o;
        *out = accumulate ? *out + total : total;
    }
}

/* c = (c if accumulate) + the sum of the split-k partial results, in order. */
KERNEL gemm_sum_slices(const real *partials, real *c, long count, long slices, long accumulate)
{
    long i = thread_index();
    if (i >= count) return;
    real total = accumulate ? c[i] : (real)0;
    for (long s = 0; s < slices; s++) total += partials[s * count + i];
    c[i] = total;
}

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
