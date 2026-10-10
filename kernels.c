/*
 * NanoGrad's optional compiled kernels.
 *
 * kernels.py compiles this file with the system C compiler the first time it
 * is needed and calls it through ctypes. If that is not possible, kernels.py
 * runs the pure-Python versions instead, which compute the same thing and
 * are the readable reference. Tests check that the two agree.
 *
 * Matrices are flat row-major arrays of `real` numbers, exactly like
 * Tensor.data: double by default. kernels_c.py also builds a float32 copy of
 * this file with -DREAL=float. Scalar settings (eps, learning rates) stay
 * double, and so do the running sums within one row or channel; sums
 * across rows (e.g. weight gradients) are stored as `real`.
 */

#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#ifndef REAL
#define REAL double
#endif
typedef REAL real;

/*
 * Threads (OpenMP, when the compiler supports -fopenmp). Only the matrix
 * products and convolutions use them, and only by splitting the output into
 * blocks along tile boundaries: every output number is still summed by one
 * thread, in the same order and by the same code as with one thread, so the
 * results are the same, bit for bit, for any thread count.
 *
 * With one thread (the default) no parallel region is ever entered, so a
 * process forked from a single-threaded one is safe (libgomp's thread pool
 * does not survive fork; kernels_c.py drops forked children to 1 thread).
 */
#ifdef _OPENMP
#include <omp.h>
#define OMP(directive) _Pragma(#directive)
#else
#define OMP(directive)
#endif

static int thread_limit = 1;

void set_threads(int threads)
{
#ifdef _OPENMP
    thread_limit = threads < 1 ? 1 : threads;
#else
    (void)threads;
#endif
}

int get_threads(void)
{
    return thread_limit;
}

/* How many threads a job of `work` multiply-adds, split into `units`
 * independent blocks, is worth: each must get at least ~1M multiply-adds
 * (about 25 microseconds), or starting the threads costs more than they
 * save (measured: 256K made a small vision transformer slower, 4M left a
 * ResNet's smaller layers on one thread). It sets how many blocks a job is
 * cut into; a parallel region itself always asks for all thread_limit
 * threads, because libgomp ends the surplus threads when a region asks for
 * fewer than the last one and starts new ones when a later region asks for
 * more, and starting threads on every product is slow. */
static inline int team_size(double work, long units)
{
    long by_work = (long)(work / (1 << 20));
    long team = thread_limit;
    if (team > by_work) team = by_work;
    if (team > units) team = units;
    return team < 1 ? 1 : (int)team;
}

/* Block t of `units` blocks shared out among `team` threads: [*lo, *hi). */
static inline void share(long units, int team, int t, long *lo, long *hi)
{
    *lo = units * t / team;
    *hi = units * (t + 1) / team;
}

/*
 * Batched matrix product, accumulated into c:
 *
 *     for every batch:  C += op(A) @ op(B)
 *
 *     op(A) is (m x k): A itself, or A^T when trans_a (A stored as k x m)
 *     op(B) is (k x n): B itself, or B^T when trans_b (B stored as n x k)
 *
 * a_step, b_step and c_step are the offsets between consecutive batch
 * entries. A step of 0 reuses the same matrix for every batch; for c that
 * means the batch results are summed, which is what a gradient needs when a
 * weight matrix is shared across the batch.
 */
/* Entry (i, p) of op(A): A itself is (m x k), or (k x m) when transposed. */
#define A_AT(i, p) (trans_a ? A[(long)(p) * m + (i)] : A[(long)(i) * k + (p)])

/*
 * The fast path, used when the compiler may emit wide vector instructions
 * (AVX: 32 bytes at once, AVX-512: 64 bytes, i.e. 8 doubles or 16 floats;
 * kernels.py builds with -march=native).
 * It is the core idea of every fast BLAS, in three steps:
 *
 * 1. Register tile: compute a TILE_ROWS x (2 vectors) block of C at a time,
 *    kept in vector registers for the whole sum over p. Each step loads one
 *    row slice of B (2 vectors) and TILE_ROWS numbers of A, and does
 *    2 * TILE_ROWS vector multiply-adds, instead of loading and storing rows
 *    of C every step. AVX-512 has 32 vector registers, so 12 rows (24
 *    accumulators); AVX2 has 16, so 6 rows (measured: each is 10-35% faster
 *    than 8 rows on its own hardware).
 * 2. Packing: copy the B columns this tile needs into a small contiguous
 *    "panel", and each block of A tile by tile, so the inner loop reads
 *    memory in order whichever way A and B are stored.
 * 3. Blocking: do the sum over p in chunks of 256, and the rows of A in
 *    blocks of about 64K numbers, so one block of A stays in cache while
 *    every column panel of B passes over it (and the panel in the fastest
 *    cache while the block's row tiles pass over it).
 *
 * Vector types are a GCC/Clang extension: `vec` is VECTOR_WIDTH doubles
 * that + and * act on elementwise. Without AVX the plain loop below is
 * used, since emulating wide vectors would only slow things down.
 */
#if defined(__AVX512F__)
#define VECTOR_BYTES 64
#elif defined(__AVX__)
#define VECTOR_BYTES 32
#endif

#ifdef VECTOR_BYTES
#define VECTOR_WIDTH (VECTOR_BYTES / (int)sizeof(real))   /* 8 doubles or 16 floats */
typedef real vec __attribute__((vector_size(VECTOR_BYTES)));
#if VECTOR_BYTES == 64
#define TILE_ROWS 12
#else
#define TILE_ROWS 6
#endif
#define TILE_COLS (2 * VECTOR_WIDTH)
#define K_BLOCK 256
#define A_BLOCK 65536          /* numbers of op(A) per cached block (m_block * kb) */
#define VEC_LOAD(v, p) memcpy(&(v), (p), sizeof(vec))
#define VEC_STORE(p, v) memcpy((p), &(v), sizeof(vec))

/* One register tile: C[r, c] += sum over p < kb of a(r, p) * panel[p, c]
 * for rows (<= TILE_ROWS) rows and cols (<= TILE_COLS) columns, where
 * a(r, p) = A[r * row_step + p * col_step] (so A can be a plain matrix, its
 * transpose, or a packed block) and C rows are n apart. The panel holds
 * B's columns, zero-padded past cols, with rows panel_step apart
 * (TILE_COLS when packed; n when the tile reads B in place). Always inlined,
 * so the full-tile call (rows = TILE_ROWS, cols = TILE_COLS) gets constants
 * the compiler can unroll. */
static inline __attribute__((always_inline)) void tile(
    const real *A, long row_step, long col_step, real *C, int n,
    const real *panel, long panel_step, int kb, int rows, int cols)
{
    vec acc[TILE_ROWS][2];
    memset(acc, 0, sizeof acc);
#pragma GCC unroll 4          /* lets the scheduler overlap iterations: float64 +25% */
    for (int p = 0; p < kb; p++) {
        vec b0, b1;
        VEC_LOAD(b0, panel + (long)p * panel_step);
        VEC_LOAD(b1, panel + (long)p * panel_step + VECTOR_WIDTH);
        for (int r = 0; r < rows; r++) {
            real a_rp = A[r * row_step + p * col_step];
            acc[r][0] += a_rp * b0;
            acc[r][1] += a_rp * b1;
        }
    }
    for (int r = 0; r < rows; r++) {
        real *c_row = C + (long)r * n;
        if (cols == TILE_COLS) {
            vec c0, c1;
            VEC_LOAD(c0, c_row);
            VEC_LOAD(c1, c_row + VECTOR_WIDTH);
            c0 += acc[r][0];
            c1 += acc[r][1];
            VEC_STORE(c_row, c0);
            VEC_STORE(c_row + VECTOR_WIDTH, c1);
        } else {                                   /* a partial tile at the right edge */
            real sums[TILE_COLS];
            VEC_STORE(sums, acc[r][0]);
            VEC_STORE(sums + VECTOR_WIDTH, acc[r][1]);
            for (int c = 0; c < cols; c++) c_row[c] += sums[c];
        }
    }
}

/* Every register tile down one column panel: rows of C, TILE_ROWS at a
 * time. Row tile t starts at A + t * tile_step. Inlined at each call, so
 * constant steps (the packed block) are folded into the inner loop. */
static inline __attribute__((always_inline)) void row_tiles(
    const real *A, long row_step, long col_step, long tile_step,
    real *C, int n, const real *panel, long panel_step, int kb, int rows, int cols)
{
    int i = 0;
    for (; i + TILE_ROWS <= rows; i += TILE_ROWS) {
        const real *a = A + (i / TILE_ROWS) * tile_step;
        if (cols == TILE_COLS)
            tile(a, row_step, col_step, C + (long)i * n, n, panel, panel_step, kb, TILE_ROWS, TILE_COLS);
        else
            tile(a, row_step, col_step, C + (long)i * n, n, panel, panel_step, kb, TILE_ROWS, cols);
    }
    if (i < rows)                                              /* leftover rows */
        tile(A + (i / TILE_ROWS) * tile_step, row_step, col_step, C + (long)i * n, n,
             panel, panel_step, kb, rows - i, cols);
}

/* C (m x n) += op(A) (m x k) @ op(B) (k x n), for rows i_lo..i_hi-1 and
 * columns j_lo..j_hi-1 of C only (a thread's share; i_lo is a multiple of
 * TILE_ROWS and j_lo of TILE_COLS, so the tiles are the same as for the
 * whole matrix). Packing reads B either way, so op(B) = B^T needs no
 * transposed copy.
 * panel: K_BLOCK * TILE_COLS numbers; packed: A_BLOCK + K_BLOCK * TILE_ROWS. */
static void gemm_tiled(const real *A, const real *B, real *C,
                       int m, int k, int n, int trans_a, int trans_b, real *panel, real *packed,
                       int i_lo, int i_hi, int j_lo, int j_hi)
{
    for (int p0 = 0; p0 < k; p0 += K_BLOCK) {
        int kb = k - p0 < K_BLOCK ? k - p0 : K_BLOCK;
        int m_block = A_BLOCK / kb / TILE_ROWS * TILE_ROWS;
        if (m_block < TILE_ROWS) m_block = TILE_ROWS;
        for (int i0 = i_lo; i0 < i_hi; i0 += m_block) {       /* 3. a block of A stays cached */
            int i_end = i_hi - i0 < m_block ? i_hi : i0 + m_block;
            /* 2. pack the block: TILE_ROWS rows at a time, a(r, p) at [p * TILE_ROWS + r]
             * (worth it only when several column tiles will reuse it) */
            int pack_a = j_hi - j_lo >= 4 * TILE_COLS;
            if (pack_a && trans_a) {
                /* A^T: row p of A holds a(., p) for every row of the block, so
                 * read each once, in order, and deal it out to the row tiles. */
                for (int p = 0; p < kb; p++) {
                    const real *src = A + (long)(p0 + p) * m;
                    for (int i = i0; i < i_end; i += TILE_ROWS) {
                        real *dst = packed + (long)(i - i0) * kb + p * TILE_ROWS;
                        int rows = i_end - i < TILE_ROWS ? i_end - i : TILE_ROWS;
                        for (int r = 0; r < rows; r++) dst[r] = src[i + r];
                    }
                }
            } else {
                for (int i = i0; pack_a && i < i_end; i += TILE_ROWS) {
                    real *dst = packed + (long)(i - i0) * kb;
                    int rows = i_end - i < TILE_ROWS ? i_end - i : TILE_ROWS;
                    for (int p = 0; p < kb; p++)
                        for (int r = 0; r < rows; r++) dst[p * TILE_ROWS + r] = A_AT(i + r, p0 + p);
                }
            }
            for (int j = j_lo; j < j_hi; j += TILE_COLS) {
                int cols = n - j < TILE_COLS ? n - j : TILE_COLS;
                const real *b_panel = panel;
                long b_step = TILE_COLS;
                if (!trans_b && cols == TILE_COLS && i_end - i0 <= 2 * TILE_ROWS) {
                    /* Few rows: the panel would serve only one or two row
                     * tiles, so read B where it is instead of copying it. */
                    b_panel = B + (long)p0 * n + j;
                    b_step = n;
                } else {
                if (cols < TILE_COLS)                            /* 2. pack B (zero-padded) */
                    memset(panel, 0, sizeof(real) * K_BLOCK * TILE_COLS);
                if (trans_b) {                                   /* op(B)[p, c] = B[c, p] */
                    for (int c = 0; c < cols; c++) {
                        const real *b_row = B + (long)(j + c) * k + p0;
                        for (int p = 0; p < kb; p++) panel[(long)p * TILE_COLS + c] = b_row[p];
                    }
                } else {
                    for (int p = 0; p < kb; p++)
                        memcpy(panel + (long)p * TILE_COLS, B + (long)(p0 + p) * n + j, sizeof(real) * cols);
                }
                }
                if (pack_a)                                      /* 1. register tiles */
                    row_tiles(packed, 1, TILE_ROWS, (long)TILE_ROWS * kb, C + (long)i0 * n + j, n,
                              b_panel, b_step, kb, i_end - i0, cols);
                else if (trans_a)
                    row_tiles(A + (long)p0 * m + i0, 1, m, TILE_ROWS, C + (long)i0 * n + j, n,
                              b_panel, b_step, kb, i_end - i0, cols);
                else
                    row_tiles(A + (long)i0 * k + p0, k, 1, (long)TILE_ROWS * k, C + (long)i0 * n + j, n,
                              b_panel, b_step, kb, i_end - i0, cols);
            }
        }
    }
}
#endif

#ifdef VECTOR_WIDTH
/* Block `chunk` of `chunks` of the tiled product. If each batch entry has
 * its own C (by_batch), a block is a run of whole entries; otherwise (one
 * entry, or entries summed into one C) it is a band of C: row tiles
 * (by_rows) or column tiles, done for every entry in order. Either way each
 * number's sum is done by one thread, in the same order as with one. */
static void gemm_chunk(const real *a, const real *b, real *c,
                       int batches, int m, int k, int n,
                       long a_step, long b_step, long c_step,
                       int trans_a, int trans_b, int by_batch, int by_rows,
                       int chunk, int chunks, real *panel, real *packed)
{
    long lo, hi;
    if (by_batch) {
        share(batches, chunks, chunk, &lo, &hi);
        for (long batch = lo; batch < hi; batch++)
            gemm_tiled(a + batch * a_step, b + batch * b_step, c + batch * c_step,
                       m, k, n, trans_a, trans_b, panel, packed, 0, m, 0, n);
        return;
    }
    int i_lo = 0, i_hi = m, j_lo = 0, j_hi = n;
    if (by_rows) {
        share((m + TILE_ROWS - 1) / TILE_ROWS, chunks, chunk, &lo, &hi);
        i_lo = lo * TILE_ROWS;
        i_hi = hi * TILE_ROWS < m ? hi * TILE_ROWS : m;
    } else {
        share((n + TILE_COLS - 1) / TILE_COLS, chunks, chunk, &lo, &hi);
        j_lo = lo * TILE_COLS;
        j_hi = hi * TILE_COLS < n ? hi * TILE_COLS : n;
    }
    for (long batch = 0; i_lo < i_hi && j_lo < j_hi && batch < batches; batch++)
        gemm_tiled(a + batch * a_step, b + batch * b_step, c + batch * c_step,
                   m, k, n, trans_a, trans_b, panel, packed, i_lo, i_hi, j_lo, j_hi);
}

/* Each thread's packing buffers, allocated by its first product and then
 * kept: allocating them (about half a megabyte) on every call costs more
 * than a small product. They are one block, freed by a thread-specific
 * key's destructor when the thread ends; without that, every thread
 * libgomp replaced leaked its buffers. */
static _Thread_local real *thread_panel, *thread_packed;
static pthread_key_t buffers_key;
static pthread_once_t buffers_key_once = PTHREAD_ONCE_INIT;

static void make_buffers_key(void)
{
    if (pthread_key_create(&buffers_key, free) != 0) {
        abort();
    }
}

static void thread_buffers(real **panel, real **packed)
{
    if (thread_panel == NULL) {
        size_t panel_count = (size_t)K_BLOCK * TILE_COLS;         /* a multiple of 16: packed stays aligned */
        size_t bytes = sizeof(real) * (panel_count + A_BLOCK + (size_t)K_BLOCK * TILE_ROWS);
        real *block = aligned_alloc(64, (bytes + 63) / 64 * 64);  /* cache-line aligned vector loads */
        if (block == NULL) {
            abort();
        }
        pthread_once(&buffers_key_once, make_buffers_key);
        pthread_setspecific(buffers_key, block);
        thread_panel = block;
        thread_packed = block + panel_count;
    }
    *panel = thread_panel;
    *packed = thread_packed;
}
#endif

void gemm(
    const real *a, const real *b, real *c,
    int batches, int m, int k, int n,
    long a_step, long b_step, long c_step,
    int trans_a, int trans_b)
{
#ifdef VECTOR_WIDTH
    if (n >= TILE_COLS) {
        long row_tiles_count = (m + TILE_ROWS - 1) / TILE_ROWS;
        long col_tiles_count = (n + TILE_COLS - 1) / TILE_COLS;
        int by_batch = c_step != 0 && batches >= thread_limit;
        int by_rows = row_tiles_count > col_tiles_count;
        long units = by_batch ? batches : by_rows ? row_tiles_count : col_tiles_count;
        int team = team_size((double)batches * m * k * n, units);
        /* Four blocks per thread, handed out as threads come free: a thread
         * slowed by another program on its core then does fewer of them. */
        int chunks = team == 1 ? 1 : (units < 4L * team ? (int)units : 4 * team);
        OMP(omp parallel num_threads(thread_limit) if (team > 1))
        {
            real *panel, *packed;
            thread_buffers(&panel, &packed);
            OMP(omp for schedule(dynamic, 1))
            for (int chunk = 0; chunk < chunks; chunk++)
                gemm_chunk(a, b, c, batches, m, k, n, a_step, b_step, c_step, trans_a, trans_b,
                           by_batch, by_rows, chunk, chunks, panel, packed);
        }
        return;
    }
#endif
    /* Without the tiled path, op(B) = B^T is first copied into a scratch
     * (k x n) buffer once per batch entry, so the inner loop below always
     * walks contiguous rows of op(B). (Summing dot products instead would be
     * a reduction, which compilers will not vectorize without permission to
     * reorder additions.) */
    real *scratch = trans_b ? malloc(sizeof(real) * (size_t)k * n) : NULL;
    if (trans_b && scratch == NULL) {
        abort();  /* out of memory: fail loudly rather than return zeros */
    }

    for (int batch = 0; batch < batches; batch++) {
        const real *A = a + batch * a_step;
        const real *B = b + batch * b_step;
        real *C = c + batch * c_step;

        if (trans_b) {
            for (int j = 0; j < n; j++)
                for (int p = 0; p < k; p++)
                    scratch[(long)p * n + j] = B[(long)j * k + p];
            B = scratch;
        }

        /* Row i of C is a weighted sum of rows of op(B):
         * c_row += A'(i, p) * op(B)[p, :] for each p. The inner loop runs
         * along contiguous memory, which compilers vectorize well.
         *
         * Rows of C are done four at a time, so every row of op(B) loaded
         * from memory is used four times. (Any leftover rows, one by one.) */
        int i = 0;
        for (; i + 4 <= m; i += 4) {
            real *c0 = C + (long)i * n, *c1 = c0 + n, *c2 = c1 + n, *c3 = c2 + n;
            for (int p = 0; p < k; p++) {
                real a0 = A_AT(i, p), a1 = A_AT(i + 1, p), a2 = A_AT(i + 2, p), a3 = A_AT(i + 3, p);
                const real *b_row = B + (long)p * n;
                for (int j = 0; j < n; j++) {
                    real b = b_row[j];
                    c0[j] += a0 * b;
                    c1[j] += a1 * b;
                    c2[j] += a2 * b;
                    c3[j] += a3 * b;
                }
            }
        }
        for (; i < m; i++) {
            real *c_row = C + (long)i * n;
            for (int p = 0; p < k; p++) {
                real a_ip = A_AT(i, p);
                const real *b_row = B + (long)p * n;
                for (int j = 0; j < n; j++) {
                    c_row[j] += a_ip * b_row[j];
                }
            }
        }
    }
    free(scratch);
}

/*
 * Convolution forward without building the patch matrix ("implicit GEMM").
 *
 *     out_b (filters x positions) = kernel (filters x K) @ patches_b^T
 *     patches_b[j][q] = image_b[index[j * K + q]]        (K = patch_size)
 *
 * The tiled product reads its B panel straight from the image through the
 * index table, instead of first gathering every patch into a (positions x
 * K) matrix and transposing it: same arithmetic, two fewer large copies.
 */
void conv_forward(const real *image, const long *index, const real *kernel, real *out,
                  long batches, long image_size, int filters, int patch_size, int positions)
{
    const int trans_a = 0, m = filters, k = patch_size, n = positions;
    const real *A = kernel;
    (void)trans_a;
#ifdef VECTOR_WIDTH
    /* Threads share out (image, column tile) pairs; each writes its own block of out. */
    long col_tiles = (n + TILE_COLS - 1) / TILE_COLS, units = batches * col_tiles;
    int team = team_size((double)batches * m * k * n, units);
    int chunks = team == 1 ? 1 : (units < 4L * team ? (int)units : 4 * team);
    OMP(omp parallel num_threads(thread_limit) if (team > 1))
    {
        real *panel, *packed;
        thread_buffers(&panel, &packed);
        OMP(omp for schedule(dynamic, 1))
        for (int chunk = 0; chunk < chunks; chunk++) {
            long lo, hi;
            share(units, chunks, chunk, &lo, &hi);
            for (long unit = lo; unit < hi; unit++) {
                long batch = unit / col_tiles;
                int j = (int)(unit % col_tiles) * TILE_COLS;
                const real *img = image + batch * image_size;
                real *C = out + batch * (long)m * n;
                int cols = n - j < TILE_COLS ? n - j : TILE_COLS;
                for (int p0 = 0; p0 < k; p0 += K_BLOCK) {
                    int kb = k - p0 < K_BLOCK ? k - p0 : K_BLOCK;
                    if (cols < TILE_COLS) memset(panel, 0, sizeof(real) * K_BLOCK * TILE_COLS);
                    /* Pack through the index. Every entry is a position's base plus a patch
                     * offset: index[j * k + q] = index[j * k] + index[q]. Positions next to
                     * each other in an output row have consecutive bases, so then each
                     * panel row is a plain copy of `cols` neighbouring pixels. */
                    long base = index[(long)j * k];
                    int contiguous = index[(long)(j + cols - 1) * k] - base == cols - 1;
                    for (int p = 0; p < kb; p++) {
                        real *dst = panel + (long)p * TILE_COLS;
                        const real *src = img + index[p0 + p];
                        if (contiguous) {
                            for (int jj = 0; jj < cols; jj++) dst[jj] = src[base + jj];
                        } else {
                            for (int jj = 0; jj < cols; jj++) dst[jj] = src[index[(long)(j + jj) * k]];
                        }
                    }
                    row_tiles(A + p0, k, 1, (long)TILE_ROWS * k, C + j, n, panel, TILE_COLS, kb, m, cols);
                }
            }
        }
    }
#else
    for (long batch = 0; batch < batches; batch++) {           /* without the vector path */
        const real *img = image + batch * image_size;
        real *C = out + batch * (long)m * n;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double total = 0.0;
                const long *cells = index + (long)j * k;
                for (int q = 0; q < k; q++) total += A_AT(i, q) * img[cells[q]];
                C[(long)i * n + j] += total;
            }
        }
    }
#endif
}

/*
 * The kernel gradient of conv_forward, also without a patch matrix:
 *
 *     dkernel (filters x K) += sum over images of grad_b (filters x positions) @ patches_b
 *
 * Here the product's inner dimension is the positions, and row p of B is
 * patch p, read through the index table while packing the panel.
 */
void conv_kernel_grad(const real *image, const long *index, const real *grad, real *dkernel,
                      long batches, long image_size, int filters, int patch_size, int positions)
{
    const int trans_a = 0, m = filters, k = positions, n = patch_size;
    (void)trans_a;
#ifdef VECTOR_WIDTH
    /* Every image adds into the same dkernel, so threads share out blocks
     * of dkernel instead (column tiles, and bands of row tiles when there
     * are few columns); each thread adds the images into its blocks in order. */
    long col_tiles = (n + TILE_COLS - 1) / TILE_COLS, row_tiles_count = (m + TILE_ROWS - 1) / TILE_ROWS;
    long bands = (thread_limit + col_tiles - 1) / col_tiles;
    if (bands > row_tiles_count) bands = row_tiles_count;
    long units = col_tiles * bands;
    int team = team_size((double)batches * m * k * n, units);
    if (team == 1) bands = 1, units = col_tiles;
    int chunks = team == 1 ? 1 : (units < 4L * team ? (int)units : 4 * team);
    OMP(omp parallel num_threads(thread_limit) if (team > 1))
    {
        real *panel, *packed;
        thread_buffers(&panel, &packed);
        OMP(omp for schedule(dynamic, 1))
        for (int chunk = 0; chunk < chunks; chunk++) {
            long lo, hi;
            share(units, chunks, chunk, &lo, &hi);
            for (long batch = 0; batch < batches; batch++) {      /* images outermost: each is reused */
                const real *img = image + batch * image_size;     /* by every block while cached */
                const real *A = grad + batch * (long)m * k;
                for (long unit = lo; unit < hi; unit++) {
                    int j = (int)(unit / bands) * TILE_COLS;
                    long band_lo, band_hi;
                    share(row_tiles_count, (int)bands, (int)(unit % bands), &band_lo, &band_hi);
                    int i_lo = (int)band_lo * TILE_ROWS;
                    int i_hi = band_hi * TILE_ROWS < m ? (int)band_hi * TILE_ROWS : m;
                    int cols = n - j < TILE_COLS ? n - j : TILE_COLS;
                    for (int p0 = 0; p0 < k; p0 += K_BLOCK) {
                        int kb = k - p0 < K_BLOCK ? k - p0 : K_BLOCK;
                        if (cols < TILE_COLS) memset(panel, 0, sizeof(real) * K_BLOCK * TILE_COLS);
                        /* Pack patch rows. index[pos * n + q] = index[pos * n] + index[q] (a position's
                         * base plus a patch offset), so the (huge) index table is read once per
                         * position here instead of once per number. */
                        if (k >= 4096) {                       /* large images: column by column, */
                            long base[K_BLOCK];                /* each a run of neighbouring pixels */
                            for (int p = 0; p < kb; p++) base[p] = index[(long)(p0 + p) * n];
                            for (int jj = 0; jj < cols; jj++) {
                                const real *src = img + index[j + jj];
                                for (int p = 0; p < kb; p++) panel[(long)p * TILE_COLS + jj] = src[base[p]];
                            }
                        } else {                               /* small ones: row by row */
                            long offset[TILE_COLS];
                            for (int jj = 0; jj < cols; jj++) offset[jj] = index[j + jj];
                            for (int p = 0; p < kb; p++) {
                                const real *src = img + index[(long)(p0 + p) * n];
                                real *dst = panel + (long)p * TILE_COLS;
                                for (int jj = 0; jj < cols; jj++) dst[jj] = src[offset[jj]];
                            }
                        }
                        row_tiles(A + (long)i_lo * k + p0, k, 1, (long)TILE_ROWS * k, dkernel + (long)i_lo * n + j,
                                  n, panel, TILE_COLS, kb, i_hi - i_lo, cols);
                    }
                }
            }
        }
    }
#else
    for (long batch = 0; batch < batches; batch++) {           /* without the vector path */
        const real *img = image + batch * image_size;
        const real *A = grad + batch * (long)m * k;
        for (int i = 0; i < m; i++) {
            for (int q = 0; q < n; q++) {
                double total = 0.0;
                for (int p = 0; p < k; p++) total += A_AT(i, p) * img[index[(long)p * n + q]];
                dkernel[(long)i * n + q] += total;
            }
        }
    }
#endif
}

/*
 * The image gradient of conv_forward ("col2im"), without a patch-gradient
 * matrix: for each image,
 *
 *     dpatches (positions x K) = grad_b^T (positions x filters) @ kernel (filters x K)
 *     image_grad_b[index[j * K + q]] += dpatches[j][q]
 *
 * computed TILE_ROWS positions at a time into a small block that stays in
 * cache, then scattered straight onto the pixels the patches came from.
 */
void conv_input_grad(const real *grad, const real *kernel, const long *index, real *image_grad,
                     long batches, long image_size, int filters, int patch_size, int positions)
{
    const int trans_a = 1, m = positions, k = filters, n = patch_size;
    (void)trans_a;
#ifdef VECTOR_WIDTH
    /* The kernel is small and the same for every block of positions, so
     * it is packed into column panels once: panel j/TILE_COLS holds its
     * columns j.. j+TILE_COLS-1 (zero-padded), row p at [p * TILE_COLS]. */
    const int block_rows = 4 * TILE_ROWS;
    int panels = (n + TILE_COLS - 1) / TILE_COLS;
    real *packed = calloc((size_t)panels * k * TILE_COLS, sizeof(real));
    if (packed == NULL) {
        abort();
    }
    for (int j = 0; j < n; j += TILE_COLS) {
        int cols = n - j < TILE_COLS ? n - j : TILE_COLS;
        real *panel = packed + (long)(j / TILE_COLS) * k * TILE_COLS;
        for (int p = 0; p < k; p++)
            memcpy(panel + (long)p * TILE_COLS, kernel + (long)p * n + j, sizeof(real) * cols);
    }
#else
    const int block_rows = 8;
#endif
    /* Threads share out the images (neighbouring patches overlap, so one
     * image's pixels are only ever added to by one thread). */
#ifdef VECTOR_WIDTH
    int team = team_size((double)batches * m * k * n, batches);
#else
    int team = 1;
#endif
    (void)team;                                                /* used only with OpenMP */
    OMP(omp parallel num_threads(thread_limit) if (team > 1))
    {
        real *block = malloc(sizeof(real) * block_rows * n);
        if (block == NULL) {
            abort();
        }
        OMP(omp for schedule(dynamic, 1))
        for (long batch = 0; batch < batches; batch++) {
            const real *A = grad + batch * (long)k * m;           /* (filters x positions) */
            real *img_grad = image_grad + batch * image_size;
            for (int i = 0; i < m; i += block_rows) {
                int rows = m - i < block_rows ? m - i : block_rows;
                memset(block, 0, sizeof(real) * rows * n);
#ifdef VECTOR_WIDTH
                for (int j = 0; j < n; j += TILE_COLS) {
                    int cols = n - j < TILE_COLS ? n - j : TILE_COLS;
                    const real *panel = packed + (long)(j / TILE_COLS) * k * TILE_COLS;
                    for (int p0 = 0; p0 < k; p0 += K_BLOCK) {
                        int kb = k - p0 < K_BLOCK ? k - p0 : K_BLOCK;
                        /* A + i: positions i.. of grad_b^T become the block's rows 0.. */
                        row_tiles(A + i + (long)p0 * m, 1, m, TILE_ROWS, block + j, n,
                                  panel + (long)p0 * TILE_COLS, TILE_COLS, kb, rows, cols);
                    }
                }
#else
                for (int r = 0; r < rows; r++) {                   /* without the vector path */
                    for (int q = 0; q < n; q++) {
                        double total = 0.0;
                        for (int p = 0; p < k; p++) total += A[(long)p * m + i + r] * kernel[(long)p * n + q];
                        block[(long)r * n + q] += total;
                    }
                }
#endif
                for (int r = 0; r < rows; r++) {                   /* col2im */
                    /* index[pos * n + q] = index[pos * n] + index[q]: the patch offsets
                     * (the first n entries) stay cached; one base per position. */
                    real *dst = img_grad + index[(long)(i + r) * n];
                    const real *values = block + (long)r * n;
                    for (int q = 0; q < n; q++) dst[index[q]] += values[q];
                }
            }
        }
        free(block);
    }
#ifdef VECTOR_WIDTH
    free(packed);
#endif
}

/*
 * Direct convolution, stride 1, for layers with few filters. With 8
 * filters, packing patches for the matrix product above costs more than
 * the multiplications; here nothing is packed. Each output row is cut into
 * vectors of VECTOR_WIDTH pixels, and for 8 filters at once
 *
 *     out[f, y, x:x+V] += kernel[f, c, i, j] * image[c, y+i, x+j : x+j+V]
 *
 * over every (c, i, j): one vector load serves 8 multiply-adds, and the 8
 * sums stay in registers until the row segment is done.
 * image (B, C, H, W), kernel (F, C, kh, kw), out (B, F, H-kh+1, W-kw+1).
 */
#define DIRECT_GROUP 8

#ifdef VECTOR_WIDTH
/* V values from src, or the ones before `end` (zeros after) near the end of the buffer. */
static inline vec load_until(const real *src, const real *end)
{
    vec v;
    if (src + VECTOR_WIDTH <= end) {
        VEC_LOAD(v, src);
    } else {
        real tmp[VECTOR_WIDTH] = {0};
        for (long l = 0; src + l < end; l++) tmp[l] = src[l];
        VEC_LOAD(v, tmp);
    }
    return v;
}

/* One row segment of `valid` <= V outputs for a group of 8 filters, of
 * which the first `stored` are written; weights are packed [c][i][j][g]. */
static inline void direct_segment(
    const real *image, const real *end, const real *weights, real *out, long out_plane,
    int channels, int height, int width, int kh, int kw, int y, int x0, int valid, int stored)
{
    const int group = DIRECT_GROUP;
    vec acc[DIRECT_GROUP];
    memset(acc, 0, sizeof acc);
    const real *w = weights;
    for (int c = 0; c < channels; c++)
        for (int i = 0; i < kh; i++) {
            const real *row = image + ((long)c * height + y + i) * width + x0;
            for (int j = 0; j < kw; j++, w += group) {
                vec v = load_until(row + j, end);     /* lanes past the row only feed outputs not stored */
                for (int g = 0; g < group; g++) acc[g] += w[g] * v;
            }
        }
    for (int g = 0; g < stored; g++) {
        real *dst = out + g * out_plane;
        if (valid == VECTOR_WIDTH) {
            VEC_STORE(dst, acc[g]);
        } else {
            real tmp[VECTOR_WIDTH];
            VEC_STORE(tmp, acc[g]);
            for (int l = 0; l < valid; l++) dst[l] = tmp[l];
        }
    }
}
#endif

void conv_direct(const real *image, const real *kernel, real *out, long batches,
                 int channels, int height, int width, int filters, int kh, int kw)
{
    int oh = height - kh + 1, ow = width - kw + 1;
    long patch = (long)channels * kh * kw, image_size = (long)channels * height * width;
#ifdef VECTOR_WIDTH
    /* Weights of each group of 8 filters, as [c][i][j][g]. A last, partial
     * group is padded with zero filters: computing them costs less than a
     * loop whose length the compiler does not know (its sums would not stay
     * in registers). */
    int groups = (filters + DIRECT_GROUP - 1) / DIRECT_GROUP;
    real *weights = calloc((size_t)groups * DIRECT_GROUP * patch, sizeof(real));
    if (weights == NULL) abort();
    for (int f = 0; f < filters; f++)
        for (long q = 0; q < patch; q++)
            weights[(f / DIRECT_GROUP * patch + q) * DIRECT_GROUP + f % DIRECT_GROUP] = kernel[f * patch + q];
    const real *end = image + batches * image_size;
    /* Threads share out (image, filter group) pairs; each writes its own planes of out. */
    long units = batches * groups;
    int team = team_size((double)batches * filters * patch * oh * ow, units);
    (void)team;                                                /* used only with OpenMP */
    OMP(omp parallel for num_threads(thread_limit) if (team > 1) schedule(dynamic, 1))
    for (long unit = 0; unit < units; unit++) {
        long b = unit / groups;
        int f0 = (int)(unit % groups) * DIRECT_GROUP;
        int group = filters - f0 < DIRECT_GROUP ? filters - f0 : DIRECT_GROUP;
        const real *img = image + b * image_size;
        real *dst = out + (b * filters + f0) * (long)oh * ow;
        for (int y = 0; y < oh; y++)
            for (int x0 = 0; x0 < ow; x0 += VECTOR_WIDTH) {
                int valid = ow - x0 < VECTOR_WIDTH ? ow - x0 : VECTOR_WIDTH;
                direct_segment(img, end, weights + f0 * patch, dst + (long)y * ow + x0, (long)oh * ow,
                               channels, height, width, kh, kw, y, x0, valid, group);
            }
    }
    free(weights);
#else
    for (long b = 0; b < batches; b++)
        for (int f = 0; f < filters; f++)
            for (int y = 0; y < oh; y++)
                for (int x = 0; x < ow; x++) {
                    double total = 0.0;
                    for (int c = 0; c < channels; c++)
                        for (int i = 0; i < kh; i++)
                            for (int j = 0; j < kw; j++)
                                total += kernel[f * patch + ((long)c * kh + i) * kw + j]
                                       * image[b * image_size + ((long)c * height + y + i) * width + x + j];
                    out[((b * filters + f) * oh + y) * (long)ow + x] = (real) total;
                }
#endif
}

#undef A_AT

/*
 * Depthwise convolution: every channel is filtered by its own (kh x kw)
 * kernel, with no mixing between channels (MobileNet's building block;
 * a 1x1 convolution mixes the channels afterwards). index holds, for each
 * of the P output positions, the Q = kh * kw pixel offsets of its patch
 * inside one channel image of size channel_size.
 *     out[b, c, p]       = sum_q image[b, c, index[p, q]] * kernel[c, q]
 *     dkernel[c, q]     += sum_b,p grad[b, c, p] * image[b, c, index[p, q]]
 *     dimage[b, c, index[p, q]] += grad[b, c, p] * kernel[c, q]
 */
void depthwise_forward(const real *image, const long *index, const real *kernel, real *out,
                       long batches, long channels, long channel_size, long positions, long q_size)
{
    for (long bc = 0; bc < batches * channels; bc++) {
        const real *img = image + bc * channel_size, *k = kernel + (bc % channels) * q_size;
        real *o = out + bc * positions;
        for (long p = 0; p < positions; p++) {
            const long *cells = index + p * q_size;
            double total = 0.0;
            for (long q = 0; q < q_size; q++) total += img[cells[q]] * k[q];
            o[p] = total;
        }
    }
}

void depthwise_backward(const real *image, const long *index, const real *kernel, const real *grad,
                        real *dkernel, real *dimage, long batches, long channels, long channel_size,
                        long positions, long q_size)
{
    for (long bc = 0; bc < batches * channels; bc++) {
        const real *img = image + bc * channel_size, *k = kernel + (bc % channels) * q_size;
        const real *g = grad + bc * positions;
        real *dk = dkernel + (bc % channels) * q_size, *di = dimage + bc * channel_size;
        for (long p = 0; p < positions; p++) {
            const long *cells = index + p * q_size;
            for (long q = 0; q < q_size; q++) {
                dk[q] += g[p] * img[cells[q]];
                di[cells[q]] += g[p] * k[q];
            }
        }
    }
}


static inline double fast_exp(double x);      /* defined with softmax, below */

/*
 * erf over a block of doubles, in place. The C library's erf takes one
 * number at a time; glibc (2.35+) also ships vector versions ("libmvec"),
 * 8 or 4 numbers per call, within a few units in the last place. The build
 * (kernels_c.py) defines NANOGRAD_LIBMVEC when it can link them.
 */
#if defined(NANOGRAD_LIBMVEC) && defined(__AVX512F__)
#include <immintrin.h>
__m512d _ZGVeN8v_erf(__m512d);
static void erf_block(double *v, long n)
{
    long i = 0;
    for (; i + 8 <= n; i += 8) _mm512_storeu_pd(v + i, _ZGVeN8v_erf(_mm512_loadu_pd(v + i)));
    for (; i < n; i++) v[i] = erf(v[i]);
}
#elif defined(NANOGRAD_LIBMVEC) && defined(__AVX2__)
#include <immintrin.h>
__m256d _ZGVdN4v_erf(__m256d);
static void erf_block(double *v, long n)
{
    long i = 0;
    for (; i + 4 <= n; i += 4) _mm256_storeu_pd(v + i, _ZGVdN4v_erf(_mm256_loadu_pd(v + i)));
    for (; i < n; i++) v[i] = erf(v[i]);
}
#else
static void erf_block(double *v, long n)
{
    for (long i = 0; i < n; i++) v[i] = erf(v[i]);
}
#endif

/*
 * GELU, exact erf form:  y = x * Phi(x),  dy/dx = Phi(x) + x * phi(x)
 *     Phi(x) = (1 + erf(x / sqrt 2)) / 2,   phi(x) = e^(-x^2 / 2) / sqrt(2 pi)
 * The forward pass also stores Phi(x) ("cdf") for the backward pass. Both
 * work through a double buffer in chunks, so erf and exp run vectorized.
 */
void gelu_forward(const real *x, real *y, real *cdf, long count)
{
    double buffer[256];
    for (long start = 0; start < count; start += 256) {
        long size = count - start < 256 ? count - start : 256;
        for (long i = 0; i < size; i++) buffer[i] = (double) x[start + i] * 0.7071067811865476;
        erf_block(buffer, size);
        for (long i = 0; i < size; i++) {
            double phi = 0.5 * (1.0 + buffer[i]);
            cdf[start + i] = (real) phi;
            y[start + i] = (real) (x[start + i] * phi);
        }
    }
}

void gelu_backward(const real *x, const real *cdf, const real *grad, real *dx, long count)
{
    const double inv_sqrt_two_pi = 0.3989422804014327;  /* 1 / sqrt(2 pi) */
    double buffer[256];
    for (long start = 0; start < count; start += 256) {
        long size = count - start < 256 ? count - start : 256;
        for (long i = 0; i < size; i++) buffer[i] = -0.5 * (double) x[start + i] * x[start + i];
        for (long i = 0; i < size; i++) buffer[i] = fast_exp(buffer[i]);
        for (long i = 0; i < size; i++)
            dx[start + i] = (real) (grad[start + i] * (cdf[start + i] + x[start + i] * buffer[i] * inv_sqrt_two_pi));
    }
}

/*
 * Softmax over each contiguous row of `width` values:
 *     y_i = e^(x_i - max) / sum_j e^(x_j - max)
 *     dx_i = y_i * (dy_i - sum_j dy_j y_j)
 */
/*
 * e^x the way math libraries compute it, written so a loop of them
 * vectorizes (libm's exp is a function call, one number at a time):
 *
 *     x = k ln 2 + r,  k an integer, |r| <= ln(2) / 2
 *     e^x = 2^k * e^r
 *
 * e^r is its Taylor series up to r^13 (the next term is below 1e-17 of the
 * result), and 2^k is built directly as the bits of a double (as 2 * 2^(k-1),
 * so k = 1024 still fits). Accurate to a few units in the last place.
 * Results below e^-708 (about 1e-308) are returned as 0, above the largest
 * double as infinity.
 */
static inline double fast_exp(double x)
{
    const double log2e = 1.4426950408889634, ln2_hi = 0.6931471803691238, ln2_lo = 1.9082149292705877e-10;
    const double round_magic = 6755399441055744.0;            /* 1.5 * 2^52: adding it rounds to an integer */
    const double largest = 709.782712893384;                   /* ln(largest double) */
    double clamped = x < -708.0 ? -708.0 : (x > largest ? largest : x);
    double k = (clamped * log2e + round_magic) - round_magic;
    double r = (clamped - k * ln2_hi) - k * ln2_lo;            /* ln 2 in two parts keeps r exact */
    double p = 1.0 / 6227020800.0;                             /* 1/13!, then Horner down to 1 */
    p = p * r + 1.0 / 479001600.0;
    p = p * r + 1.0 / 39916800.0;
    p = p * r + 1.0 / 3628800.0;
    p = p * r + 1.0 / 362880.0;
    p = p * r + 1.0 / 40320.0;
    p = p * r + 1.0 / 5040.0;
    p = p * r + 1.0 / 720.0;
    p = p * r + 1.0 / 120.0;
    p = p * r + 1.0 / 24.0;
    p = p * r + 1.0 / 6.0;
    p = p * r + 0.5;
    p = p * r + 1.0;
    p = p * r + 1.0;
    long bits = ((long)k - 1 + 1023) << 52;                     /* the double 2^(k-1) */
    double half_two_k;
    memcpy(&half_two_k, &bits, sizeof half_two_k);
    double result = 2.0 * (p * half_two_k);
    result = x < -708.0 ? 0.0 : result;
    return x > largest ? HUGE_VAL : result;
}

/* y[i] = e^(x[i] - shift), returning the sum of the y[i]. The work goes
 * through a small double buffer in chunks: the loop over doubles
 * vectorizes for both precisions (GCC will not vectorize the one that mixes
 * floats into it), and the sum uses 8 running totals so it vectorizes too. */
/*
 * e^x in single precision, for the float32 build: the same reduction as
 * fast_exp, with the Taylor series stopped at r^7 (the next term is below
 * 2^-24 of the result) and twice as many numbers per vector instruction.
 */
static inline float fast_expf(float x)
{
    const float log2e = 1.44269504f, ln2_hi = 0.693145752f, ln2_lo = 1.42860677e-6f;
    const float round_magic = 12582912.0f;                     /* 1.5 * 2^23 */
    float clamped = x < -87.0f ? -87.0f : (x > 88.0f ? 88.0f : x);
    float k = (clamped * log2e + round_magic) - round_magic;
    float r = (clamped - k * ln2_hi) - k * ln2_lo;
    float p = 1.0f / 5040.0f;
    p = p * r + 1.0f / 720.0f;
    p = p * r + 1.0f / 120.0f;
    p = p * r + 1.0f / 24.0f;
    p = p * r + 1.0f / 6.0f;
    p = p * r + 0.5f;
    p = p * r + 1.0f;
    p = p * r + 1.0f;
    int bits = ((int)k - 1 + 127) << 23;                         /* the float 2^(k-1) */
    float half_two_k;
    memcpy(&half_two_k, &bits, sizeof half_two_k);
    float result = 2.0f * (p * half_two_k);
    result = x < -87.0f ? 0.0f : result;
    return x > 88.7f ? HUGE_VALF : result;
}

/* exp_shifted for floats: a float buffer and fast_expf (see below). */
static double exp_shifted_float(const float *x, double shift, float *y, long count)
{
    enum { CHUNK = 256, LANES = 16 };
    float buffer[CHUNK];
    double lanes[LANES] = {0};
    float shift_f = (float) shift;
    for (long start = 0; start < count; start += CHUNK) {
        long size = count - start < CHUNK ? count - start : CHUNK;
        for (long i = 0; i < size; i++) buffer[i] = fast_expf(x[start + i] - shift_f);
        long i = 0;
        for (; i + LANES <= size; i += LANES)
            for (int l = 0; l < LANES; l++) lanes[l] += buffer[i + l];
        for (; i < size; i++) lanes[0] += buffer[i];
        memcpy(y + start, buffer, sizeof(float) * size);
    }
    double total = 0.0;
    for (int l = 0; l < LANES; l++) total += lanes[l];
    return total;
}

static double exp_shifted(const real *x, double shift, real *y, long count)
{
    enum { CHUNK = 256, LANES = 8 };
    if (sizeof(real) == sizeof(float))                         /* the float32 build */
        return exp_shifted_float((const float *) x, shift, (float *) y, count);
    double buffer[CHUNK], lanes[LANES] = {0};
    for (long start = 0; start < count; start += CHUNK) {
        long size = count - start < CHUNK ? count - start : CHUNK;
        for (long i = 0; i < size; i++) buffer[i] = (double) x[start + i] - shift;
        for (long i = 0; i < size; i++) buffer[i] = fast_exp(buffer[i]);
        long i = 0;
        for (; i + LANES <= size; i += LANES)
            for (int l = 0; l < LANES; l++) lanes[l] += buffer[i + l];
        for (; i < size; i++) lanes[0] += buffer[i];
        for (i = 0; i < size; i++) y[start + i] = (real) buffer[i];
    }
    double total = 0.0;
    for (int l = 0; l < LANES; l++) total += lanes[l];
    return total;
}

/*
 * tanh and sigmoid from the same exp, so their loops vectorize too:
 *     tanh(a)    = 1 - 2 / (e^(2a) + 1)           for |a| >= 0.25
 *     tanh(a)    = a - a^3/3 + 2a^5/15 - ...       its Taylor series up to a^19 below
 *                                                  that (where the formula above
 *                                                  would lose digits to cancellation)
 *     sigmoid(x) = 1 / (1 + e^-x),  or  e^x / (1 + e^x) for x < 0
 * Both are within 4 units in the last place of the C library's.
 */
static inline double fast_tanh(double x)
{
    double a = fabs(x), s = a * a;
    double p = -443861162.0 / 1856156927625.0;                 /* Horner, from a^19 down */
    p = p * s + 6404582.0 / 10854718875.0;
    p = p * s - 929569.0 / 638512875.0;
    p = p * s + 21844.0 / 6081075.0;
    p = p * s - 1382.0 / 155925.0;
    p = p * s + 62.0 / 2835.0;
    p = p * s - 17.0 / 315.0;
    p = p * s + 2.0 / 15.0;
    p = p * s - 1.0 / 3.0;
    double near_zero = a + a * s * p;
    double elsewhere = 1.0 - 2.0 / (fast_exp(2.0 * a) + 1.0);
    double t = a < 0.25 ? near_zero : elsewhere;
    return x < 0.0 ? -t : t;
}

static inline double fast_sigmoid(double x)
{
    double e = fast_exp(-fabs(x));                              /* never overflows */
    double s = 1.0 / (1.0 + e);
    return x >= 0.0 ? s : e * s;
}

/* Single-precision tanh and sigmoid for the float32 build, from fast_expf;
 * tanh's series near zero stops at a^11 (enough for 24 bits below 0.25). */
static inline float fast_tanhf(float x)
{
    float a = fabsf(x), s = a * a;
    float p = 1382.0f / 155925.0f;
    p = p * s - 17.0f / 315.0f;
    p = p * s + 2.0f / 15.0f;
    p = p * s - 1.0f / 3.0f;
    float near_zero = a + a * s * p;
    float elsewhere = 1.0f - 2.0f / (fast_expf(2.0f * a) + 1.0f);
    float t = a < 0.25f ? near_zero : elsewhere;
    return x < 0.0f ? -t : t;
}

static inline float fast_sigmoidf(float x)
{
    float e = fast_expf(-fabsf(x));
    float s = 1.0f / (1.0f + e);
    return x >= 0.0f ? s : e * s;
}

/* y[i] = FN(x[i]) over floats, in place-free chunks the compiler vectorizes. */
#define MAP_FLOATS(FN, x, y, count)                                                \
    do {                                                                          \
        const float *xf_ = (const float *)(x);                                    \
        float *yf_ = (float *)(y);                                                \
        for (long i_ = 0; i_ < (count); i_++) yf_[i_] = FN(xf_[i_]);             \
    } while (0)

/* y[i] = FN(x[i]) through a double buffer in chunks (see exp_shifted). */
#define MAP_CHUNKED(FN, x, y, count)                                              \
    do {                                                                          \
        double buffer_[256];                                                      \
        for (long start_ = 0; start_ < (count); start_ += 256) {                  \
            long size_ = (count) - start_ < 256 ? (count) - start_ : 256;         \
            for (long i_ = 0; i_ < size_; i_++) buffer_[i_] = (double)(x)[start_ + i_]; \
            for (long i_ = 0; i_ < size_; i_++) buffer_[i_] = FN(buffer_[i_]);   \
            for (long i_ = 0; i_ < size_; i_++) (y)[start_ + i_] = (real)buffer_[i_]; \
        }                                                                         \
    } while (0)

void exp_vector(const real *x, real *y, long count)
{
    exp_shifted(x, 0.0, y, count);
}

/* The largest of n >= 1 values, with 16 running maxima so the loop
 * vectorizes (one running maximum is a chain the compiler leaves scalar). */
static double largest(const real *x, long n)
{
    enum { LANES = 16 };
    real part[LANES];
    for (int l = 0; l < LANES; l++) part[l] = x[0];
    long i = 0;
    for (; i + LANES <= n; i += LANES)
        for (int l = 0; l < LANES; l++) part[l] = x[i + l] > part[l] ? x[i + l] : part[l];
    for (; i < n; i++) part[0] = x[i] > part[0] ? x[i] : part[0];
    for (int l = 1; l < LANES; l++) part[0] = part[l] > part[0] ? part[l] : part[0];
    return part[0];
}

/* Threads for a kernel that works row by row: rows are shared out whole,
 * so each row's sums are done by one thread in the usual order. A row
 * element (an exp, a few multiplies) costs about as much as 64 of a matrix
 * product's vectorized multiply-adds. */
static inline int rows_team(long rows, long width)
{
    return team_size((double)rows * width * 64, rows);
}

void softmax_rows(const real *x, real *y, long rows, long width)
{
    int team = rows_team(rows, width);
    (void)team;                                                /* used only with OpenMP */
    OMP(omp parallel for num_threads(thread_limit) if (team > 1) schedule(static))
    for (long r = 0; r < rows; r++) {
        const real *in = x + r * width;
        real *out = y + r * width;
        double row_max = largest(in, width);
        double inverse = 1.0 / exp_shifted(in, row_max, out, width);
        for (long i = 0; i < width; i++) out[i] *= inverse;
    }
}

void softmax_rows_backward(const real *y, const real *grad, real *dx, long rows, long width)
{
    int team = rows_team(rows, width);
    (void)team;                                                /* used only with OpenMP */
    OMP(omp parallel for num_threads(thread_limit) if (team > 1) schedule(static))
    for (long r = 0; r < rows; r++) {
        const real *yr = y + r * width, *gr = grad + r * width;
        real *out = dx + r * width;
        double weighted = 0.0;
        for (long i = 0; i < width; i++) weighted += gr[i] * yr[i];
        for (long i = 0; i < width; i++) out[i] = yr[i] * (gr[i] - weighted);
    }
}

/*
 * Softmax cross entropy, in place, for the fused linear + loss
 * (losses.linear_cross_entropy). Each row of logits x becomes
 *
 *     factor * (softmax(x) - one_hot(target))
 *
 * the gradient of factor * (the sum of the rows' losses), and nll[r] gets
 * the row's loss, -log p_target. It does what softmax_rows, scale_vector
 * and the target fix in losses.softmax_cross_entropy do, with the same
 * operations in the same order, so the result is the same bit for bit.
 * (Only when p_target underflows to 0 does the loss come from the row's
 * logsumexp here, with fast_exp sums instead of the Python path's fsum.)
 */
void softmax_xent_rows(real *x, const long *targets, double factor, double *nll, long rows, long width)
{
    int team = rows_team(rows, width);
    (void)team;                                                /* used only with OpenMP */
    OMP(omp parallel for num_threads(thread_limit) if (team > 1) schedule(static))
    for (long r = 0; r < rows; r++) {
        real *row = x + r * width;
        long target = targets[r];
        double x_target = row[target];
        double row_max = largest(row, width);
        double total = exp_shifted(row, row_max, row, width);
        double inverse = 1.0 / total;
        for (long i = 0; i < width; i++) row[i] *= inverse;
        double p = row[target];
        nll[r] = p > 0.0 ? -log(p) : row_max + log(total) - x_target;
        for (long i = 0; i < width; i++) row[i] = row[i] * factor;
        row[target] = (real)((double)row[target] - factor);
    }
}

/* out[s] += x[o * width + s] for every row o, in order: sum_repeated's bias
 * sum without the reset, so chunks of rows can be added one after another.
 * Threads share out columns, and each adds the rows in order. */
void add_rows(const real *x, real *out, long rows, long width)
{
    enum { COLUMNS = 512 };
    long blocks = (width + COLUMNS - 1) / COLUMNS;
    int team = team_size((double)rows * width * 4, blocks);
    (void)team;                                                /* used only with OpenMP */
    OMP(omp parallel for num_threads(thread_limit) if (team > 1) schedule(static))
    for (long block = 0; block < blocks; block++) {
        long s0 = block * COLUMNS, s1 = s0 + COLUMNS < width ? s0 + COLUMNS : width;
        for (long o = 0; o < rows; o++)
            for (long s = s0; s < s1; s++) out[s] += x[o * width + s];
    }
}

/*
 * Sums over a plane with 8 independent partial sums. One running total is
 * a chain of adds, each waiting for the last; separate partials let the
 * compiler vectorize the loop (the result differs only in rounding).
 */
#define PARTIALS 8

static double plane_sum(const real *x, long n)
{
    double part[PARTIALS] = {0};
    long i = 0;
    for (; i + PARTIALS <= n; i += PARTIALS)
        for (int k = 0; k < PARTIALS; k++) part[k] += x[i + k];
    double total = 0.0;
    for (; i < n; i++) total += x[i];
    for (int k = 0; k < PARTIALS; k++) total += part[k];
    return total;
}

static double plane_squares(const real *x, long n, double m)       /* sum of (x - m)^2 */
{
    double part[PARTIALS] = {0};
    long i = 0;
    for (; i + PARTIALS <= n; i += PARTIALS)
        for (int k = 0; k < PARTIALS; k++) part[k] += (x[i + k] - m) * (x[i + k] - m);
    double total = 0.0;
    for (; i < n; i++) total += (x[i] - m) * (x[i] - m);
    for (int k = 0; k < PARTIALS; k++) total += part[k];
    return total;
}

static double plane_dot(const real *a, const real *b, long n)
{
    double part[PARTIALS] = {0};
    long i = 0;
    for (; i + PARTIALS <= n; i += PARTIALS)
        for (int k = 0; k < PARTIALS; k++) part[k] += (double) a[i + k] * b[i + k];
    double total = 0.0;
    for (; i < n; i++) total += (double) a[i] * b[i];
    for (int k = 0; k < PARTIALS; k++) total += part[k];
    return total;
}

/*
 * LayerNorm over each contiguous row of `width` values:
 *     x_hat = (x - mean) / sqrt(var + eps),   y = x_hat * weight + bias
 * Saves x_hat and inv_std = 1 / sqrt(var + eps) for the backward pass:
 *     g = dy * weight
 *     dx = inv_std * (g - mean(g) - x_hat * mean(g * x_hat))
 *     dweight += dy * x_hat,  dbias += dy        (summed over rows)
 */
void layer_norm_rows(const real *x, const real *weight, const real *bias,
                     real *y, real *x_hat, real *inv_std,
                     long rows, long width, double eps)
{
    for (long r = 0; r < rows; r++) {
        const real *in = x + r * width;
        double mean = plane_sum(in, width) / width;
        double variance = plane_squares(in, width, mean) / width;
        double scale = 1.0 / sqrt(variance + eps);
        inv_std[r] = scale;
        for (long i = 0; i < width; i++) {
            double h = (in[i] - mean) * scale;
            x_hat[r * width + i] = h;
            y[r * width + i] = h * weight[i] + bias[i];
        }
    }
}

void layer_norm_rows_backward(const real *grad, const real *x_hat, const real *inv_std,
                              const real *weight, real *dx, real *dweight, real *dbias,
                              long rows, long width, int need_dx)
{
    for (long r = 0; r < rows; r++) {
        const real *dy = grad + r * width, *h = x_hat + r * width;
        double mean_g = 0.0, mean_g_h = 0.0;
        for (long i = 0; i < width; i++) {
            dweight[i] += dy[i] * h[i];
            dbias[i] += dy[i];
            double g = dy[i] * weight[i];
            mean_g += g;
            mean_g_h += g * h[i];
        }
        if (!need_dx) continue;
        mean_g /= width;
        mean_g_h /= width;
        for (long i = 0; i < width; i++) {
            double g = dy[i] * weight[i];
            dx[r * width + i] = inv_std[r] * (g - mean_g - h[i] * mean_g_h);
        }
    }
}

/*
 * im2col / col2im with a precomputed index table: patch value i of every
 * image reads pixel index[i] (an offset inside one image).
 *     gather:  out[b, i] = image[b, index[i]]
 *     scatter: image_grad[b, index[i]] += grad[b, i]   (overlapping patches add up)
 */
void gather_indexed(const real *image, const long *index, real *out,
                    long batches, long image_size, long count)
{
    for (long b = 0; b < batches; b++) {
        const real *in = image + b * image_size;
        real *o = out + b * count;
        for (long i = 0; i < count; i++) o[i] = in[index[i]];
    }
}

/*
 * Zero padding of 2D planes, a row at a time: plane p's row r goes to row
 * r + pad_rows of the padded plane, starting pad_cols in (out starts as
 * zeros). crop_planes is its backward pass, copying the same rows back out.
 */
void pad_planes(const real *x, real *out, long planes, int rows, int cols, int pad_rows, int pad_cols)
{
    long padded_cols = cols + 2 * pad_cols, padded_size = (rows + 2L * pad_rows) * padded_cols;
    for (long p = 0; p < planes; p++)
        for (int r = 0; r < rows; r++)
            memcpy(out + p * padded_size + (r + pad_rows) * padded_cols + pad_cols,
                   x + (p * rows + r) * cols, sizeof(real) * cols);
}

void crop_planes(const real *padded, real *out, long planes, int rows, int cols, int pad_rows, int pad_cols)
{
    long padded_cols = cols + 2 * pad_cols, padded_size = (rows + 2L * pad_rows) * padded_cols;
    for (long p = 0; p < planes; p++)
        for (int r = 0; r < rows; r++)
            memcpy(out + (p * rows + r) * cols,
                   padded + p * padded_size + (r + pad_rows) * padded_cols + pad_cols, sizeof(real) * cols);
}

void scatter_add_indexed(const real *grad, const long *index, real *image_grad,
                         long batches, long image_size, long count)
{
    for (long b = 0; b < batches; b++) {
        const real *g = grad + b * count;
        real *o = image_grad + b * image_size;
        for (long i = 0; i < count; i++) o[index[i]] += g[i];
    }
}

/*
 * Sparse (rows x inner) matrix times dense (inner x n) matrix. The sparse
 * matrix is stored by rows (CSR): the nonzeros of row r are entries
 * row_start[r] .. row_start[r + 1] - 1, with columns cols[e] and values
 * values[e]. Each nonzero adds a scaled copy of one dense row:
 *     out[r, :] += values[e] * dense[cols[e], :]
 */
void csr_matmul(const long *row_start, const long *cols, const real *values,
                const real *dense, real *out, long rows, long n)
{
    for (long r = 0; r < rows; r++) {
        real *o = out + r * n;
        for (long e = row_start[r]; e < row_start[r + 1]; e++) {
            const real *d = dense + cols[e] * n;
            real v = values[e];
            for (long j = 0; j < n; j++) o[j] += v * d[j];
        }
    }
}

/*
 * Embedding lookup: whole rows of a (rows x width) table.
 *     gather:  out[i, :] = table[ids[i], :]
 *     scatter: table_grad[ids[i], :] += grad[i, :]   (repeated ids add up)
 */
void gather_rows(const real *table, const long *ids, real *out, long count, long width)
{
    for (long i = 0; i < count; i++)
        memcpy(out + i * width, table + ids[i] * width, width * sizeof(real));
}

void scatter_add_rows(const real *grad, const long *ids, real *table_grad, long count, long width)
{
    for (long i = 0; i < count; i++) {
        const real *g = grad + i * width;
        real *o = table_grad + ids[i] * width;
        for (long j = 0; j < width; j++) o[j] += g[j];
    }
}

/*
 * tanh, elementwise:  y = tanh(x),  dx = dy * (1 - y^2)
 */
void tanh_forward(const real *x, real *y, long count)
{
    if (sizeof(real) == sizeof(float)) MAP_FLOATS(fast_tanhf, x, y, count);
    else MAP_CHUNKED(fast_tanh, x, y, count);
}

void tanh_backward(const real *y, const real *grad, real *dx, long count)
{
    for (long i = 0; i < count; i++) dx[i] = grad[i] * (1.0 - y[i] * y[i]);
}

/*
 * Pooling over windows given by an index table: window w of a plane reads
 * the plane offsets index[w * window_size ... (w + 1) * window_size - 1].
 *     mean:  out[p, w] = (1 / window_size) * sum_j plane_p[index[w, j]]
 *     max:   out[p, w] = max_j plane_p[index[w, j]], and argmax[p, w] is the
 *            winning pixel's offset in the whole input (first one on ties)
 */
void window_mean(const real *image, const long *index, real *out,
                 long planes, long plane_size, long windows, long window_size)
{
    for (long p = 0; p < planes; p++) {
        const real *in = image + p * plane_size;
        for (long w = 0; w < windows; w++) {
            const long *cells = index + w * window_size;
            double total = 0.0;
            for (long j = 0; j < window_size; j++) total += in[cells[j]];
            out[p * windows + w] = total / window_size;
        }
    }
}

void window_mean_backward(const real *grad, const long *index, real *image_grad,
                          long planes, long plane_size, long windows, long window_size)
{
    for (long p = 0; p < planes; p++) {
        real *g = image_grad + p * plane_size;
        for (long w = 0; w < windows; w++) {
            const long *cells = index + w * window_size;
            double share = grad[p * windows + w] / window_size;
            for (long j = 0; j < window_size; j++) g[cells[j]] += share;
        }
    }
}

void window_max(const real *image, const long *index, real *out, long *argmax,
                long planes, long plane_size, long windows, long window_size)
{
    for (long p = 0; p < planes; p++) {
        const real *in = image + p * plane_size;
        for (long w = 0; w < windows; w++) {
            const long *cells = index + w * window_size;
            long best = cells[0];
            for (long j = 1; j < window_size; j++) if (in[cells[j]] > in[best]) best = cells[j];
            out[p * windows + w] = in[best];
            argmax[p * windows + w] = p * plane_size + best;
        }
    }
}

/*
 * Elementwise arithmetic on equal-length arrays.
 */
void add_vectors(const real *x, const real *y, real *out, long count)
{
    for (long i = 0; i < count; i++) out[i] = x[i] + y[i];
}

void multiply_vectors(const real *x, const real *y, real *out, long count)
{
    for (long i = 0; i < count; i++) out[i] = x[i] * y[i];
}

void scale_vector(const real *x, double factor, real *out, long count)
{
    for (long i = 0; i < count; i++) out[i] = x[i] * factor;
}

/*
 * Dropout mask: each entry is 0 with probability p, else 1 / (1 - p).
 * The random numbers come from splitmix64 (a tiny, well-tested generator:
 * entry i uses the i-th output after `seed`), turned into uniform doubles
 * in [0, 1) from their top 53 bits.
 */
void dropout_mask(real *mask, long count, double p, unsigned long seed)
{
    double keep = 1.0 / (1.0 - p);
    for (long i = 0; i < count; i++) {
        unsigned long z = seed + (unsigned long)(i + 1) * 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        z ^= z >> 31;
        double u = (double)(z >> 11) * 0x1.0p-53;
        mask[i] = u < p ? 0.0 : keep;
    }
}

void reciprocal_vector(const real *x, real *out, long count)
{
    for (long i = 0; i < count; i++) out[i] = 1.0 / x[i];
}

void log_vector(const real *x, real *out, long count)
{
    for (long i = 0; i < count; i++) out[i] = log(x[i]);
}

void sqrt_vector(const real *x, real *out, long count)
{
    for (long i = 0; i < count; i++) out[i] = sqrt(x[i]);
}

/* out = x + factor * y: an SGD step without a temporary for factor * y */
void add_scaled(const real *x, const real *y, double factor, real *out, long count)
{
    for (long i = 0; i < count; i++) out[i] = x[i] + factor * y[i];
}

/*
 * One step of a sign-gradient (FGSM / PGD) attack on images x in [0, 1]:
 *     d = clip(delta + alpha * sign(grad), -eps, eps),  then clip(d, -x, 1 - x)
 */
void projected_sign_step(const real *delta, const real *grad, const real *x, double alpha, double eps,
                         real *out, long count)
{
    for (long i = 0; i < count; i++) {
        double d = delta[i] + (grad[i] > 0 ? alpha : (grad[i] < 0 ? -alpha : 0.0));
        d = d < -eps ? -eps : (d > eps ? eps : d);
        d = d < -x[i] ? -x[i] : d;
        out[i] = d > 1.0 - x[i] ? 1.0 - x[i] : d;
    }
}

/* sum of x[i]^2, accumulated in double (for gradient norms) */
double sum_of_squares(const real *x, long count)
{
    double total = 0.0;
    for (long i = 0; i < count; i++) total += (double) x[i] * x[i];
    return total;
}

/*
 * One Adam step for one parameter tensor, in the same arithmetic order as
 * the Python reference in kernels.py:
 *     g     = grad * grad_scale
 *     m     = beta1 * m + (1 - beta1) * g          (updated in place)
 *     v     = beta2 * v + (1 - beta2) * g * g      (updated in place)
 *     new_w = w * decay - step_size * m / (sqrt(v) / sqrt_correction + eps)
 */
void adam_update(const real *w, const real *grad, real *m, real *v, real *new_w,
                 long count, double grad_scale, double beta1, double beta2, double decay,
                 double step_size, double sqrt_correction, double eps)
{
    for (long i = 0; i < count; i++) {
        double g = grad[i] * grad_scale;
        m[i] = beta1 * m[i] + (1.0 - beta1) * g;
        v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;
        new_w[i] = w[i] * decay - step_size * m[i] / (sqrt(v[i]) / sqrt_correction + eps);
    }
}

/*
 * ReLU, elementwise:  y = max(0, x),  dx = dy where x > 0, else 0
 */
void relu_forward(const real *x, real *y, long count)
{
    for (long i = 0; i < count; i++) y[i] = x[i] > 0.0 ? x[i] : 0.0;
}

void relu_backward(const real *x, const real *grad, real *dx, long count)
{
    for (long i = 0; i < count; i++) dx[i] = x[i] > 0.0 ? grad[i] : 0.0;
}

/*
 * The backward pass of broadcasting a block: x holds `outer` copies of
 * [each of `size` source values repeated `inner` times]; sum every copy of
 * each source value back into out[s].
 */
void sum_repeated(const real *x, real *out, long outer, long size, long inner)
{
    for (long s = 0; s < size; s++) out[s] = 0.0;
    if (inner == 1) {                  /* a bias over rows: add whole rows, which vectorizes */
        for (long o = 0; o < outer; o++)
            for (long s = 0; s < size; s++) out[s] += x[o * size + s];
        return;
    }
    for (long o = 0; o < outer; o++) {
        for (long s = 0; s < size; s++) {
            const real *run = x + (o * size + s) * inner;
            double total = 0.0;
            for (long i = 0; i < inner; i++) total += run[i];
            out[s] += total;
        }
    }
}

/*
 * RMSNorm over each contiguous row of `width` values (LLaMA's LayerNorm):
 *     r = 1 / sqrt(mean(x^2) + eps),   y = x * r * weight
 * Saves r per row. Backward, with g = dy * weight:
 *     dx = r * (g - x * r^2 * mean(g * x)),   dweight += dy * x * r
 */
void rms_norm_rows(const real *x, const real *weight, real *y, real *inv_rms,
                   long rows, long width, double eps)
{
    for (long r = 0; r < rows; r++) {
        const real *in = x + r * width;
        double squares = 0.0;
        for (long i = 0; i < width; i++) squares += in[i] * in[i];
        double scale = 1.0 / sqrt(squares / width + eps);
        inv_rms[r] = scale;
        for (long i = 0; i < width; i++) y[r * width + i] = in[i] * scale * weight[i];
    }
}

void rms_norm_rows_backward(const real *grad, const real *x, const real *inv_rms,
                            const real *weight, real *dx, real *dweight,
                            long rows, long width)
{
    for (long r = 0; r < rows; r++) {
        const real *dy = grad + r * width, *in = x + r * width;
        double scale = inv_rms[r], mean_g_x = 0.0;
        for (long i = 0; i < width; i++) {
            dweight[i] += dy[i] * in[i] * scale;
            mean_g_x += dy[i] * weight[i] * in[i];
        }
        mean_g_x /= width;
        for (long i = 0; i < width; i++)
            dx[r * width + i] = scale * (dy[i] * weight[i] - in[i] * scale * scale * mean_g_x);
    }
}

/*
 * Batch norm in training mode, for (batch, channels, plane) data: each
 * channel is normalized over every image and pixel it has:
 *     x_hat = (x - mean_c) * inv_std_c,   y = x_hat * weight_c + bias_c
 * Also returns each channel's mean and (biased) variance. The backward
 * pass is LayerNorm's, per channel instead of per row (n = batch * plane):
 *     g = dy * weight_c
 *     dx = inv_std_c * (g - mean(g) - x_hat * mean(g * x_hat))
 *     dweight_c = sum dy * x_hat,   dbias_c = sum dy
 */

void batch_norm_channels(const real *x, const real *weight, const real *bias,
                         real *y, real *x_hat, real *inv_std, real *mean, real *variance,
                         long batch, long channels, long plane, double eps)
{
    double count = (double)(batch * plane);
    for (long c = 0; c < channels; c++) {
        double total = 0.0;
        for (long b = 0; b < batch; b++) total += plane_sum(x + (b * channels + c) * plane, plane);
        double m = total / count, squares = 0.0;
        for (long b = 0; b < batch; b++) squares += plane_squares(x + (b * channels + c) * plane, plane, m);
        double v = squares / count, scale = 1.0 / sqrt(v + eps);
        mean[c] = m;
        variance[c] = v;
        inv_std[c] = scale;
        for (long b = 0; b < batch; b++) {
            long offset = (b * channels + c) * plane;
            for (long i = 0; i < plane; i++) {
                double h = (x[offset + i] - m) * scale;
                x_hat[offset + i] = h;
                y[offset + i] = h * weight[c] + bias[c];
            }
        }
    }
}

void batch_norm_channels_backward(const real *grad, const real *x_hat, const real *inv_std,
                                  const real *weight, real *dx, real *dweight, real *dbias,
                                  long batch, long channels, long plane)
{
    double count = (double)(batch * plane);
    for (long c = 0; c < channels; c++) {
        double sum_dy = 0.0, sum_dy_h = 0.0;
        for (long b = 0; b < batch; b++) {
            long offset = (b * channels + c) * plane;
            sum_dy += plane_sum(grad + offset, plane);
            sum_dy_h += plane_dot(grad + offset, x_hat + offset, plane);
        }
        dweight[c] = sum_dy_h;
        dbias[c] = sum_dy;
        double mean_g = weight[c] * sum_dy / count, mean_g_h = weight[c] * sum_dy_h / count;
        for (long b = 0; b < batch; b++) {
            long offset = (b * channels + c) * plane;
            for (long i = 0; i < plane; i++)
                dx[offset + i] = inv_std[c] * (grad[offset + i] * weight[c] - mean_g - x_hat[offset + i] * mean_g_h);
        }
    }
}

/*
 * Sigmoid, elementwise:  y = 1 / (1 + e^-x),  dx = dy * y * (1 - y)
 * Two algebraically equal forms, each safe from overflow on its own side.
 */
void sigmoid_forward(const real *x, real *y, long count)
{
    if (sizeof(real) == sizeof(float)) MAP_FLOATS(fast_sigmoidf, x, y, count);
    else MAP_CHUNKED(fast_sigmoid, x, y, count);
}

void sigmoid_backward(const real *y, const real *grad, real *dx, long count)
{
    for (long i = 0; i < count; i++) dx[i] = grad[i] * y[i] * (1.0 - y[i]);
}

/*
 * The LSTM cell (rnn.py), fused into two steps. gates is (batch x 4H) in
 * PyTorch's order i, f, g, o:
 *     memory:  c' = sigmoid(f) * c + sigmoid(i) * tanh(g)
 *     output:  h  = sigmoid(o) * tanh(c')
 * The backward passes recompute the gate activations from the gates (a few
 * exps) instead of storing them, and write the gate columns they do not
 * use as zero.
 */
static inline double gate_sigmoid(double x)        /* the fast, vectorizable forms, in the build's precision */
{
    return sizeof(real) == sizeof(float) ? fast_sigmoidf((float)x) : fast_sigmoid(x);
}

static inline double gate_tanh(double x)
{
    return sizeof(real) == sizeof(float) ? fast_tanhf((float)x) : fast_tanh(x);
}

void lstm_memory(const real *gates, const real *c, real *c_new, long batch, long hidden)
{
    for (long b = 0; b < batch; b++) {
        const real *gi = gates + b * 4 * hidden, *gf = gi + hidden, *gg = gf + hidden;
        for (long j = 0; j < hidden; j++)
            c_new[b * hidden + j] = gate_sigmoid(gf[j]) * c[b * hidden + j] + gate_sigmoid(gi[j]) * gate_tanh(gg[j]);
    }
}

void lstm_memory_backward(const real *gates, const real *c, const real *grad,
                          real *dgates, real *dc, long batch, long hidden)
{
    for (long b = 0; b < batch; b++) {
        const real *gi = gates + b * 4 * hidden, *gf = gi + hidden, *gg = gf + hidden;
        real *di = dgates + b * 4 * hidden, *df = di + hidden, *dg = df + hidden, *d_o = dg + hidden;
        for (long j = 0; j < hidden; j++) {
            double i = gate_sigmoid(gi[j]), f = gate_sigmoid(gf[j]), g = gate_tanh(gg[j]), d = grad[b * hidden + j];
            di[j] = d * g * i * (1.0 - i);
            df[j] = d * c[b * hidden + j] * f * (1.0 - f);
            dg[j] = d * i * (1.0 - g * g);
            d_o[j] = 0.0;
            dc[b * hidden + j] = d * f;
        }
    }
}

void lstm_output(const real *gates, const real *c_new, real *h, long batch, long hidden)
{
    for (long b = 0; b < batch; b++) {
        const real *go = gates + b * 4 * hidden + 3 * hidden;
        for (long j = 0; j < hidden; j++) h[b * hidden + j] = gate_sigmoid(go[j]) * gate_tanh(c_new[b * hidden + j]);
    }
}

void lstm_output_backward(const real *gates, const real *c_new, const real *grad,
                          real *dgates, real *dc_new, long batch, long hidden)
{
    for (long b = 0; b < batch; b++) {
        const real *go = gates + b * 4 * hidden + 3 * hidden;
        real *row = dgates + b * 4 * hidden;
        for (long j = 0; j < 3 * hidden; j++) row[j] = 0.0;
        for (long j = 0; j < hidden; j++) {
            double o = gate_sigmoid(go[j]), t = gate_tanh(c_new[b * hidden + j]), d = grad[b * hidden + j];
            row[3 * hidden + j] = d * t * o * (1.0 - o);
            dc_new[b * hidden + j] = d * o * (1.0 - t * t);
        }
    }
}

/*
 * log(1 + u) for u in [0, 1], written to vectorize like fast_exp:
 *
 *     log y = 2 atanh(s) = 2 (s + s^3/3 + s^5/5 + ...),   s = (y - 1) / (y + 1)
 *
 * y = 1 + u stays in [1, sqrt 2], or is halved into [sqrt 2 / 2, 1] (adding
 * ln 2), so |s| <= 0.172 and the series to s^21 reaches double precision.
 * s is computed from u directly ((y - 1) / (y + 1) = u / (2 + u), or
 * (u - 1) / (u + 3) for y / 2), so 1 + u is never rounded.
 */
static inline double fast_log1p_unit(double u)
{
    const double ln2 = 0.6931471805599453;
    int halve = u > 0.41421356237309503;                       /* sqrt 2 - 1 */
    double s = (halve ? u - 1.0 : u) / (halve ? u + 3.0 : u + 2.0);   /* one division */
    double s2 = s * s;
    double p = 1.0 / 21.0;                                     /* Horner down to 1 */
    p = p * s2 + 1.0 / 19.0;
    p = p * s2 + 1.0 / 17.0;
    p = p * s2 + 1.0 / 15.0;
    p = p * s2 + 1.0 / 13.0;
    p = p * s2 + 1.0 / 11.0;
    p = p * s2 + 1.0 / 9.0;
    p = p * s2 + 1.0 / 7.0;
    p = p * s2 + 1.0 / 5.0;
    p = p * s2 + 1.0 / 3.0;
    p = p * s2 + 1.0;
    return (halve ? ln2 : 0.0) + 2.0 * s * p;
}

/*
 * Binary cross entropy on logits, summed:
 *     sum of max(x, 0) - x * t + log(1 + e^-|x|)
 * (the stable form of -[t log p + (1 - t) log(1 - p)] with p = sigmoid(x)).
 * In chunks through a double buffer, as in exp_shifted, so the exp and log
 * vectorize, with 8 running totals.
 */
double bce_with_logits_sum(const real *x, const real *t, long count)
{
    enum { CHUNK = 256, LANES = 8 };
    double buffer[CHUNK], lanes[LANES] = {0};
    for (long start = 0; start < count; start += CHUNK) {
        long size = count - start < CHUNK ? count - start : CHUNK;
        for (long i = 0; i < size; i++) buffer[i] = -fabs((double) x[start + i]);
        for (long i = 0; i < size; i++) buffer[i] = fast_exp(buffer[i]);
        for (long i = 0; i < size; i++) buffer[i] = fast_log1p_unit(buffer[i]);
        for (long i = 0; i < size; i++) {
            double value = x[start + i];
            buffer[i] += (value > 0.0 ? value : 0.0) - value * t[start + i];
        }
        long i = 0;
        for (; i + LANES <= size; i += LANES)
            for (int l = 0; l < LANES; l++) lanes[l] += buffer[i + l];
        for (; i < size; i++) lanes[0] += buffer[i];
    }
    double total = 0.0;
    for (int l = 0; l < LANES; l++) total += lanes[l];
    return total;
}

/* Its gradient, scaled by the output's gradient g:  dx = g * (sigmoid(x) - t). */
void bce_with_logits_backward(const real *x, const real *t, double g, real *dx, long count)
{
    double buffer[256];
    for (long start = 0; start < count; start += 256) {
        long size = count - start < 256 ? count - start : 256;
        for (long i = 0; i < size; i++) buffer[i] = x[start + i];
        for (long i = 0; i < size; i++) buffer[i] = fast_sigmoid(buffer[i]);
        for (long i = 0; i < size; i++) dx[start + i] = (real) (g * (buffer[i] - t[start + i]));
    }
}
