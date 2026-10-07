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
#include <stdlib.h>
#include <string.h>

#ifndef REAL
#define REAL double
#endif
typedef REAL real;

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
 * 1. Register tile: compute an 8 x (2 vectors) block of C at a time, kept
 *    in 16 vector registers for the whole sum over p. Each step loads one
 *    row slice of B (2 vectors) and 8 numbers of A, and does 16 vector
 *    multiply-adds, instead of loading and storing rows of C every step.
 * 2. Packing: copy the B columns this tile needs into a small contiguous
 *    "panel", so the inner loop reads memory in order.
 * 3. k-blocking: do the sum over p in chunks of 256, so the panel stays in
 *    the fastest cache while every row tile of A passes over it.
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
#define TILE_ROWS 8
#define TILE_COLS (2 * VECTOR_WIDTH)
#define K_BLOCK 256
#define VEC_LOAD(v, p) memcpy(&(v), (p), sizeof(vec))
#define VEC_STORE(p, v) memcpy((p), &(v), sizeof(vec))

/* One register tile: rows (<= TILE_ROWS) rows of C, TILE_COLS columns at
 * column j, summing over p0 .. p0 + kb - 1 with B's columns packed in panel.
 * Always inlined, so the full-tile call (rows = TILE_ROWS) gets a constant
 * row count the compiler can unroll. */
static inline __attribute__((always_inline)) void tile(
    const real *A, real *C, int m, int k, int n, int trans_a,
    const real *panel, int i, int j, int p0, int kb, int rows)
{
    vec acc[TILE_ROWS][2];
    memset(acc, 0, sizeof acc);
    for (int p = 0; p < kb; p++) {
        vec b0, b1;
        VEC_LOAD(b0, panel + (long)p * TILE_COLS);
        VEC_LOAD(b1, panel + (long)p * TILE_COLS + VECTOR_WIDTH);
        for (int r = 0; r < rows; r++) {
            real a_rp = A_AT(i + r, p0 + p);
            acc[r][0] += a_rp * b0;
            acc[r][1] += a_rp * b1;
        }
    }
    for (int r = 0; r < rows; r++) {
        real *c_row = C + (long)(i + r) * n + j;
        vec c0, c1;
        VEC_LOAD(c0, c_row);
        VEC_LOAD(c1, c_row + VECTOR_WIDTH);
        c0 += acc[r][0];
        c1 += acc[r][1];
        VEC_STORE(c_row, c0);
        VEC_STORE(c_row + VECTOR_WIDTH, c1);
    }
}

/* C (m x n) += op(A) (m x k) @ B (k x n, row-major). panel: K_BLOCK * TILE_COLS doubles. */
static void gemm_tiled(const real *A, const real *B, real *C,
                       int m, int k, int n, int trans_a, real *panel)
{
    int full_cols = n - n % TILE_COLS;
    for (int j = 0; j < full_cols; j += TILE_COLS) {
        for (int p0 = 0; p0 < k; p0 += K_BLOCK) {
            int kb = k - p0 < K_BLOCK ? k - p0 : K_BLOCK;
            for (int p = 0; p < kb; p++)                       /* 2. pack */
                memcpy(panel + (long)p * TILE_COLS, B + (long)(p0 + p) * n + j, sizeof(real) * TILE_COLS);

            int i = 0;
            for (; i + TILE_ROWS <= m; i += TILE_ROWS)          /* 1. register tiles */
                tile(A, C, m, k, n, trans_a, panel, i, j, p0, kb, TILE_ROWS);
            if (i < m)                                           /* leftover rows */
                tile(A, C, m, k, n, trans_a, panel, i, j, p0, kb, m - i);
        }
    }
    if (full_cols < n) {                                       /* leftover columns */
        for (int i = 0; i < m; i++) {
            real *c_row = C + (long)i * n;
            for (int p = 0; p < k; p++) {
                real a_ip = A_AT(i, p);
                const real *b_row = B + (long)p * n;
                for (int jj = full_cols; jj < n; jj++) c_row[jj] += a_ip * b_row[jj];
            }
        }
    }
}
#endif

void gemm(
    const real *a, const real *b, real *c,
    int batches, int m, int k, int n,
    long a_step, long b_step, long c_step,
    int trans_a, int trans_b)
{
    /* For op(B) = B^T, copy B^T into a scratch (k x n) buffer once per batch
     * entry, so the inner loop below always walks contiguous rows of op(B).
     * (Summing dot products instead would be a reduction, which compilers
     * will not vectorize without permission to reorder additions.) */
    real *scratch = trans_b ? malloc(sizeof(real) * (size_t)k * n) : NULL;
    if (trans_b && scratch == NULL) {
        abort();  /* out of memory: fail loudly rather than return zeros */
    }
#ifdef VECTOR_WIDTH
    real *panel = malloc(sizeof(real) * K_BLOCK * TILE_COLS);
    if (panel == NULL) {
        abort();
    }
#endif

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

#ifdef VECTOR_WIDTH
        if (n >= TILE_COLS) {
            gemm_tiled(A, B, C, m, k, n, trans_a, panel);
            continue;
        }
#endif

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
#ifdef VECTOR_WIDTH
    free(panel);
#endif
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
    real *panel = malloc(sizeof(real) * K_BLOCK * TILE_COLS);
    if (panel == NULL) {
        abort();
    }
    int full_cols = n >= TILE_COLS ? n - n % TILE_COLS : 0;
#else
    int full_cols = 0;
#endif
    for (long batch = 0; batch < batches; batch++) {
        const real *img = image + batch * image_size;
        real *C = out + batch * (long)m * n;
#ifdef VECTOR_WIDTH
        for (int j = 0; j < full_cols; j += TILE_COLS) {
            for (int p0 = 0; p0 < k; p0 += K_BLOCK) {
                int kb = k - p0 < K_BLOCK ? k - p0 : K_BLOCK;
                for (int p = 0; p < kb; p++)                   /* pack through the index */
                    for (int jj = 0; jj < TILE_COLS; jj++)
                        panel[(long)p * TILE_COLS + jj] = img[index[(long)(j + jj) * k + p0 + p]];
                int i = 0;
                for (; i + TILE_ROWS <= m; i += TILE_ROWS)
                    tile(A, C, m, k, n, 0, panel, i, j, p0, kb, TILE_ROWS);
                if (i < m)
                    tile(A, C, m, k, n, 0, panel, i, j, p0, kb, m - i);
            }
        }
#endif
        for (int i = 0; i < m; i++) {                          /* leftover (or all) columns */
            for (int j = full_cols; j < n; j++) {
                double total = 0.0;
                const long *cells = index + (long)j * k;
                for (int q = 0; q < k; q++) total += A_AT(i, q) * img[cells[q]];
                C[(long)i * n + j] += total;
            }
        }
    }
#ifdef VECTOR_WIDTH
    free(panel);
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
    real *panel = malloc(sizeof(real) * K_BLOCK * TILE_COLS);
    if (panel == NULL) {
        abort();
    }
    int full_cols = n >= TILE_COLS ? n - n % TILE_COLS : 0;
#else
    int full_cols = 0;
#endif
    for (long batch = 0; batch < batches; batch++) {
        const real *img = image + batch * image_size;
        const real *A = grad + batch * (long)m * k;
#ifdef VECTOR_WIDTH
        for (int j = 0; j < full_cols; j += TILE_COLS) {
            for (int p0 = 0; p0 < k; p0 += K_BLOCK) {
                int kb = k - p0 < K_BLOCK ? k - p0 : K_BLOCK;
                for (int p = 0; p < kb; p++) {                 /* pack patch rows */
                    const long *cells = index + (long)(p0 + p) * n + j;
                    for (int jj = 0; jj < TILE_COLS; jj++)
                        panel[(long)p * TILE_COLS + jj] = img[cells[jj]];
                }
                int i = 0;
                for (; i + TILE_ROWS <= m; i += TILE_ROWS)
                    tile(A, dkernel, m, k, n, 0, panel, i, j, p0, kb, TILE_ROWS);
                if (i < m)
                    tile(A, dkernel, m, k, n, 0, panel, i, j, p0, kb, m - i);
            }
        }
#endif
        for (int i = 0; i < m; i++) {                          /* leftover (or all) columns */
            for (int q = full_cols; q < n; q++) {
                double total = 0.0;
                for (int p = 0; p < k; p++) total += A_AT(i, p) * img[index[(long)p * n + q]];
                dkernel[(long)i * n + q] += total;
            }
        }
    }
#ifdef VECTOR_WIDTH
    free(panel);
#endif
}

#undef A_AT


/*
 * GELU, exact erf form:  y = x * Phi(x),  dy/dx = Phi(x) + x * phi(x)
 * The forward pass also stores Phi(x) ("cdf") for the backward pass.
 */
void gelu_forward(const real *x, real *y, real *cdf, long count)
{
    for (long i = 0; i < count; i++) {
        cdf[i] = 0.5 * (1.0 + erf(x[i] / sqrt(2.0)));
        y[i] = x[i] * cdf[i];
    }
}

void gelu_backward(const real *x, const real *cdf, const real *grad, real *dx, long count)
{
    const double inv_sqrt_two_pi = 0.3989422804014327;  /* 1 / sqrt(2 pi) */
    for (long i = 0; i < count; i++) {
        double pdf = exp(-0.5 * x[i] * x[i]) * inv_sqrt_two_pi;
        dx[i] = grad[i] * (cdf[i] + x[i] * pdf);
    }
}

/*
 * Softmax over each contiguous row of `width` values:
 *     y_i = e^(x_i - max) / sum_j e^(x_j - max)
 *     dx_i = y_i * (dy_i - sum_j dy_j y_j)
 */
void softmax_rows(const real *x, real *y, long rows, long width)
{
    for (long r = 0; r < rows; r++) {
        const real *in = x + r * width;
        real *out = y + r * width;
        double row_max = in[0];
        for (long i = 1; i < width; i++) if (in[i] > row_max) row_max = in[i];
        double total = 0.0;
        for (long i = 0; i < width; i++) { out[i] = exp(in[i] - row_max); total += out[i]; }
        for (long i = 0; i < width; i++) out[i] /= total;
    }
}

void softmax_rows_backward(const real *y, const real *grad, real *dx, long rows, long width)
{
    for (long r = 0; r < rows; r++) {
        const real *yr = y + r * width, *gr = grad + r * width;
        real *out = dx + r * width;
        double weighted = 0.0;
        for (long i = 0; i < width; i++) weighted += gr[i] * yr[i];
        for (long i = 0; i < width; i++) out[i] = yr[i] * (gr[i] - weighted);
    }
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
        double mean = 0.0, variance = 0.0;
        for (long i = 0; i < width; i++) mean += in[i];
        mean /= width;
        for (long i = 0; i < width; i++) variance += (in[i] - mean) * (in[i] - mean);
        variance /= width;
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
 * tanh, elementwise:  y = tanh(x),  dx = dy * (1 - y^2)
 */
void tanh_forward(const real *x, real *y, long count)
{
    for (long i = 0; i < count; i++) y[i] = tanh(x[i]);
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
        for (long b = 0; b < batch; b++) {
            const real *in = x + (b * channels + c) * plane;
            for (long i = 0; i < plane; i++) total += in[i];
        }
        double m = total / count, squares = 0.0;
        for (long b = 0; b < batch; b++) {
            const real *in = x + (b * channels + c) * plane;
            for (long i = 0; i < plane; i++) squares += (in[i] - m) * (in[i] - m);
        }
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
            for (long i = 0; i < plane; i++) {
                sum_dy += grad[offset + i];
                sum_dy_h += grad[offset + i] * x_hat[offset + i];
            }
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
    for (long i = 0; i < count; i++) {
        if (x[i] >= 0.0) {
            y[i] = 1.0 / (1.0 + exp(-x[i]));
        } else {
            double e = exp(x[i]);
            y[i] = e / (1.0 + e);
        }
    }
}

void sigmoid_backward(const real *y, const real *grad, real *dx, long count)
{
    for (long i = 0; i < count; i++) dx[i] = grad[i] * y[i] * (1.0 - y[i]);
}

/*
 * Binary cross entropy on logits, summed:
 *     sum of max(x, 0) - x * t + log(1 + e^-|x|)
 * (the stable form of -[t log p + (1 - t) log(1 - p)] with p = sigmoid(x)).
 */
double bce_with_logits_sum(const real *x, const real *t, long count)
{
    double total = 0.0;
    for (long i = 0; i < count; i++) {
        double value = x[i];
        total += (value > 0.0 ? value : 0.0) - value * t[i] + log1p(exp(-fabs(value)));
    }
    return total;
}
