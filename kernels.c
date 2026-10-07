/*
 * NanoGrad's optional compiled kernels.
 *
 * kernels.py compiles this file with the system C compiler the first time it
 * is needed and calls it through ctypes. If that is not possible, kernels.py
 * runs the pure-Python versions instead, which compute the same thing and
 * are the readable reference. Tests check that the two agree.
 *
 * Matrices are flat row-major arrays of doubles, exactly like Tensor.data.
 */

#include <math.h>
#include <stdlib.h>

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
void gemm(
    const double *a, const double *b, double *c,
    int batches, int m, int k, int n,
    long a_step, long b_step, long c_step,
    int trans_a, int trans_b)
{
    /* For op(B) = B^T, copy B^T into a scratch (k x n) buffer once per batch
     * entry, so the inner loop below always walks contiguous rows of op(B).
     * (Summing dot products instead would be a reduction, which compilers
     * will not vectorize without permission to reorder additions.) */
    double *scratch = trans_b ? malloc(sizeof(double) * (size_t)k * n) : NULL;
    if (trans_b && scratch == NULL) {
        abort();  /* out of memory: fail loudly rather than return zeros */
    }

    for (int batch = 0; batch < batches; batch++) {
        const double *A = a + batch * a_step;
        const double *B = b + batch * b_step;
        double *C = c + batch * c_step;

        if (trans_b) {
            for (int j = 0; j < n; j++)
                for (int p = 0; p < k; p++)
                    scratch[(long)p * n + j] = B[(long)j * k + p];
            B = scratch;
        }

        /* Row i of C is a weighted sum of rows of op(B):
         * c_row += A'(i, p) * op(B)[p, :] for each p. The inner loop runs
         * along contiguous memory, which compilers vectorize well. */
        for (int i = 0; i < m; i++) {
            double *c_row = C + (long)i * n;
            for (int p = 0; p < k; p++) {
                double a_ip = trans_a ? A[(long)p * m + i] : A[(long)i * k + p];
                const double *b_row = B + (long)p * n;
                for (int j = 0; j < n; j++) {
                    c_row[j] += a_ip * b_row[j];
                }
            }
        }
    }
    free(scratch);
}


/*
 * GELU, exact erf form:  y = x * Phi(x),  dy/dx = Phi(x) + x * phi(x)
 * The forward pass also stores Phi(x) ("cdf") for the backward pass.
 */
void gelu_forward(const double *x, double *y, double *cdf, long count)
{
    for (long i = 0; i < count; i++) {
        cdf[i] = 0.5 * (1.0 + erf(x[i] / sqrt(2.0)));
        y[i] = x[i] * cdf[i];
    }
}

void gelu_backward(const double *x, const double *cdf, const double *grad, double *dx, long count)
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
void softmax_rows(const double *x, double *y, long rows, long width)
{
    for (long r = 0; r < rows; r++) {
        const double *in = x + r * width;
        double *out = y + r * width;
        double row_max = in[0];
        for (long i = 1; i < width; i++) if (in[i] > row_max) row_max = in[i];
        double total = 0.0;
        for (long i = 0; i < width; i++) { out[i] = exp(in[i] - row_max); total += out[i]; }
        for (long i = 0; i < width; i++) out[i] /= total;
    }
}

void softmax_rows_backward(const double *y, const double *grad, double *dx, long rows, long width)
{
    for (long r = 0; r < rows; r++) {
        const double *yr = y + r * width, *gr = grad + r * width;
        double *out = dx + r * width;
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
void layer_norm_rows(const double *x, const double *weight, const double *bias,
                     double *y, double *x_hat, double *inv_std,
                     long rows, long width, double eps)
{
    for (long r = 0; r < rows; r++) {
        const double *in = x + r * width;
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

void layer_norm_rows_backward(const double *grad, const double *x_hat, const double *inv_std,
                              const double *weight, double *dx, double *dweight, double *dbias,
                              long rows, long width, int need_dx)
{
    for (long r = 0; r < rows; r++) {
        const double *dy = grad + r * width, *h = x_hat + r * width;
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
