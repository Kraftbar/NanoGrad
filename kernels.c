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
#include <stdlib.h>

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
