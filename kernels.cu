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
