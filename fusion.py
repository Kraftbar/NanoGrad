"""A small op compiler for the GPU: chains of elementwise kernels fused into one.

``relu(x * w + b)`` runs three kernels on the GPU: each reads its inputs
from GPU memory and writes its result back, and the next one reads that
result again. Elementwise kernels do almost no arithmetic per value, so
their time is that memory traffic. A *fused* kernel does the whole chain
per value in registers: one thread reads x, w and b once, computes
``relu(x * w + b)``, and writes only what is needed later. This is what
``torch.compile`` and XLA do for elementwise code, in miniature.

``capture(step, *inputs)`` works like ``kernels_cuda.capture`` (a CUDA
graph of one training step), with one compiler pass in between:

1. **Trace.** The step runs once, as when it is recorded, but every
   kernel launch, copy and clear is noted down (the kernel's name and its
   arguments) instead of being sent to the GPU. Python decides everything
   on the host, so the list is the whole step, in order.
2. **Group.** Neighbouring launches of the kernels in ``FUSIBLE`` (one
   thread per value: adds, products, activations, their gradients, bias
   broadcasts, copies, Adam's update, ...) that cover the same number of
   values form a group. Thread i of a fused kernel computes value i of
   every kernel in the group, in order, so each kernel sees exactly what
   it saw before. That holds as long as a kernel reads other values than
   i (a bias at i % n, a padded image) only from arrays the group does
   not write; otherwise the group ends there.
3. **Drop stores.** A value written inside a group that no later kernel
   reads, and that no Python object still holds when the step ends (a
   temporary), stays in a register: it is never written to memory.
4. **Batch.** Neighbouring groups that compute the same thing on
   independent arrays (Adam's update of each parameter, one after the
   other) become one launch of one kernel, which reads each group's
   arrays from a table (PyTorch's "multi-tensor apply").
5. **Generate.** Each group becomes CUDA source: the kernels' arithmetic,
   taken from ``FUSIBLE``, one call after another on registers. Pointers
   and settings come from a table in GPU memory, not from the source, so
   groups that do the same thing share one kernel, and the source does
   not change from run to run: NVRTC compiles it once and the binary is
   cached on disk, as ``kernels.cu`` is.
6. **Record.** The new list (fused kernels and the remaining originals)
   is sent to the GPU while the CUDA graph records, and replays as usual.

Why only inside a CUDA graph? The trace and the compile cost far more
than a step; a graph pays them once and replays the result thousands of
times. Why neighbouring kernels only? Next to each other, in one thread
per value, they need no proof that reordering them is safe. The one
reordering done (a small piece in the way of a group, such as a bias's
copy, runs first) needs only this one: the two share no array. Arrays
share memory only if they have the same pointer (see Buffer), so sets
of pointers decide it.

Bit for bit. A GPU compiler contracts ``a * b + c`` into one fused
multiply-add, rounded once instead of twice. Inside one kernel that is
how ``kernels.cu`` was compiled too, but across what used to be two
kernels it would change the result: the product used to be rounded when
it was stored. So every value handed from one kernel to the next goes
through ``settle`` (adding -0, an explicitly rounded add that the
compiler may not contract), and the fused step gives the same bits as
the unfused one (``test_kernels_cuda.py`` checks every template and
whole training steps). Inside one kernel, too, the compiler is free to
contract or not, and may decide differently in a fused kernel: Adam's
``w * decay - step`` did, so kernels.cu now rounds that product
explicitly, and the template does the same.
"""

from __future__ import annotations

import ctypes
import gc
import hashlib
import os
import struct
import weakref
from collections.abc import Callable
from pathlib import Path

import kernels_cuda
from kernels_cuda import BLOCK, DeviceArray, blocks


# ---------------------------------------------------------------------------
# The kernels that can be fused, as per-value templates
# ---------------------------------------------------------------------------

# Each entry: the kernel's arguments in kernels.cu's order, and its
# arithmetic on value i (copied from kernels.cu, so that the results are
# the same). Arguments:
#     x        an array, read at i             >out   an array written at i
#     ~m       read and written at i (in place)
#     @x       an array the body indexes itself (a bias at i % n): a pointer,
#              of real numbers, or of doubles (@x:d) or longs (@x:l)
#     name:d   a double setting                 name:l   a long setting
# ``size`` is how many values the launch covers (one thread each), from the
# settings.


class Template:
    def __init__(self, arguments: str, body: str, size: str = "count", when: str = "True") -> None:
        self.arguments = [_parse_argument(word) for word in arguments.split()]
        self.body = body
        self.size = compile(size, "<size>", "eval")
        self.when = compile(when, "<when>", "eval")      # which launches of the kernel this covers

    def function(self, name: str) -> str:
        """The template as a CUDA function: value i's arithmetic."""

        types = {"in": "real {}", "out": "real &{}", "inout": "real &{}", "setting:d": "double {}",
                 "setting:l": "long {}", "pointer": "const real *{}", "pointer:d": "const double *{}",
                 "pointer:l": "const long *{}"}
        parameters = ", ".join(["long i"] + [types[kind].format(word) for kind, word in self.arguments])
        body = "\n".join("    " + line for line in self.body.strip().splitlines())
        return f"__device__ __forceinline__ void {name}({parameters})\n{{\n{body}\n}}\n"


def _parse_argument(word: str) -> tuple[str, str]:
    if word.startswith(">"):
        return "out", word[1:]
    if word.startswith("~"):
        return "inout", word[1:]
    name, _, kind = word.lstrip("@").partition(":")
    if word.startswith("@"):
        return "pointer" + (":" + kind if kind else ""), name
    return ("setting:" + kind if kind else "in"), name


FUSIBLE = {
    "add_vectors": Template("x y >out count:l", "out = x + y;"),
    "multiply_vectors": Template("x y >out count:l", "out = x * y;"),
    "scale_vector": Template("x factor:d >out count:l", "out = (real)(x * factor);"),
    "add_scaled": Template("x y factor:d >out count:l", "out = (real)(x + factor * y);"),
    "reciprocal_vector": Template("x >out count:l", "out = (real)1 / x;"),
    "exp_vector": Template("x >out count:l", "out = exp(x);"),
    "log_vector": Template("x >out count:l", "out = log(x);"),
    "sqrt_vector": Template("x >out count:l", "out = sqrt(x);"),
    "relu_forward": Template("x >out count:l", "out = x > 0 ? x : (real)0;"),
    "relu_backward": Template("x grad >dx count:l", "dx = x > 0 ? grad : (real)0;"),
    "tanh_forward": Template("x >out count:l", "out = tanh(x);"),
    "tanh_backward": Template("y grad >dx count:l", "dx = grad * ((real)1 - y * y);"),
    "sigmoid_forward": Template("x >out count:l", "out = (real)1 / ((real)1 + exp(-x));"),
    "sigmoid_backward": Template("y grad >dx count:l", "dx = grad * y * ((real)1 - y);"),
    "gelu_forward": Template("x >y >cdf count:l", """
        real phi = (real)0.5 * ((real)1 + erf(x * (real)0.7071067811865476));
        cdf = phi;
        y = x * phi;"""),
    "gelu_backward": Template("x cdf grad >dx count:l", """
        real density = exp((real)-0.5 * x * x) * (real)0.3989422804014327;
        dx = grad * (cdf + x * density);"""),
    "bce_with_logits_backward": Template("x t g:d >dx count:l", """
        real e = exp(-fabs(x)), s = (real)1 / ((real)1 + e);
        real sigmoid = x >= 0 ? s : e * s;
        dx = (real)(g * (sigmoid - t));"""),
    "projected_sign_step": Template("delta grad x alpha:d eps:d >out count:l", """
        real a = (real)alpha, e = (real)eps, low = -x, high = (real)1 - x;
        real d = delta + (grad > 0 ? a : (grad < 0 ? -a : (real)0));
        d = d < -e ? -e : (d > e ? e : d);
        d = d < low ? low : d;
        out = d > high ? high : d;"""),
    "dropout_mask": Template(">mask count:l p:d seed:l", """
        unsigned long long z = (unsigned long long)seed + (unsigned long long)(i + 1) * 0x9E3779B97F4A7C15ULL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z ^= z >> 31;
        double u = (double)(z >> 11) * 0x1.0p-53;
        mask = u < p ? (real)0 : (real)(1.0 / (1.0 - p));"""),
    "adam_update": Template("w grad ~m ~v >new_w count:l grad_scale:d beta1:d beta2:d decay:d step_size:d "
                            "sqrt_correction:d eps:d",
                            "new_w = adam_value(w, grad, m, v, grad_scale, beta1, beta2, decay, step_size, "
                            "sqrt_correction, eps);"),
    "adam_update_graphed": Template("~w grad ~m ~v count:l @settings:d @grad_scale:d use_decay:l beta1:d beta2:d eps:d",
                                    "w = adam_value(w, grad, m, v, grad_scale ? grad_scale[0] : 1.0, beta1, beta2,\n"
                                    "               use_decay ? settings[2] : 1.0, settings[0], settings[1], eps);"),
    "scale_by_first": Template("x @s >out count:l", "out = x * s[0];"),
    # Broadcasts and index maps: value i reads another index of an array.
    "repeat_values": Template("@x count:l >out total:l", "out = x[i % count];", size="total"),
    "expand_repeated": Template("@x >out size:l inner:l total:l", "out = x[(i / inner) % size];", size="total"),
    "gather_indexed": Template("@image @index:l >out batches:l image_size:l count:l",
                               "out = image[i / count * image_size + index[i % count]];", size="batches * count"),
    "scatter_add_indexed": Template("@grad @starts:l @sources:l ~image_grad batches:l image_size:l count:l", """
        const real *g = grad + i / image_size * count;
        long t = i % image_size;
        for (long e = starts[t]; e < starts[t + 1]; e++) image_grad += g[sources[e]];""",
                                    size="batches * image_size"),
    "pad_planes": Template("@x >out planes:l rows:l cols:l pad_rows:l pad_cols:l", """
        long padded_rows = rows + 2 * pad_rows, padded_cols = cols + 2 * pad_cols;
        long c = i % padded_cols - pad_cols, r = i / padded_cols % padded_rows - pad_rows;
        long p = i / (padded_cols * padded_rows);
        out = r >= 0 && r < rows && c >= 0 && c < cols ? x[(p * rows + r) * cols + c] : (real)0;""",
                           size="planes * (rows + 2 * pad_rows) * (cols + 2 * pad_cols)"),
    "crop_planes": Template("@padded >out planes:l rows:l cols:l pad_rows:l pad_cols:l", """
        long padded_rows = rows + 2 * pad_rows, padded_cols = cols + 2 * pad_cols;
        long c = i % cols, r = i / cols % rows, p = i / (cols * rows);
        out = padded[(p * padded_rows + r + pad_rows) * padded_cols + c + pad_cols];""",
                            size="planes * rows * cols"),
    # Not kernels: kernels_cuda.zeros (a memset), and a copy of a whole array
    # (a reshape's), which a kernel does as well as a memcpy and can then skip.
    "zeros": Template(">out count:l", "out = (real)0;"),
    "copy": Template("x start:l >out offset:l count:l", "out = x;", when="start == 0 and offset == 0"),
}

# Shared by the templates, and the boundary between two fused kernels.
PRELUDE = r"""
typedef REAL real;
#define KERNEL extern "C" __global__ void

/* A value handed from one kernel to the next, rounded as storing it did:
 * adding -0 changes nothing, and an explicitly rounded add (__fadd_rn) is
 * never contracted with a product before it into one fused multiply-add. */
__device__ __forceinline__ float settle(float x) { return __fadd_rn(x, -0.0f); }
__device__ __forceinline__ double settle(double x) { return __dadd_rn(x, -0.0); }

/* As in kernels.cu: a * b, never contracted with the subtraction after it. */
__device__ __forceinline__ float rounded_product(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ double rounded_product(double a, double b) { return __dmul_rn(a, b); }

/* kernels.cu's adam_step, on values: m and v are updated, the new weight returned. */
__device__ __forceinline__ real adam_value(real w, real grad, real &m, real &v, double grad_scale, double beta1,
                                           double beta2, double decay, double step_size, double sqrt_correction,
                                           double eps)
{
    real g = grad * (real)grad_scale;
    real m_i = (real)beta1 * m + (real)(1.0 - beta1) * g;
    real v_i = (real)beta2 * v + (real)(1.0 - beta2) * g * g;
    m = m_i;
    v = v_i;
    return rounded_product(w, (real)decay) - (real)step_size * m_i / (sqrt(v_i) / (real)sqrt_correction + (real)eps);
}
"""


# ---------------------------------------------------------------------------
# Tracing: the step's stream work, noted down instead of sent
# ---------------------------------------------------------------------------


class Buffer:
    """One DeviceArray the traced step used. Holds it only weakly: the step
    frees its temporaries as it goes, and their blocks are reused, so two
    Buffers can have one pointer (one after the other). Two arrays never
    share part of a block (a DeviceArray has a block of its own, and slices
    are copies), so arrays that meet have the same pointer."""

    __slots__ = ("serial", "pointer", "typecode", "_ref")

    def __init__(self, x: DeviceArray, serial: int) -> None:
        self.serial, self.pointer, self.typecode = serial, x.pointer, x.typecode
        self._ref = weakref.ref(x)

    def alive(self) -> bool:
        """Whether something still holds the array (once the step is done: its results, the weights)."""

        return self._ref() is not None


class Op:
    """A launch (``name`` a kernel), a copy or a clear, with Buffers for its arrays."""

    __slots__ = ("kind", "name", "code", "grid", "block", "args", "shared")

    def __init__(self, kind: str, name: str, code: str, args: tuple, grid=None, block=None, shared: int = 0) -> None:
        self.kind, self.name, self.code, self.args = kind, name, code, args
        self.grid, self.block, self.shared = grid, block, shared

    def buffers(self) -> list[Buffer]:
        return [a for a in self.args if type(a) is Buffer]


class Tracer:
    """Set as the recording graph's ``tracer``: kernels_cuda hands it every
    launch, copy and clear of the step (see kernels_cuda.launch)."""

    def __init__(self) -> None:
        self.ops: list[Op] = []
        self._buffers: dict[int, Buffer] = {}
        self._serials = 0

    def buffer(self, x: DeviceArray) -> Buffer:
        known = self._buffers.get(id(x))
        if known is None or known._ref() is not x:          # (an id is reused once its array is gone)
            known = self._buffers[id(x)] = Buffer(x, self._serials)
            self._serials += 1
        return known

    def _wrap(self, args) -> tuple:
        return tuple(self.buffer(a) if type(a) is DeviceArray else a for a in args)

    def launch(self, name, code, grid, block, args, shared) -> None:
        self.ops.append(Op("launch", name, code, self._wrap(args), grid, block, shared))

    def copy(self, source, start, target, offset, count) -> None:
        self.ops.append(Op("copy", "copy", source.typecode, self._wrap((source, start, target, offset, count))))

    def clear(self, out) -> None:
        self.ops.append(Op("clear", "zeros", out.typecode, self._wrap((out, out.count))))


# ---------------------------------------------------------------------------
# Grouping: which neighbouring kernels become one
# ---------------------------------------------------------------------------


class Use:
    """One fusible op: its template, how many values it covers, and its arguments by role."""

    def __init__(self, index: int, op: Op, template: Template, size: int) -> None:
        self.index, self.op, self.template, self.size = index, op, template, size
        self.bound = list(zip(template.arguments, op.args))       # ((kind, name), value)
        self.writes = [value for (kind, _), value in self.bound if kind in ("out", "inout")]
        self.indexed = {value.pointer for (kind, _), value in self.bound
                        if kind.startswith("pointer") and type(value) is Buffer}


def _use(index: int, op: Op) -> Use | None:
    """The op as a fusible Use, or None (not in FUSIBLE, or launched unusually)."""

    template = FUSIBLE.get(op.name)
    if template is None or op.code not in ("f", "d") or len(op.args) != len(template.arguments):
        return None
    settings = {}
    for (kind, name), value in zip(template.arguments, op.args):
        if kind in ("in", "out", "inout"):
            if type(value) is not Buffer or value.typecode != op.code:
                return None
        elif kind.startswith("pointer"):
            if type(value) is not Buffer and value != 0:
                return None
        else:
            if type(value) not in (int, float):
                return None
            settings[name] = value
    size = eval(template.size, {}, settings)
    if size <= 0 or not eval(template.when, {}, settings):
        return None
    if op.kind == "launch" and (op.grid != blocks(size) or op.block != BLOCK or op.shared):
        return None
    return Use(index, op, template, size)


class Group:
    """Neighbouring fusible ops that one kernel runs, thread i doing value i of each."""

    def __init__(self, first: Use) -> None:
        self.uses: list[Use] = []
        self.size, self.code = first.size, first.op.code
        self.written: set[int] = set()                     # pointers: of the arrays the group writes,
        self.indexed: set[int] = set()                     # of those it reads at other indexes than i,
        self.used: set[int] = set()                        # and of all it touches
        self.add(first)

    def add(self, use: Use) -> None:
        self.uses.append(use)
        self.written |= {b.pointer for b in use.writes}
        self.indexed |= use.indexed
        self.used |= {b.pointer for b in use.op.buffers()}

    def accepts(self, use: Use) -> bool:
        # Value i of another index of an array (@ arguments) must come from
        # before the group: no thread of the group may write that array.
        return (use.size == self.size and use.op.code == self.code
                and not use.indexed & self.written and not {b.pointer for b in use.writes} & self.indexed)

    def independent_of(self, other: Group | Batch) -> bool:
        """Whether running the two at once is safe: neither writes what the other uses."""

        return not self.written & other.used and not other.written & self.used


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------


class Kernel:
    """The CUDA source of one group, and its row of the table: what the
    source reads from the table (pointers, settings), in order. Groups that
    do the same thing get the same source, and different rows."""

    def __init__(self, group: Group, stored: set[int]) -> None:
        self.code = group.code
        self.size = group.size
        self.names = [u.op.name for u in group.uses]
        self.fields: list[int] = []                        # the row after (count, first block)
        lines, value_of, count = [], {}, 0                 # value_of: Buffer serial -> its register

        def field(value) -> str:
            if type(value) is Buffer:
                value = value.pointer
            elif type(value) is float:                     # a double, as its 64 bits
                value = struct.unpack("q", struct.pack("d", value))[0]
            self.fields.append(value)
            return f"row[{len(self.fields) + 1}]"

        def register() -> str:
            nonlocal count
            count += 1
            return f"v{count - 1}"

        def read(buffer: Buffer) -> str:
            if buffer.serial not in value_of:              # first use: from memory
                name = register()
                lines.append(f"real {name} = ((const real *){field(buffer)})[i];")
                value_of[buffer.serial] = name
            return value_of[buffer.serial]

        final = {}                                         # serial -> (register, Buffer) of its last write
        for use in group.uses:
            arguments, settled = ["i"], []
            for (kind, _), value in use.bound:
                if kind == "in":
                    arguments.append(read(value))
                elif kind in ("out", "inout"):
                    name = register()
                    lines.append(f"real {name} = {read(value)};" if kind == "inout" else f"real {name};")
                    arguments.append(name)
                    settled.append((name, value))
                elif kind.startswith("pointer"):
                    pointee = {"pointer": "real", "pointer:d": "double", "pointer:l": "long"}[kind]
                    arguments.append(f"(const {pointee} *){field(value)}")
                elif kind == "setting:d":
                    arguments.append(f"__longlong_as_double({field(float(value))})")
                else:
                    arguments.append(field(int(value)))
            lines.append(f"{use.op.name}({', '.join(arguments)});")
            for name, buffer in settled:
                lines.append(f"{name} = settle({name});")
                value_of[buffer.serial] = name
                final[buffer.serial] = (name, buffer)
        for serial, (name, buffer) in final.items():
            if serial in stored:
                lines.append(f"((real *){field(buffer)})[i] = {name};")

        width = len(self.fields) + 2
        body = "\n".join("    " + line for line in lines)
        text = f"""(const long *rows, long segments)
{{
    long first = 0, last = segments;              /* this block's row: the last one starting at or before it */
    while (last - first > 1) {{
        long middle = (first + last) / 2;
        if (rows[middle * {width} + 1] <= blockIdx.x) first = middle; else last = middle;
    }}
    const long *row = rows + first * {width};     /* count, first block, then the fields */
    long i = (blockIdx.x - row[1]) * (long)blockDim.x + threadIdx.x;
    if (i >= row[0]) return;
{body}
}}
"""
        self.name = "fused_" + hashlib.sha256((self.code + text).encode()).hexdigest()[:12]
        self.source = f"/* {' -> '.join(self.names)} */\nKERNEL {self.name}{text}"

    def key(self) -> tuple[str, str]:
        return self.code, self.source


class Batch:
    """Groups with the same kernel, on arrays that do not meet: one launch,
    each group a row of the table."""

    def __init__(self, kernel: Kernel, group: Group) -> None:
        self.kernel, self.kernels, self.groups = kernel, [kernel], [group]
        self.written, self.used = set(group.written), set(group.used)

    def accepts(self, kernel: Kernel, group: Group) -> bool:
        return kernel.key() == self.kernel.key() and group.independent_of(self)

    def add(self, kernel: Kernel, group: Group) -> None:
        self.kernels.append(kernel)
        self.groups.append(group)
        self.written |= group.written
        self.used |= group.used


class Program:
    """A traced step, compiled: ``steps`` are what is sent to the GPU, in
    order, each an Op (sent as it was) or a Batch (one generated kernel)."""

    def __init__(self, ops: list[Op], kept: Callable[[Buffer], bool] = Buffer.alive) -> None:
        self.ops = ops
        self.launches_before = sum(op.kind == "launch" for op in ops)      # kernels (and below, with
        self.nodes_before = len(ops)                                       # copies and clears)
        last_use: dict[int, int] = {}
        for index, op in enumerate(ops):
            for buffer in op.buffers():
                last_use[buffer.serial] = index

        # Neighbouring fusible ops of one size, in groups.
        pieces: list = []                                  # Ops and Groups, in order
        for index, op in enumerate(ops):
            use = _use(index, op)
            if use is None:
                pieces.append(op)
            elif pieces and type(pieces[-1]) is Group and pieces[-1].accepts(use):
                pieces[-1].add(use)
            elif len(pieces) > 1 and type(pieces[-2]) is Group and pieces[-2].accepts(use) and _apart(*pieces[-2:]):
                # One piece in the way (a bias's copy between a product and
                # its sum) that shares no array with the group before it:
                # it runs first, and the group goes on.
                pieces[-2], pieces[-1] = pieces[-1], pieces[-2]
                pieces[-1].add(use)
            else:
                pieces.append(Group(use))
        # A group of one kernel on independent arrays (Adam on two parameters
        # of one size) is split again, so that it joins the batch of all the
        # other parameters' updates below.
        pieces = [part for piece in pieces for part in (_split(piece) if type(piece) is Group else [piece])]

        # Each group's code: a value is stored if something reads it later or holds it at the end.
        self.steps: list = []
        for piece in pieces:
            if type(piece) is not Group:
                self.steps.append(piece)
                continue
            end = piece.uses[-1].index
            stored = {b.serial for u in piece.uses for b in u.writes if kept(b) or last_use[b.serial] > end}
            kernel = Kernel(piece, stored)
            previous = self.steps[-1] if self.steps else None
            # The same kernel on independent arrays right before: one launch for both.
            if type(previous) is Batch and previous.accepts(kernel, piece):
                previous.add(kernel, piece)
            else:
                self.steps.append(Batch(kernel, piece))
        # A lone kernel gains nothing from being generated: it runs as it was.
        self.steps = [step.groups[0].uses[0].op if type(step) is Batch and len(step.groups) == 1
                      and len(step.groups[0].uses) == 1 else step for step in self.steps]
        self.kernels = {}                                  # (code, source) -> Kernel, one of each
        for step in self.steps:
            if type(step) is Batch:
                self.kernels.setdefault(step.kernel.key(), step.kernel)
        self.launches_after = sum(type(step) is Batch or step.kind == "launch" for step in self.steps)
        self.nodes_after = len(self.steps)
        self._tables: list[DeviceArray] = []

    def source(self, code: str) -> str:
        """The generated CUDA source for one precision (all of its kernels)."""

        used = sorted({name for (c, _), kernel in self.kernels.items() if c == code for name in kernel.names})
        functions = "\n".join(FUSIBLE[name].function(name) for name in used)
        kernels = "\n".join(kernel.source for (c, _), kernel in self.kernels.items() if c == code)
        return f"/* Generated by fusion.py */\n{PRELUDE}\n{functions}\n{kernels}"

    def __str__(self) -> str:
        lines = [f"{self.launches_before} kernels -> {self.launches_after}; "
                 f"with copies and clears {self.nodes_before} -> {self.nodes_after}"]
        for step in self.steps:
            if type(step) is Batch:
                names = " -> ".join(step.kernel.names)
                lines.append(f"  fused: {names}" + (f"  (x{len(step.groups)} arrays)" if len(step.groups) > 1 else ""))
        return "\n".join(lines)

    # -- sending it to the GPU --------------------------------------------

    def compile(self) -> None:
        for code in sorted({c for c, _ in self.kernels}):
            names = [kernel.name for (c, _), kernel in self.kernels.items() if c == code]
            kernels_cuda.load_kernels(_compile(self.source(code), code), names, code)

    def send(self) -> None:
        """Send every step to the GPU (while a graph records: into the graph)."""

        self.compile()
        driver, check = kernels_cuda._driver, kernels_cuda._check
        for step in self.steps:
            if type(step) is Batch:
                self._launch(step)
            elif step.kind == "launch":
                kernels_cuda.launch(step.name, step.code, step.grid, step.block, *_raw(step.args), shared=step.shared)
            elif step.kind == "copy":
                source, start, target, offset, count = step.args
                size = kernels_cuda._ITEM_SIZE[source.typecode]
                check(driver.cuMemcpyDtoDAsync_v2(target.pointer + offset * size, source.pointer + start * size,
                                                  count * size, kernels_cuda._stream), "cuMemcpyDtoD")
            else:
                out, count = step.args
                size = count * kernels_cuda._ITEM_SIZE[out.typecode]
                check(driver.cuMemsetD8Async(out.pointer, 0, size, kernels_cuda._stream), "cuMemsetD8")

    def _launch(self, batch: Batch) -> None:
        rows, first = [], 0
        for kernel in batch.kernels:
            rows += [kernel.size, first] + kernel.fields
            first += blocks(kernel.size)
        table = DeviceArray("l", rows)                     # uploaded while recording: a constant of the graph
        self._tables.append(table)
        kernels_cuda.launch(batch.kernel.name, batch.kernel.code, first, BLOCK, table, len(batch.kernels))


def _apart(group: Group, piece: Group | Op) -> bool:
    """Whether ``piece`` (right after ``group``) can run before it: neither
    writes an array the other uses (an Op counts as writing all of its)."""

    if type(piece) is Group:
        return group.independent_of(piece)
    return not {b.pointer for b in piece.buffers()} & group.used


def _split(group: Group) -> list[Group]:
    """The group, or its ops one by one if they are one kernel on arrays that do not meet."""

    if len({use.op.name for use in group.uses}) > 1:
        return [group]
    singles = [Group(use) for use in group.uses]
    together = Group(group.uses[0])
    for single in singles[1:]:
        if not single.independent_of(together):
            return [group]
        together.add(single.uses[0])
    return singles


def _raw(args) -> list:
    """Launch arguments with each Buffer as its pointer (the same 8 bytes)."""

    return [a.pointer if type(a) is Buffer else a for a in args]


# ---------------------------------------------------------------------------
# Compiling (NVRTC, cached on disk as kernels_cuda does for kernels.cu)
# ---------------------------------------------------------------------------


def _architecture() -> str:
    major, minor = ctypes.c_int(), ctypes.c_int()
    kernels_cuda._check(kernels_cuda._driver.cuDeviceGetAttribute(ctypes.byref(major), 75, 0), "compute capability")
    kernels_cuda._check(kernels_cuda._driver.cuDeviceGetAttribute(ctypes.byref(minor), 76, 0), "compute capability")
    return f"sm_{major.value}{minor.value}"


def _compile(source: str, code: str) -> bytes:
    """Generated source as a cubin, with kernels.cu's options."""

    options = [f"--gpu-architecture={_architecture()}", f"-DREAL={'double' if code == 'd' else 'float'}",
               "-default-device", "--std=c++17"]
    digest = hashlib.sha256((source + " ".join(options)).encode()).hexdigest()[:16]
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "nanograd"
    cached = cache / f"fused_{digest}.cubin"
    if cached.exists():
        return cached.read_bytes()
    nvrtc = ctypes.CDLL(kernels_cuda._find_nvrtc())
    program = ctypes.c_void_p()
    if nvrtc.nvrtcCreateProgram(ctypes.byref(program), source.encode(), b"fused.cu", 0, None, None):
        raise RuntimeError("nvrtcCreateProgram failed")
    flags = (ctypes.c_char_p * len(options))(*[o.encode() for o in options])
    status = nvrtc.nvrtcCompileProgram(program, len(options), flags)
    size = ctypes.c_size_t()
    nvrtc.nvrtcGetProgramLogSize(program, ctypes.byref(size))
    log = ctypes.create_string_buffer(size.value)
    nvrtc.nvrtcGetProgramLog(program, log)
    if status != 0:
        raise RuntimeError("generated kernels did not compile:\n" + log.value.decode() + "\n" + source)
    nvrtc.nvrtcGetCUBINSize(program, ctypes.byref(size))
    cubin = ctypes.create_string_buffer(size.value)
    nvrtc.nvrtcGetCUBIN(program, cubin)
    nvrtc.nvrtcDestroyProgram(ctypes.byref(program))
    cache.mkdir(parents=True, exist_ok=True)
    temporary = cached.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_bytes(cubin.raw)
    temporary.replace(cached)
    return cubin.raw


# ---------------------------------------------------------------------------
# The API
# ---------------------------------------------------------------------------


def capture(step: Callable[[], object], *inputs: DeviceArray) -> kernels_cuda.Graph:
    """``kernels_cuda.capture`` with the step's elementwise kernels fused.

    Same contract (run the step eagerly once first; the step must not read
    GPU values). The returned graph's ``program`` is the compiled step:
    ``print(graph.program)`` lists the fused groups and the launch counts,
    ``graph.program.source("f")`` is the generated CUDA.
    """

    def traced():
        graph = kernels_cuda._recording
        graph.tracer = tracer = Tracer()
        try:
            output = step()
        finally:
            graph.tracer = None
        gc.collect()                              # temporaries in reference cycles: gone, so not stored
        program = Program(tracer.ops)
        program.send()
        graph.program = program
        return output

    return kernels_cuda.capture(traced, *inputs)
