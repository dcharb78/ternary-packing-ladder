#!/usr/bin/env python3
"""Train fp, ternary, or binary byte-level GPT arms on the enwik8 split."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten


VOCAB_SIZE = 256
CTX = 256
N_LAYER = 6
N_HEAD = 6
BATCH_SIZE = 32
TRAIN_BYTES = 90_000_000
VAL_BYTES = 5_000_000
EXPECTED_BYTES = 100_000_000
EVAL_WINDOWS = 200
EVAL_STRIDE = 24_999
EVAL_BATCH_SIZE = 20
LOG2 = math.log(2.0)


class QuantLinear(nn.Module):
    """Linear layer with a live fp32 master and an optional STE quantizer."""

    def __init__(
        self, input_dims: int, output_dims: int, arm: str, bias: bool = True
    ) -> None:
        super().__init__()
        scale = math.sqrt(1.0 / input_dims)
        # Masters are explicitly fp32 for QAT stability; checkpoint accounting
        # below follows the experiment's requested 16-bit non-quantized budget.
        self.weight = mx.random.uniform(
            low=-scale, high=scale, shape=(output_dims, input_dims)
        ).astype(mx.float32)
        if bias:
            self.bias = mx.random.uniform(
                low=-scale, high=scale, shape=(output_dims,)
            ).astype(mx.float32)
        self.arm = arm

    def quantized_weight(self) -> tuple[mx.array, mx.array]:
        """Return the current discrete symbols and per-tensor gamma."""
        gamma = mx.mean(mx.abs(self.weight))
        if self.arm == "ternary":
            divisor = mx.maximum(gamma, mx.array(1e-12, dtype=self.weight.dtype))
            q = mx.clip(mx.round(self.weight / divisor), -1, 1)
        elif self.arm == "binary":
            # mx.sign(0) is zero; the binary contract requires sign(0) = +1.
            q = mx.where(self.weight >= 0, 1, -1)
        else:
            raise ValueError("quantized_weight is only defined for quantized arms")
        return q.astype(self.weight.dtype), gamma

    def __call__(self, x: mx.array) -> mx.array:
        if self.arm == "fp":
            weight = self.weight
        else:
            q, gamma = self.quantized_weight()
            quantized = gamma * q
            weight = self.weight + mx.stop_gradient(quantized - self.weight)
        if "bias" in self:
            return mx.addmm(self.bias, x, weight.T)
        return x @ weight.T


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, arm: str) -> None:
        super().__init__()
        self.qkv = QuantLinear(dim, 3 * dim, arm, bias=True)
        self.proj = QuantLinear(dim, dim, arm, bias=True)
        self.dim = dim
        self.head_dim = dim // N_HEAD

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        batch, length, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = mx.split(qkv, 3, axis=-1)
        q = q.reshape(batch, length, N_HEAD, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, length, N_HEAD, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, length, N_HEAD, self.head_dim).transpose(0, 2, 1, 3)
        scores = (q @ k.swapaxes(-1, -2)) * (self.head_dim**-0.5)
        probs = mx.softmax(scores + mask, axis=-1)
        out = probs @ v
        out = out.transpose(0, 2, 1, 3).reshape(batch, length, self.dim)
        return self.proj(out)


def gelu(x: mx.array) -> mx.array:
    """Exact GELU, kept inline so no activation-version default is implicit."""
    return 0.5 * x * (1.0 + mx.erf(x / math.sqrt(2.0)))


RELU2 = False

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, arm: str) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, arm)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn_in = QuantLinear(dim, 4 * dim, arm, bias=True)
        self.ffn_out = QuantLinear(4 * dim, dim, arm, bias=True)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        x = x + self.attn(self.ln1(x), mask)
        h = self.ffn_in(self.ln2(x))
        h = mx.square(mx.maximum(h, 0.0)) if RELU2 else gelu(h)
        x = x + self.ffn_out(h)
        return x


class ByteGPT(nn.Module):
    def __init__(self, dim: int, arm: str) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, dim)
        self.position_embedding = nn.Embedding(CTX, dim)
        self.blocks = [TransformerBlock(dim, arm) for _ in range(N_LAYER)]
        self.final_norm = nn.LayerNorm(dim)
        self.lm_head = QuantLinear(dim, VOCAB_SIZE, arm, bias=False)

    def __call__(self, tokens: mx.array) -> mx.array:
        length = tokens.shape[1]
        positions = mx.arange(length)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        mask = nn.MultiHeadAttention.create_additive_causal_mask(length, x.dtype)
        for block in self.blocks:
            x = block(x, mask)
        return self.lm_head(self.final_norm(x))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an enwik8 byte-GPT weight-quantization arm."
    )
    parser.add_argument("--arm", choices=("fp", "ternary", "binary"), required=True)
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--eval_every", type=int, default=250)
    parser.add_argument("--wd", type=float, default=0.05,
                        help="weight decay on rank>=2 matrices (capacity pilot)")
    parser.add_argument("--dzreg", type=float, default=0.0,
                        help="dead-zone hinge coefficient: pushes |W| out of the zero band")
    parser.add_argument("--relu2", action="store_true",
                        help="use ReLU^2 MLP activation (BitNet-style) instead of GELU")
    parser.add_argument("--zinduce", type=float, default=0.0,
                        help="inverse hinge: penalizes |W| above gamma/2, pushing weights INTO the dead zone")
    args = parser.parse_args()
    if args.dim <= 0 or args.dim % N_HEAD != 0:
        parser.error(f"--dim must be positive and divisible by {N_HEAD}")
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.eval_every <= 0:
        parser.error("--eval_every must be positive")
    if args.lr is not None and args.lr <= 0:
        parser.error("--lr must be positive")
    return args


def load_data(root: Path) -> tuple[np.ndarray, np.ndarray]:
    path = root / "data" / "enwik8"
    if not path.is_file():
        raise FileNotFoundError(f"missing dataset: {path}")
    size = path.stat().st_size
    if size != EXPECTED_BYTES:
        raise ValueError(f"expected {EXPECTED_BYTES} bytes in {path}, found {size}")
    # Map exactly train+val bytes. The final 5,000,000 bytes are never mapped.
    mapped = np.memmap(
        path, mode="r", dtype=np.uint8, shape=(TRAIN_BYTES + VAL_BYTES,)
    )
    return mapped[:TRAIN_BYTES], mapped[TRAIN_BYTES : TRAIN_BYTES + VAL_BYTES]


def sample_batch(
    train: np.ndarray, rng: np.random.Generator
) -> tuple[mx.array, mx.array]:
    starts = rng.integers(0, len(train) - CTX, size=BATCH_SIZE, dtype=np.int64)
    offsets = np.arange(CTX + 1, dtype=np.int64)
    windows = np.asarray(train[starts[:, None] + offsets[None, :]], dtype=np.int32)
    return mx.array(windows[:, :-1]), mx.array(windows[:, 1:])


def fixed_validation_batches(val: np.ndarray) -> list[tuple[mx.array, mx.array]]:
    starts = [
        i * EVAL_STRIDE
        for i in range(EVAL_WINDOWS)
        if i * EVAL_STRIDE + CTX + 1 <= len(val)
    ][:EVAL_WINDOWS]
    if len(starts) != EVAL_WINDOWS:
        raise ValueError(f"validation split only admits {len(starts)} fixed windows")
    batches: list[tuple[mx.array, mx.array]] = []
    offsets = np.arange(CTX + 1, dtype=np.int64)
    for first in range(0, EVAL_WINDOWS, EVAL_BATCH_SIZE):
        batch_starts = np.asarray(
            starts[first : first + EVAL_BATCH_SIZE], dtype=np.int64
        )
        windows = np.asarray(
            val[batch_starts[:, None] + offsets[None, :]], dtype=np.int32
        )
        batches.append((mx.array(windows[:, :-1]), mx.array(windows[:, 1:])))
    return batches


def loss_fn(model: ByteGPT, x: mx.array, y: mx.array) -> mx.array:
    return nn.losses.cross_entropy(model(x), y, reduction="mean")


def evaluate(
    model: ByteGPT, val_batches: list[tuple[mx.array, mx.array]]
) -> float:
    total_nats = 0.0
    total_tokens = 0
    for x, y in val_batches:
        loss = loss_fn(model, x, y)
        mx.eval(loss)
        tokens = int(y.size)
        total_nats += float(loss.item()) * tokens
        total_tokens += tokens
    return total_nats / total_tokens / LOG2


def learning_rate_at(step: int, steps: int, peak_lr: float) -> float:
    if step <= 100:
        return peak_lr * step / 100.0
    progress = (step - 100) / (steps - 100)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * (0.1 + 0.9 * cosine)


def quant_layers(model: ByteGPT) -> list[tuple[str, QuantLinear]]:
    layers: list[tuple[str, QuantLinear]] = []

    def collect(name: str, module: nn.Module) -> None:
        if isinstance(module, QuantLinear):
            layers.append((name, module))

    model.apply_to_modules(collect)
    return sorted(layers, key=lambda item: item[0])


def parameter_counts(model: ByteGPT, arm: str) -> tuple[int, int]:
    total = sum(int(param.size) for _, param in tree_flatten(model.parameters()))
    quantized = (
        sum(int(layer.weight.size) for _, layer in quant_layers(model))
        if arm != "fp"
        else 0
    )
    return total, quantized


def final_quantization(
    model: ByteGPT, arm: str, dim: int, root: Path
) -> tuple[list[dict[str, object]], float]:
    stats: list[dict[str, object]] = []
    arrays: dict[str, np.ndarray] = {}
    realized_quantized_bits = 0.0
    for name, layer in quant_layers(model):
        q_mx, gamma_mx = layer.quantized_weight()
        mx.eval(q_mx, gamma_mx)
        q = np.asarray(q_mx, dtype=np.int8)
        gamma = np.asarray(gamma_mx, dtype=np.float32)
        n = int(q.size)
        counts = np.asarray(
            [np.count_nonzero(q == -1), np.count_nonzero(q == 0), np.count_nonzero(q == 1)],
            dtype=np.int64,
        )
        probabilities = counts.astype(np.float64) / n
        entropy = -sum(float(p) * math.log2(float(p)) for p in probabilities if p)
        stats.append(
            {
                "name": name,
                "n": n,
                "p_m1": float(probabilities[0]),
                "p_0": float(probabilities[1]),
                "p_p1": float(probabilities[2]),
                "entropy_bits": float(entropy),
            }
        )
        arrays[f"q__{name}"] = q
        arrays[f"gamma__{name}"] = gamma
        realized_quantized_bits += n * entropy
    np.savez(root / f"ckpt_{arm}_{dim}.npz", **arrays)
    return stats, realized_quantized_bits


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    train, val = load_data(root)

    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    mx.random.seed(args.seed)

    global RELU2
    RELU2 = bool(getattr(args, "relu2", False))
    model = ByteGPT(args.dim, args.arm)
    # Keep every master parameter fp32, including embeddings, norms, and biases.
    model.apply(lambda param: param.astype(mx.float32))
    mx.eval(model.parameters())

    peak_lr = args.lr if args.lr is not None else (3e-3 if args.dim <= 192 else 1.5e-3)
    # Use 0.05 AdamW decay only for rank>=2 weight matrices. Biases and norm
    # vectors use a separate zero-decay AdamW arm through MultiOptimizer.
    decay = optim.AdamW(learning_rate=peak_lr, weight_decay=args.wd)
    no_decay = optim.AdamW(learning_rate=peak_lr, weight_decay=0.0)
    optimizer = optim.MultiOptimizer(
        [decay, no_decay], filters=[lambda _name, param: param.ndim >= 2]
    )
    val_batches = fixed_validation_batches(val)

    def batch_loss(x: mx.array, y: mx.array) -> mx.array:
        loss = loss_fn(model, x, y)
        if (args.dzreg > 0.0 or args.zinduce > 0.0) and args.arm != "fp":
            reg = mx.array(0.0)
            zreg = mx.array(0.0)
            layers = quant_layers(model)
            for _, layer in layers:
                w = layer.weight
                gamma = mx.mean(mx.abs(w))
                if args.dzreg > 0.0:
                    reg = reg + mx.mean(mx.maximum(0.5 - mx.abs(w) / gamma, 0.0))
                if args.zinduce > 0.0:
                    zreg = zreg + mx.mean(mx.maximum(mx.abs(w) / gamma - 0.5, 0.0))
            loss = loss + (args.dzreg * reg + args.zinduce * zreg) / len(layers)
        return loss

    loss_and_grad = nn.value_and_grad(model, batch_loss)
    started = time.perf_counter()
    final_val_bpc = math.nan
    train_bpc_last = math.nan

    # Rotation/concurrency readouts (quantized arms): at each eval boundary,
    # flip_frac = fraction of trits changed since the previous eval snapshot
    # (the rotation letters of the training walk), plus the running alphabet
    # (p_-1, p_0, p_+1) so the settling of the ternary alphabet is a time
    # series, not just a final readout. Readouts only — verdicts unchanged.
    prev_q: dict[str, np.ndarray] = {}
    flip_path: list[dict[str, float]] = []

    def snapshot_quant(step: int) -> None:
        nonlocal prev_q
        if args.arm == "fp":
            return
        cur: dict[str, np.ndarray] = {}
        total = changed = 0
        counts = np.zeros(3, dtype=np.int64)
        for name, layer in quant_layers(model):
            q_mx, _ = layer.quantized_weight()
            mx.eval(q_mx)
            q = np.asarray(q_mx, dtype=np.int8)
            cur[name] = q
            counts += [np.count_nonzero(q == v) for v in (-1, 0, 1)]
            if name in prev_q:
                total += q.size
                changed += int(np.count_nonzero(q != prev_q[name]))
        entry = {
            "step": step,
            "p_m1": float(counts[0] / counts.sum()),
            "p_0": float(counts[1] / counts.sum()),
            "p_p1": float(counts[2] / counts.sum()),
        }
        if total:
            entry["flip_frac"] = float(changed / total)
        flip_path.append(entry)
        prev_q = cur

    for step in range(1, args.steps + 1):
        x, y = sample_batch(train, rng)
        scheduled_lr = learning_rate_at(step, args.steps, peak_lr)
        optimizer.learning_rate = scheduled_lr
        loss, grads = loss_and_grad(x, y)
        grads, _ = optim.clip_grad_norm(grads, max_norm=1.0)
        optimizer.update(model, grads)
        # Force the loss, new fp masters, and all AdamW state each step.
        mx.eval(loss, model.parameters(), optimizer.state)
        train_bpc_last = float(loss.item()) / LOG2

        if step % args.eval_every == 0 or step == args.steps:
            final_val_bpc = evaluate(model, val_batches)
            snapshot_quant(step)
            elapsed = time.perf_counter() - started
            tok_s = step * BATCH_SIZE * CTX / elapsed
            print(
                f"step {step} train_bpc {train_bpc_last:.6f} "
                f"val_bpc {final_val_bpc:.6f} tok_s {tok_s:.2f}",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    tokens = args.steps * BATCH_SIZE * CTX
    params_total, params_quantized = parameter_counts(model, args.arm)
    if args.arm == "fp":
        weight_stats: list[dict[str, object]] = []
        bits_ideal = params_total * 16
        bits_realized = float(bits_ideal)
    else:
        weight_stats, realized_quantized_bits = final_quantization(
            model, args.arm, args.dim, root
        )
        fp_params = params_total - params_quantized
        # Exact container arithmetic (repo register, never a float log):
        # ternary ideal = minimal P with 3^N <= 2^P = (3**N).bit_length()
        # (3^N is never a power of two for N >= 1); binary ideal = N bits.
        if args.arm == "ternary":
            bits_ideal = (3 ** params_quantized).bit_length() + fp_params * 16
        else:
            bits_ideal = params_quantized + fp_params * 16
        bits_realized = realized_quantized_bits + fp_params * 16

    result = {
        "arm": args.arm,
        "dim": args.dim,
        "steps": args.steps,
        "seed": args.seed,
        "tokens": tokens,
        "val_bpc": float(final_val_bpc),
        "train_bpc_last": float(train_bpc_last),
        "params_total": int(params_total),
        "params_quantized": int(params_quantized),
        "weight_stats": weight_stats,
        "flip_path": flip_path,
        "bits_ideal": int(bits_ideal),
        "bits_realized": float(bits_realized),
        "lr": float(peak_lr),
        "wd": float(args.wd),
        "dzreg": float(args.dzreg),
        "zinduce": float(args.zinduce),
        "relu2": bool(getattr(args, "relu2", False)),
        "time_s": float(elapsed),
        "tok_per_s": float(tokens / elapsed),
    }
    if not all(
        math.isfinite(result[key])
        for key in ("val_bpc", "train_bpc_last", "time_s", "tok_per_s")
    ):
        raise RuntimeError("refusing to write non-finite run metrics")
    with (root / "runs.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
