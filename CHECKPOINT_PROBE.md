# Real checkpoint probe — BitNet b1.58 2B-4T

*Targeted axis / pad-to-tax0 / structure scan on the public ternary payload.
Checkpoint downloaded locally to `repack/data/bitnet/` (gitignored, ~1.18 GB).*

```bash
# once: place model.safetensors under repack/data/bitnet/
python3 repack/checkpoint_axis_probe.py --rt-sample 5 --structure-sample 8
```

## Headline (210 tensors, 2,084,044,800 trits)

| ledger | bytes |
|--------|------:|
| shipped U8 2-bit | 521,011,200 |
| flat `fmt_5_8` | 416,808,960 |
| best-axis fiber `fmt_41_65` | 457,344,000 |
| axis choice savings (within 41_65) | **168,960** |
| axis bits saved (holonomy) | 215,040 |
| pad-to-tax0 vs fiber-41 on best axis (approx) | **~26.2 MB** |

Round-trip samples (5 tensors): flat `fmt_5_8` + best-axis `(5,)` fiber pack — exact.

## By module (axis savings)

| module | tensors | trits | axis bits saved | save41 B |
|--------|--------:|------:|----------------:|---------:|
| down/gate/up_proj | 30 each | 530.8M each | 46,080 each | 30,720 each |
| k_proj / v_proj | 30 each | 49.2M each | 38,400 each | 38,400 each |
| q_proj / o_proj | 30 each | 196.6M each | **0** | **0** |

Squares (2560×2560) contribute nothing; rectangles (6912×2560, 640×2560) carry all of the free axis win. Matches the scale-probe holonomy story.

Unique logical shapes: `(2560,6912)`, `(6912,2560)`, `(640,2560)`, `(2560,2560)`.

## Structure (Kronecker / identical tiles)

Identical-tile scan on 8 tensors: **0 hits**. This BitNet payload is dense ternary
without exact repeated block factors — the ~100× Kronecker win does **not**
apply without a redesign that keeps factors.

## Honest verdict

1. **Axis choice on the real artifact is real but small** (~165 KB at `fmt_41_65`
   fiber packing). Worth wiring as a free default; not a headline MB win here.
2. **Pad-to-tax0 is larger within the fiber-41 family** (~26 MB) but still does
   **not** beat flat `fmt_5_8` (417 MB) on this model — 5-per-byte remains the
   practical density knee for unstructured ternary, consistent with the 1-D Pareto.
3. **No exact grouped/Kronecker structure** in the shipped weights → skip
   factor-pack for this checkpoint; pursue on models that retain factors.
4. **Next system-level step** only if a checkpoint is already near the 5_8 /
   stream regime *and* has rectangular + structured layers where axis/pad/factor
   compound; otherwise prefer stream/adaptive (already measured in the main note).

Full JSON: [`repack/checkpoint_axis_results.json`](repack/checkpoint_axis_results.json).
