# CAS-HSI — Convolutional Attention Stack for Hyperspectral Reconstruction

RGB → 31-band hyperspectral reconstruction. Fully convolutional, resolution-agnostic,
exportable, with an MST++/NTIRE-faithful trainer for ARAD-1K (900 / 50 / 50).

Implements [`docs/CAS_HSI_Implementation_Steps_3_to_9.md`](docs/CAS_HSI_Implementation_Steps_3_to_9.md),
which is the normative specification for this package. Where the code deviates from it,
it says so in a comment and the reason is repeated in [Deviations](#deviations-from-the-spec)
below.

---

## Architecture

A three-resolution encoder–decoder. Spatial attention is confined to H/2 and H/4;
full resolution uses **CAS-Lite**, which keeps the transformer block shell but swaps
neighborhood attention for a 7×7 depthwise mixer — attention over H·W tokens is
memory-bound at full resolution and has no portable operator, while a large depthwise
kernel has neither problem.

```
RGB ──┬─► Conv1x1(3→31) linear prior ──────────────────────────────────┐
      │                                                                │
      └─► stem ─► CAS-Lite ×D0 ─┬─► ↓2 ─► CAS ×D1 ─┬─► ↓2 ─► CAS ×DB   │
                                │ S0               │ S1        (bottleneck)
                                │                  │            │
                                │                  └─► ↑2 ◄─────┘
                                │                      CAS ×DD1
                                └─► ↑2 ◄───────────────┘
                                    CAS-Lite ×DD0 ─► refinement ×DR
                                          │
                                          └─► Conv3x3(C→31) residual ──┤
                                                                       ▼
                                                          prediction = prior + residual
```

**Every CAS/CAS-Lite block is three pre-normalized, LayerScale-gated residual branches:**

| branch | what it does | cost |
|---|---|---|
| spatial mixer | *where* to look — dilated local attention, stripe attention, or a depthwise conv | depends on backend |
| cross-channel attention | *which spectra* to mix — a real C×C attention over latent channels, not an SE gate | O(C²·HW), linear in pixels and not O((HW)²) |
| gated conv FFN | *how* to transform — multiplicative gate, no terminal sigmoid | O(C²·HW) |

The identity path is never scaled, so `layer_scale_init=0` makes a block an exact
identity. The prediction is a **learned linear RGB→HSI prior plus a near-zero-initialized
deep residual**. The linear branch is trainable rather than camera-calibrated: before
training, it is a well-scaled initialization, not a physically meaningful spectrum.

### Variants

| variant | base_width | head_dim | widths | heads | params |
|---|---|---|---|---|---|
| `tiny` | 32 | 32 | 32 / 64 / 128 | 1 / 2 / 4 | **1,782,049** |
| `base` | 48 | 24 | 48 / 96 / 192 | 2 / 4 / 8 | **4,722,343** |
| `tiny` + `backend: edge` | 32 | 32 | 32 / 64 / 128 | — | **1,458,177** |

For scale: MST++ is 1.62 M parameters, so `tiny` is the like-for-like comparison.

### Backends

Set `backend: research` or `backend: edge` — the block shell is identical, only the
spatial mixer changes (spec 9.2), so the two are distillation-compatible.

| stage | research | edge |
|---|---|---|
| full res | depthwise 7×7 | depthwise 7×7 |
| H/2, H/4 | `dilated_local_attention` | `dilated_depthwise_conv` |
| every 3rd bottleneck block | `hybrid_local_stripe_attention` | `large_kernel_depthwise_conv` |
| cross-channel attention | **kept** | **kept** |

Cross-channel attention survives into the edge profile deliberately: its cost does not
scale with H·W and it is what carries the spectral prior. The **edge model is the export
path** — the research backend's neighborhood attention has no portable ONNX/TensorRT
operator (spec 9.7).

---

## Quick start

```bash
pip install -r requirements.txt

# 1. Verify your 900/50/50 split (checks GT coverage AND that the splits are disjoint)
python prepare_splits.py --data_root /path/to/ARAD_1K

# 2. Train (MST++ recipe)
python train_cas_hsi.py --config configs/cas_hsi_tiny.yaml --data_root /path/to/ARAD_1K

# 3. Evaluate the best checkpoint on the held-out test 50
python test_cas_ntire.py --checkpoint experiments/cas_hsi_tiny_*/checkpoints/best.pth \
                         --data_root /path/to/ARAD_1K --split test
```

No dataset or GPU needed to sanity-check the code:

```bash
python smoke_infer.py    # arbitrary sizes, tiling, edge backend
python smoke_train.py    # synthetic ARAD-1K: train → validate → checkpoint → test
python -m pytest tests/ -q
```

---

## Dataset

Standard ARAD-1K / MST++ layout:

```
ARAD_1K/
  Train_RGB/    ARAD_1K_0001.jpg …
  Train_Spec/   ARAD_1K_0001.mat …      (HDF5, 'cube' dataset, stored (bands, W, H))
  split_txt/
    train_list.txt   900 scenes
    valid_list.txt    50 scenes   → checkpoint selection
    test_list.txt     50 scenes   → reported once, at the end
```

`Valid_*` / `Test_*` directories are used when present and fall back to `Train_*`,
so a dataset that keeps every scene in one folder and separates splits purely by list
works unchanged.

**RGB is normalized per-image by its own min/max**, not `/255` — this is what MST++'s own
loader does, and mixing the two conventions silently shifts every metric. `rgb_norm: div255`
is available if you need it.

`prepare_splits.py` verifies the split rather than regenerating it. Its most useful check
is that the three lists are **disjoint**: one scene leaking from train into test turns a
held-out number into a training number, and nothing downstream would ever tell you.

> **Note on the public release.** NTIRE 2022 published 950 ground-truthed scenes (900 + 50)
> and withheld the challenge test cubes, so a public 900/50/50 with GT everywhere does not
> exist. You said your copy has all three — `--mode verify` (the default) confirms that.
> If you ever need to reproduce this on a copy that lacks a GT test set, `--mode holdout`
> carves a seeded one out of train and tells you loudly that train is then 850, not 900.

---

## Training protocol

The defaults in `configs/cas_hsi_tiny.yaml` **are** the MST++ recipe, so a number produced
by it is comparable to the ARAD-1K leaderboard and to the other models in this toolkit:

| | |
|---|---|
| loss | MRAE, ε = 1e-6, on the **unclamped** prediction |
| data | 128×128 patches, stride 8, rot90 + h/v flips (geometric only) |
| optimizer | Adam(0.9, 0.999), lr 4e-4, **no** weight decay, warmup, or gradient clipping |
| schedule | cosine → 1e-6, stepped per optimizer step |
| batch / epochs | 20 / 300, 1000 steps per epoch |
| validation | full scenes, batch 1, in fp32 |
| selection | MRAE on the MST++ centre crop (128-px border: 482×512 → 226×256) |

Deviating is fine — it just costs comparability, so say so when you report the number.

### What gets logged

Every epoch, for **both** train and validation: the loss plus **MRAE, PSNR, RMSE, SSIM**
(and SAM, MAE, which come free from the same pass).

```
epoch 42/300 | train loss 0.198441 | MRAE 0.1984  PSNR 32.14  RMSE 0.0184  SSIM 0.9312  SAM 3.71
             | val   loss 0.213077 | 50 scenes in 18.4s | EMA=off | selection: crop MRAE = 0.194502
  protocol        MRAE      RMSE      PSNR       SAM      SSIM       MAE
  ----------------------------------------------------------------------
  full          0.2131    0.0201   31.4102    3.9903    0.9268    0.0139
  crop          0.1945    0.0182   32.3110    3.7440    0.9351    0.0126
```

Train metrics are computed on the same batches as the loss, using definitions numerically
identical to the validation ones — so the train-minus-val gap means what you think it
means. Validation metrics come from the repo-wide `hsi_benchmark.metrics`, the same
implementation MSWR/SHARP/CSWin report against.

Written to the run directory: `train.log`, `config.json` (full model + training config),
`history.csv` (one row per epoch, every metric a column), `metrics.jsonl`,
`checkpoints/{last,best}.pth`.

### Two protocols, always

`full` = whole native frame (NTIRE-style). `crop` = MST++ centre region. MST++ selects on
the crop, so this trainer does too — reporting the full-frame number while selecting on the
crop would be comparing two different quantities.

### Clamping

The loss **never** sees a clamped prediction. `clamp` has zero gradient outside its range,
so clamping before the loss would silently freeze learning on exactly the elements furthest
from the target (spec 7.4). Evaluation is also unclamped by default — the NTIRE convention
and what the rest of this repo does. `--clamp_eval` exists, flatters MRAE, and must not be
mixed with unclamped numbers.

---

## Deployment

```python
from cas_hsi import build_cas_hsi
from cas_hsi.deployment import replace_attention_mixers, export_onnx

edge = build_cas_hsi("tiny", backend="edge")        # train from scratch, or
model, report = replace_attention_mixers(model)      # swap a trained research model's mixers
export_onnx(edge, "cas_hsi.onnx", opset=18)          # dynamic padded H and W
```

The ONNX graph accepts any H/W **divisible by 4** (at the traced batch size). For
an arbitrary-sized scene, apply `pad_to_multiple(rgb, 4)`, retain the `PadInfo`, run
ONNX Runtime on the padded RGB, then call `crop_to_original` on the output. This keeps
the graph's PixelUnshuffle reshapes valid at every exported resolution. Eager PyTorch
inference still accepts arbitrary sizes directly.

`replace_attention_mixers` gives the new convolutions **random weights** — it is the
starting point for distillation (spec 9.4), not a free edge model.

`benchmark_cas.py` implements the spec 9.10 protocol (params, MACs, latency percentiles,
peak memory, at 128² / 256² / 482×512 / 512²).

Benchmark note:

```bash
# Publishable protocol: keep the spec 50 warmup / 200 measured iterations.
python benchmark_cas.py --variant tiny --backend research --device cuda --dtype bf16 \
  --json bench_tiny_research_bf16.json

# CPU smoke check while developing: deliberately not a publishable latency number.
python benchmark_cas.py --variant tiny --backend edge --device cpu \
  --sizes 128x128 --warmup 1 --iters 2 --skip-memory
```

Report the backend, dtype, device, warmup/iteration counts, and whether full-scene or
tiled inference was used. Tiling is a memory workaround and an approximation for the
research backend: cross-channel attention aggregates over the spatial support it sees,
so per-tile attention is not mathematically identical to one full-scene forward.

---


## Layout

```
cas_hsi/
  config.py            CASHSIConfig, variants, construction-time validation (spec 3.2, 3.6)
  model.py             CASHSI, build_cas_hsi, build_edge_model                (spec 3)
  inference.py         tiled_inference with Hann blending                     (spec 8.7)
  layers/              padding, bias-free LayerNorm2d, LayerScale, DropPath,
                       PixelUnshuffle↓ / PixelShuffle↑, spectral heads   (spec 4.3–4.4, 6, 7, 8)
  blocks/              CASBlock, CASLiteBlock, cross-channel attention, gated FFN,
                       and every spatial mixer                           (spec 4.5–4.7, 5, 9.3)
  deployment/          replace_attention, export_onnx, quantization       (spec 9.2, 9.6, 9.8)
dataloader.py          ARAD-1K train (patches) / eval (full scenes)
metrics.py             MRAE / RMSE / PSNR / SSIM / SAM, matching hsi_benchmark
evaluation.py          the two-protocol scene evaluation shared by train and test
train_cas_hsi.py       the MST++/NTIRE trainer
test_cas_ntire.py      NTIRE evaluation: per-scene, per-band, bootstrap CI, .mat export
prepare_splits.py      verify the 900/50/50 split (GT coverage + leakage)
benchmark_cas.py       latency / MACs / memory                                (spec 9.10)
tests/                 shapes, arbitrary sizes, gradients, export, equivalence (spec 3.3)
```
