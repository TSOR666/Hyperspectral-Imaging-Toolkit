# MSWR and CSWIN assessment experiments

These changes implement testable architectural repairs and a common reference
pipeline. They do not establish a new ARAD-1K MRAE result. Existing recovery
recipes remain available; the assessment recipes require fresh training.

Install the existing MSWR and CSWIN runtime requirements before using the
shared scripts; they need both YAML/Hydra and the common MST dataset loader.

## Architecture changes

MSWR can enable `spectral_output_block`: one spectral attention + FFN residual
on the final full-resolution feature grid before reconstruction. It uses its
own `spectral_output_gate_init=0.01`, so both inner branches receive gradients
on the first update without changing gates elsewhere, adding depth throughout
the backbone, or enabling every spectral prelayer. Setting the gate to zero
preserves the old model's predictions after loading its weights. The feature
grid includes the network's normal bottom/right padding and is cropped back
to the original output size. One spectral head computes a dense channel
attention map; it does not guarantee full mathematical rank.

CSWIN can enable `sstb_residual_mode: correction`. The outer residual adds
`scale * (h - g)`, where `g` is the gated input and `h` includes that input plus
the spectral/spatial/FFN corrections. This removes the extra gated identity.
The existing `legacy` mode and outer scale remain unchanged by default.
Checkpoint export preserves the mode; resume and weight-only fine-tuning
reject a mode mismatch even though the weight shapes match.

CSWIN also accepts `cswin_attention_mode: local` to use the existing local
window operator at every resolution. `attention_operator(H, W)` reports the
selected operator without allocating a full-resolution attention map. The
historical `local_global` mode retains its threshold-dependent behavior.

## Experiment matrix

| Model | Recipe | Isolated comparison |
|---|---|---|
| MSWR | `configs/experiments/assessment_control.yaml` | Current recovery architecture, reference optimization |
| MSWR | `assessment_fullres_spectral.yaml` | Control + one full-resolution spectral block |
| MSWR | `assessment_haar.yaml` | Full-resolution block + Haar instead of db2 |
| MSWR | `assessment_no_wavelet.yaml` | Full-resolution block + wavelets disabled |
| CSWIN | `src/configs/assessment_control.yaml` | Residual-balanced recovery architecture, reference optimization |
| CSWIN | `assessment_correction.yaml` | Control + correction-only outer residual |
| CSWIN | `assessment_fixed_local.yaml` | Control + resolution-independent local attention |
| CSWIN | `assessment_candidate.yaml` | Combined correction and local attention; evaluate after isolated arms |

The controls use FP32, raw weights, Adam without weight decay, no warm-up,
no gradient clipping, exact positive-target MRAE, and 300,000 updates. They
score raw full-image predictions after removing a 128-pixel border on all
sides. The existing `baseline_mstpp.yaml` optimizer settings were also repaired
to disable warm-up, clipping, and AMP; use the fully specified assessment
control for the new comparisons.

Exact MRAE (`epsilon=0`) rejects zero/negative targets and non-finite values.
If supplied targets contain zeros, first inspect that data. An explicit floor
is a separate objective and must be reported as such; these scripts never
silently drop pixels or substitute a floor. Positive-floor legacy recipes
continue to work.

## Shared reference training

Run from the repository root, with the same data directory and exclusions for
all arms:

```powershell
python train_reconstruction_reference.py --model mswr --config mswr_v2/configs/experiments/assessment_fullres_spectral.yaml --data-root D:/ARAD_1K --output runs/mswr_fullres --device cuda
python train_reconstruction_reference.py --model cswin --config "CSWIN v2/src/configs/assessment_correction.yaml" --data-root D:/ARAD_1K --output runs/cswin_correction --device cuda
python train_reconstruction_reference.py --model mstpp --mst-root D:/MST-plus-plus --data-root D:/ARAD_1K --output runs/mstpp_control --device cuda
```

The trainer uses the same existing MST-style lazy dataset for all models:
BGR-to-RGB conversion, whole-scene min/max RGB normalization before cropping,
unmodified FP32 HSI, paired rotations/flips, and explicit split files. Missing
or corrupt scenes raise. Use repeatable `--exclude-scene ARAD_1K_0314` only when
that exclusion is intentionally shared by every arm. Unlike the standalone
CSWIN recovery trainer, the shared trainer has no implicit scene exclusion.

An epoch boundary never changes the optimizer-update budget. The common
cosine scheduler advances after each successful update. Validation restores
the previous model mode even when scoring fails. Checkpoint selection uses
raw exact MRAE; clamped MRAE is a separate deployment diagnostic. No GAN,
perceptual, SAM, or colour-consistency loss is added.

Each output directory contains `manifest.json`, `steps.jsonl`, `latest.pth`,
and `best.pth`. The manifest records the resolved architecture, optimization
and preprocessing policies, seed, code commit/dirty state, split/config hashes,
and explicit exclusions. Validation logs scene IDs, per-band MRAE, and target
intensity bucket contributions. Health samples record activation RMS, residual
contributions, gate values, and actual parameter-update ratios. The trainer is
a fresh-run, single-process reference implementation; it deliberately does not
resume optimizer state or use EMA. Use a new output directory for each run.

## Fixed-patch optimization probe

Save 4–8 paired training crops in an NPZ with `rgb[N,3,H,W]` and
`target[N,31,H,W]`, already preprocessed from whole source images. Use the same
file for MSWR, CSWIN, and MST++. Respect each model's minimum input size;
128-pixel patches are suitable for the standard architectures.

```powershell
python diagnose_reconstruction.py --model mswr --config mswr_v2/configs/experiments/assessment_fullres_spectral.yaml --patches fixed_patches.npz --steps 1000 --device cuda --output probes/mswr.json
python diagnose_reconstruction.py --model cswin --config "CSWIN v2/src/configs/assessment_correction.yaml" --patches fixed_patches.npz --steps 1000 --device cuda --output probes/cswin.json
python diagnose_reconstruction.py --model mstpp --mst-root D:/MST-plus-plus --patches fixed_patches.npz --steps 1000 --device cuda --output probes/mstpp.json
```

The probe disables augmentation, stochastic depth/dropout, AMP, weight decay,
clipping, and EMA. It uses identical raw exact MRAE and Adam for all models.
Pass `--split-file` to hash the source split lists in the report. Interpret a
small-set optimization result relative to MST++ rather than treating 0.02 as
a guaranteed threshold for arbitrary targets.

For independent checkpoint evaluation, `benchmark_hsi.py --mrae-epsilon 0`
selects the same exact denominator policy. Its general cross-dataset loader
can synthesize RGB or resample spectra, so it is not automatically an ARAD
protocol reproduction. Use the shared reference loader for ARAD parity.

## Verification and remaining measurements

Regression checks cover first-update movement of MSWR's attention/FFN and
CSWIN's spectral/spatial parameters, zero-correction identity behavior,
checkpoint semantics, resolution selection, metric/gradient agreement below
the former floors, additive dark-target contributions, validation mode
restoration, and real-loader training/checkpoint cycles for both models.
Haar/db2 round trips include constants, impulses at opposite boundaries,
ramps, random inputs, and gradients through three decomposition levels.

Local verification uses CPU PyTorch. Full ARAD training, official pretrained
MST++ evaluator parity, matched 300k runs, and multi-seed per-scene comparisons
remain empirical work. The new architecture options are candidates for those
experiments; no sub-0.20 result is claimed.
