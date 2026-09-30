# HSIFusion and SHARP assessment repairs

The canonical trainer now defaults to **300,000 successful optimizer updates**, validation every 1,000 updates, and recovery checkpoints every 5,000 updates. `--epochs 300` means 300 logical blocks of 1,000 updates; dataset size and gradient accumulation no longer multiply the budget. `--max_optimizer_steps` overrides that budget. The cosine schedule, loss-floor continuation, EMA, validation, and checkpoint counters use successful updates. An invalid microbatch discards its entire accumulation group. Data iteration continues across logical blocks and recycles at the end of the loader.

The ordinary CLI retains its floored training objective and existing model architecture. Those defaults are a stability recipe, **not an exact reference objective**. The JSON recipes below use exact positive-target MRAE, FP32 data/optimization, plain Adam, no clipping/EMA/warmup/auxiliary loss, whole-scene RGB min-max normalization before cropping, uniform paired MST rotations/flips, and a floor patch grid. HSIFusion controls also disable MoE. HSI values are never rescaled. Zero or negative targets make exact MRAE undefined and raise an error; use an explicit floored experiment if such targets are part of the intended dataset.

## Architecture controls

| Control | HSIFusion | SHARP | Effect |
| --- | --- | --- | --- |
| `qkv_layout="group_major"` | Pooled spectral QKV | Multi-scale/sparse QKV and decoder KV | Unpacks grouped convolution outputs as group → Q/K/V → channels, giving each part the same input-group support. Weight shapes stay the same, but outputs change. |
| `conv_norm=false`, `rgb_skip=true` | Stem/down/up/decoder/head | Stem/down/up/head | Leaves convolution features unnormalized and adds a learned unnormalized RGB → spectrum bypass. Transformer branch pre-normalization remains. |
| `fullres_spectral=true` | After final decoder | After final decoder | Channel attention plus gated FFN using full-resolution value maps. Attention scales linearly with pixel count; no spatial all-pairs matrix. Residual gates start at 0.01. |
| `layer_scale_init=0.01` | Encoder branches | — | Gives spatial, pooled spectral, and FFN branches a larger initial contribution; supplied as a separate ablation. |
| `aligned_skip=true` | Already has aligned concatenation | Decoder fusion | Uses `x + projected_skip + gate * attention`, preserving spatially aligned detail alongside pooled cross-attention. |
| `sparse_attention_mode="local_landmark"` | — | Every sparse stage | Holds the same attention operator across training patches and full-image evaluation. `auto` retains legacy switching; fixed `exact_topk` raises if its token limit is exceeded. |
| `head_mode="linear"`, `output_activation="none"`, `spectral_head_rank=0` | Already raw regression | Output head | Removes the multiplicative output gate, outer sigmoid, and fixed cosine correction in the linear-head experiment. |
| `strict_spectral_failures=true` | Pooled spectral branches | — | Raises projection/runtime/nonfinite errors instead of silently bypassing a failed branch. |

These controls default to legacy behavior. Existing checkpoints retain their grouped layout, norms, skips, and output parameterization. Train corrected configurations from scratch; loading old weights into a different layout silently changes their meaning even when shapes match.

## Experiments

There are 14 fresh-run recipes in [recipes](recipes): six for HSIFusion and eight for SHARP. Compare each model's `control` against `grouped_qkv`, `radiometry`, and `spectral`. HSIFusion additionally has `layer_scale`; SHARP additionally has `fixed_operator`, `aligned_skip`, and `linear_head`. Each ablation changes only its listed architecture control relative to that model's reference control. `candidate` combines the repairs for follow-up evaluation; it has **not** been established as a winning architecture.

Run from the repository root, substituting your ARAD-1K directory:

```bash
python "HSIFUSION&SHARP/unified_training.py" --recipe "HSIFUSION&SHARP/recipes/hsifusion_control.json" --data_root "path/to/ARAD_1K"
python "HSIFUSION&SHARP/unified_training.py" --recipe "HSIFUSION&SHARP/recipes/sharp_control.json" --data_root "path/to/ARAD_1K"
python "HSIFUSION&SHARP/unified_training.py" --recipe "HSIFUSION&SHARP/recipes/hsifusion_candidate.json" --data_root "path/to/ARAD_1K"
python "HSIFUSION&SHARP/unified_training.py" --recipe "HSIFUSION&SHARP/recipes/sharp_candidate.json" --data_root "path/to/ARAD_1K"
```

Explicit CLI arguments override recipe fields. Use a new `--experiment_name` for each seed/variant. Missing pairs fail by default; intentional exclusions use `--exclude_samples SCENE_ID ...` and are recorded. Validation targets always remain FP32, including when training uses FP16 storage. Reference recipes require the 128-pixel scoring border to fit every scene. Ordinary CLI runs fall back to full-frame selection for the **entire** split if any scene is too small, avoiding a partial crop average.

Before full training, prepare a fixed NPZ batch with normalized `rgb[N,3,H,W]` and `target[N,31,H,W]`, preferably four to eight 128×128 patches, then run:

```bash
python diagnose_reconstruction.py --model hsifusion --config "HSIFUSION&SHARP/recipes/hsifusion_control.json" --patches fixed_patches.npz --steps 1000 --device cuda --output hsifusion_probe.json
python diagnose_reconstruction.py --model sharp --config "HSIFUSION&SHARP/recipes/sharp_control.json" --patches fixed_patches.npz --steps 1000 --device cuda --output sharp_probe.json
```

Repeat with the ablation recipes. The probe fixes the batch, disables stochastic layers/MoE, and uses the same FP32 exact-MRAE implementation. The shared `train_reconstruction_reference.py` also accepts `hsifusion` and `sharp`, enabling comparison against MSWR, CSWIN, and the external official MST++ source using one data/optimization/evaluation loop. Full images are padded before inference and cropped back before scoring.

## Checkpoints and diagnostics

Version-2 checkpoints persist the fully resolved model dataclass, trainer policy, successful-update counter and budget, code revision/source hashes, split hashes, effective scene lists, preprocessing/storage policy, and raw/EMA weights. Inference and the repository benchmark reconstruct the saved architecture instead of selecting today's factory defaults. Inference also uses saved RGB normalization, metric epsilon, exclusions, and crop policy unless explicitly overridden.

Resume rejects changes to model semantics, data policy, update budget, optimizer/loss/selection settings, or inconsistent scheduler counters. Old version-1 checkpoints remain loadable for inference; their data-pass schedule cannot be resumed as the new update schedule. Resume restarts a shuffled data iterator and does not reproduce the exact interrupted patch order or worker random state.

`training_health.jsonl` samples actual parameter update ratios, gradient norms, residual/input RMS ratios, LayerScale/spectral gate values, and sparse operator selections. `validation_diagnostics.jsonl` reports per-scene/per-band MRAE, raw versus clamped predictions, out-of-range output fractions, and additive target-intensity bucket contributions for both scoring protocols. Selection always uses raw predictions.

## Validation limits

Synthetic CPU tests cover grouped input support, aligned local detail, operator consistency and token limits, brightness sensitivity, first-update spectral gradients, FP32 validation storage, exact MRAE, accumulation across loader cycles, resolved checkpoint round trips, semantic resume checks, and failure recovery. They verify implementation behavior. ARAD-1K convergence and improvements over the reported 0.20 MRAE require fresh matched training runs and are not claimed by these repairs.

Local validation: 147 model tests passed with the largest classifier-preset allocation test and two multiprocessing tests excluded from the combined sandbox run; both multiprocessing/DDP tests then passed separately outside the Windows sandbox, using pytest temporary directories. The 18 shared metric/benchmark/diagnostic checks passed as well. The largest classifier-preset test reproducibly stops in a native PyTorch allocation exception on this CPU environment, so a completely clean unfiltered suite is unverified. Source compilation and the whitespace check passed.

A saved [50-update synthetic probe](validation/synthetic_repair_probe.json) uses four fixed 64×64 patches and reduced 8-channel, one-block-per-stage candidate models. Raw exact MRAE fell from 0.9618 to 0.7231 for HSIFusion and from 1.4721 to 0.6781 for SHARP. These are short optimizer/gradient checks on synthetic targets, not a convergence result or a comparison against the control architectures.
