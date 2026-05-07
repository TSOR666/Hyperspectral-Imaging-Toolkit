# CSWIN v2 Audit Report

Date: 2026-05-07

## Executive Summary

Verdict: Partial before fixes; improved after this patch set, with real-data/GPU validation still required.

The codebase is intended for RGB to 31-band HSI reconstruction on ARAD/MST-style data. The main training path is `src/hsi_model/training_script_fixed.py`; the secondary path is `src/hsi_model/train_optimized.py`; inference utilities are `smoke_infer.py`, `smoke_run.py`, and `utils/patch_inference.py`.

Critical paths audited:

- Data: `utils/data/loaders.py` -> `mst_dataset.py` / `hf_arad_dataset.py` -> `mst_to_gan_batch`.
- Model: `NoiseRobustCSWinModel` -> `NoiseRobustCSWinGenerator` + `SNTransformerDiscriminator`.
- Loss: `NoiseRobustLoss` with Charbonnier, SAM, perceptual, and Sinkhorn terms; `ComputeSinkhornDiscriminatorLoss`.
- Metrics: PSNR, SSIM, SAM, MRAE, RMSE, MAE through `utils/metrics.py`.
- Runtime: AMP/scalers, optimizers, schedulers, validation, checkpoint writing/loading.

## Risk Register

### BLOCKER: SAM Loss Had a Nonzero Optimum

Evidence: `src/hsi_model/constants.py:79` set `SAM_COSINE_CLAMP = 0.999`, and the previous `SAMLoss` clamped cosine similarity to that ceiling before `acos`. A synthetic identity probe returned `0.0447248` radians for `SAMLoss(x, x)` with zero gradient.

Why it matters: Identical spectra should have zero spectral angle. A positive floor biases the objective and can keep SAM contributing loss even at the true optimum.

Fix: Replaced `acos(clamp(cos, +/-0.999))` with an `atan2(||u - (u.v)v||, u.v)` formulation in `src/hsi_model/models/losses_consolidated.py:160`. Added `tests/test_losses.py:51`.

Impact/tradeoff: Correct zero optimum and finite gradients near identity. The formulation is mathematically equivalent for normalized spectra and more stable near `cos(theta)=1`.

### HIGH: Optimized Trainer Mixed Generator and Discriminator Objectives

Evidence: `train_optimized.py` used `NoiseRobustLoss` for the generator but BCE-with-logits for the discriminator, while the main trainer uses Sinkhorn on both sides.

Why it matters: The generator was trained against a Sinkhorn distributional objective while the discriminator optimized a binary GAN objective, so adversarial dynamics did not match the documented Sinkhorn-GAN method.

Fix: Added `ComputeSinkhornDiscriminatorLoss` to `train_optimized.py:71`, pass it through the training loop, and use it for D updates at `train_optimized.py:421`. Also passed `current_iteration` into loss calls and drove the generator iteration counter at `train_optimized.py:366`.

Impact/tradeoff: The optimized path now matches the intended Sinkhorn-GAN objective. BCE-specific behavior is removed from that path.

### HIGH: Nested Attention Ignored Config Fields

Evidence: Before patch, a config with `ckpt_min_tokens=1` and `use_fp16_bias=True` still produced CSWin blocks with `_ckpt_min_tokens=4096` and fp32 relative bias tables.

Why it matters: Memory-critical config fields were effectively dead in the generator, so users could believe they enabled checkpointing or fp16 bias while the model silently used defaults.

Fix: `DualTransformerBlock` now passes `config` to `EfficientSpectralAttention` and `CSWinAttentionBlock` in `src/hsi_model/models/generator_v3.py:193`. Added `tests/test_models.py:258`.

Impact/tradeoff: Config now affects nested attention. Setting `ckpt_min_tokens` too low can slow training because checkpoint recomputation is real.

### MED: CSWin Padding Injected Zero Borders

Evidence: README says symmetric reflect padding is used, but `CSWinAttentionBlock.forward` used default constant-zero `F.pad`.

Why it matters: Zero padding creates artificial dark borders in stripe attention and contradicts the documented behavior.

Fix: `src/hsi_model/models/attention.py:420` now uses reflect padding and falls back to replicate padding for tiny feature maps where reflect padding is invalid.

Impact/tradeoff: More faithful boundary behavior. Replicate fallback preserves tiny-input robustness.

### MED: Hugging Face Dataset Source Was Blocked by Local Path Validation

Evidence: `setup_paths` always required a local `data_dir`; a synthetic Hugging Face config with missing local path raised `FileNotFoundError`.

Why it matters: `dataset_source=huggingface` could not start from config alone, despite the loader supporting it.

Fix: `setup_paths` skips local data validation for Hugging Face sources and accepts validation directory aliases for MST layouts in `src/hsi_model/utils/training_setup.py:49`. Added `tests/test_training_setup.py:19`.

Impact/tradeoff: HF runs can initialize without local ARAD folders. MST runs still validate local layout strictly.

### MED: Checkpoint Loading Was Too Permissive and Missed Script Schema

Evidence: `load_checkpoint` only read `model_state_dict`, used `strict=False`, and did not recover `iter` / `best_mrae` from training-script checkpoints.

Why it matters: Partial model loads can silently run with random parameters, and checkpoints written by the training scripts were not fully resumable through the utility loader.

Fix: `load_checkpoint` now accepts `state_dict`, defaults to strict model matching, supports both optimizer/scaler key schemas, and returns `iteration` and `best_mrae` in `src/hsi_model/utils/checkpoint.py:126`. Added `tests/test_checkpoint.py:20`.

Impact/tradeoff: Incompatible checkpoints now fail loudly unless `strict=False` is explicitly requested.

## Residual Risks

- Real ARAD/MST data was not available in this environment, so split leakage, actual `.mat` orientation, and full validation metrics remain unverified on real scenes.
- GPU profiling was not available; all executed benchmarks were CPU-only.
- ERGAS is not currently implemented in `utils/metrics.py`. Add it if it is a primary reporting metric for your experiments.
- Full-resolution CSWin stripe attention is still expensive: horizontal/vertical stripe attention scales roughly with long-axis squared terms. Patch or tiled inference is still recommended for full ARAD images.

## Verification

Executed with `.venv-audit` Python 3.11 / PyTorch 2.11 CPU:

- `python -m py_compile ...` for changed modules and `smoke_run.py`: passed.
- `python -m pytest -q`: `64 passed, 2 skipped`.
- `python smoke_run.py`: `smoke_run_ok device=cpu train_seconds=0.405 peak_cuda_mb=0.0 disc_loss=-0.088785 gen_loss=1.196475 val_psnr=5.835 val_sam=55.430`.

Benchmark/profiling note:

- Pre-fix behavioral probe: `SAMLoss(x, x)=0.0447248 rad`, finite zero gradient.
- Post-fix test: `SAMLoss(x, x)=0.0`, finite gradient.
- Legacy-equivalent attention config on CPU, 1x3x16x16 generator forward+backward: `0.1319 s`, relative-bias storage `1.4 KiB`.
- Patched memory-stress config (`ckpt_min_tokens=1`, `use_fp16_bias=True`) on CPU, same input: `1.7559 s`, relative-bias storage `0.7 KiB`.
- Interpretation: fp16 bias halves bias-table memory, while aggressive checkpointing is slower on tiny CPU inputs. Keep `ckpt_min_tokens` high for small inputs; lower it only for large GPU memory pressure.
