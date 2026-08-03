# CSWIN v2 - Quick Start Guide

**Get up and running in 10 minutes!** ⚡

---

## Prerequisites

- Python 3.10–3.12 (see `pyproject.toml`)
- NVIDIA GPU with CUDA support (recommended, CPU also works)
- ARAD-1K dataset (or compatible HSI dataset)

---

## Setup (One-Time)

### 1. Create Virtual Environment (2 minutes)

```bash
cd "CSWIN v2"
python -m venv .venv

# Activate the environment
source .venv/bin/activate  # On Linux/Mac
# OR
.venv\Scripts\activate     # On Windows
```

### 2. Install PyTorch (3 minutes)

**For CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CPU only:**
```bash
pip install torch torchvision torchaudio
```

### 3. Install Other Dependencies (2 minutes)

```bash
pip install -r requirements.txt
```

### 4. Configure Dataset Path (1 minute)

**Option A - Environment Variable:**
```bash
export HSI_DATA_DIR=/path/to/your/ARAD_1K
```

**Option B - CLI Override (no setup needed):**
```bash
# Will specify path when running (see below)
```

---

## Run Training (Immediate)

### Basic Training
```bash
python src/hsi_model/train_generator.py \
    --config-name config \
    data_dir=/path/to/ARAD_1K
```

### Fresh Validated Recovery Training
```bash
python src/hsi_model/train_generator.py \
    --config-name sota_cascade \
    data_dir=/path/to/ARAD_1K
```

### Legacy Sinkhorn-GAN Training
```bash
python src/hsi_model/training_script_fixed.py \
    --config-name config \
    data_dir=/path/to/ARAD_1K
```

### Multi-GPU Training
```bash
python -m torch.distributed.run --nproc_per_node=4 \
    src/hsi_model/train_generator.py \
    --config-name config \
    data_dir=/path/to/ARAD_1K
```

---

## Customize Training

Override any parameter from the config:

```bash
python src/hsi_model/train_generator.py \
    data_dir=/path/to/ARAD_1K \
    batch_size=16 \
    epochs=500 \
    generator_lr=1e-4 \
    objective=mrae_annealed \
    mixed_precision=true
```

---

## Monitor Training

### Logs
```bash
# Default location: ./artifacts/logs/
tail -f artifacts/logs/training.log
```

### Checkpoints
```bash
# Default location: ./artifacts/checkpoints/
ls -lh artifacts/checkpoints/
```

### Metrics
The MetricsLogger saves CSV files with:
- PSNR, SSIM, SAM per epoch
- Training loss curves
- Validation metrics

---

## Common Issues

### "ModuleNotFoundError: No module named 'torch'"
**Solution:** Install PyTorch first (step 2 above)

### "FileNotFoundError: [Errno 2] No such file or directory: './data/ARAD_1K'"
**Solution:** Set correct dataset path:
```bash
export HSI_DATA_DIR=/your/actual/path/to/ARAD_1K
# OR
python ... data_dir=/your/actual/path/to/ARAD_1K
```

### "CUDA out of memory"
**Solutions:**
1. The generator trainer first retries the same samples with automatic
   gradient microbatches (20 -> 10 -> 5 -> 3 -> 2 -> 1), preserving the
   configured effective batch size. Look for the selected `microbatch=` in
   the training log.
2. If microbatch 1 still fails, reduce batch size: `batch_size=8`
3. Reduce the progressive-stage batch sizes or patch size.
4. Set `memory_mode=lazy` to trade loader throughput for lower host RAM

### "Padding error at dimension 3"
**Solution:** This is already fixed in the code. Ensure you're using the latest version.

---

## Key Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 20 | Optimizer-step batch size for training |
| `val_batch_size` | 1 | Batch size for validation |
| `patch_size` | 128 | Size of image patches |
| `epochs` | 300 | Number of single-stage training epochs |
| `generator_lr` | 0.0004 | Generator learning rate |
| `optimizer` | `adam` | MST++-aligned generator optimizer |
| `weight_decay` | 0.0 | No weight decay |
| `objective` | `mrae_annealed` | Pure MRAE with an annealed, stabilized training denominator |
| `mrae_epsilon_start` | 1e-2 | Initial denominator floor for stable early optimization |
| `mrae_epsilon_end` | 1e-3 | Final floor for the base 300k run; validation still reports exact MRAE |
| `mrae_epsilon_anneal_iters` | 50000 | Updates used to decay the floor |
| `mixed_precision` | true | Use automatic mixed precision |
| `validation_clamp_output` | true | Match NTIRE `[0,1]` scoring |
| `excluded_scene_stems` | `[ARAD_1K_0314]` | Known-corrupt scenes omitted intentionally |
| `cswin_attention_mode` | `local_global` | Validated spatial attention; keep `cswin` and `axial` as controlled ablations |

See `src/configs/config.yaml` for all parameters.

---

## Verify Installation

Test that everything is working:

```bash
python -c "
import sys
sys.path.insert(0, 'src')
from hsi_model.models import NoiseRobustCSWinGenerator
print('✅ Installation verified!')
"
```

---

## Next Steps

1. **Review Configuration:** Check `src/configs/config.yaml`
2. **Read Full README:** See `README.md` for detailed documentation
3. **Check Architecture:** Review model details in README
4. **Monitor Training:** Use logs to track progress
5. **Adjust Hyperparameters:** Tune based on your dataset

---

## Performance Tips

### For Faster Training:
- Use `mixed_precision=true` (enabled by default)
- Increase `batch_size` if GPU memory allows
- Use `num_workers=4` or more for data loading

### For Lower Memory:
- Reduce `batch_size=8`
- Reduce batch sizes in `progressive_stages`
- Set `memory_mode=lazy` in config for file-backed MST data with bounded per-worker caches.
- Tune `lazy_cache_size` to trade RAM for random-access speed. Keep `memory_mode=standard` when throughput matters more than resident memory.

### For Better Quality:
- Use the 70k-step residual-balanced `sota_cascade` recipe. The June control
  reached MRAE 0.2737 / PSNR 29.56 near the end of a 70k cosine schedule; the
  later 300k schedule regressed after its midpoint while retaining a high LR.
- Start a fresh run after changing the objective to `mrae_annealed`
- Use `--config-name sota_cascade` only for a fresh run. It now pins the
  local/global single-pass architecture, balances the SSTB outer residuals,
  uses a normally initialized output head plus an RGB spectral prior, disables
  legacy radiometry-stripping
  normalization/denoising paths, and writes to separate recovery directories.
  Never resume either failed July 2026 checkpoint bundle.
- For the successful 2026-07-25 stable-init run, stop at the saved 164k best
  checkpoint rather than the degraded 185k latest checkpoint. Start the
  exact-objective phase with the `finetune_128_exact_mrae` config and set
  `finetune_checkpoint=/path/to/best_model.pth`.
- For the 2026-07-27 radiometric run, stop at the saved 115k best checkpoint
  (MRAE 0.524619), not the 215k state (0.577060), and use
  `--config-name finetune_radiometric_exact_mrae` with
  `finetune_checkpoint=/path/to/best_model.pth`.
- Treat 256/512 progressive fine-tuning as experimental unless a patch-size
  sweep shows it improves the deployed inference path
- Track both deployed `mrae` and diagnostic `raw_mrae`

---

## Troubleshooting

**Get detailed logs:**
```bash
HYDRA_FULL_ERROR=1 python src/hsi_model/train_generator.py ...
```

**Check GPU usage:**
```bash
watch -n 1 nvidia-smi
```

**Validate dataset:**
```python
from hsi_model.utils.data import show_dataloader_diagnostics
# Run this to check dataset loading
```

---

## Example Training Session

```bash
# 1. Set environment
export HSI_DATA_DIR=/datasets/ARAD_1K
export HSI_LOG_DIR=./experiments/run1/logs
export HSI_CKPT_DIR=./experiments/run1/checkpoints

# 2. Run training with custom params
python src/hsi_model/train_generator.py \
    --config-name config \
    batch_size=16 \
    epochs=300 \
    generator_lr=4e-4

# 3. Monitor (in another terminal)
tail -f experiments/run1/logs/training.log

# 4. Check checkpoints
ls -lh experiments/run1/checkpoints/

# Training will auto-save checkpoints and can be resumed if interrupted
```

---

## Help & Support

- **Full Documentation:** See `README.md`
- **Configs & Ablations:** See `src/configs/*.yaml` (headers document each recipe)
- **Smoke checks:** `smoke_run.py`, `smoke_train.py`, `smoke_infer.py`

---

**Ready to train!**

Just run:
```bash
python src/hsi_model/train_generator.py \
    --config-name sota_cascade \
    data_dir=/path/to/ARAD_1K
```
