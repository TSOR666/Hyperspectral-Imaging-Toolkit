# CSWIN v2 - Quick Start Guide

**Get up and running in 10 minutes!** ⚡

---

## Prerequisites

- Python 3.8 or newer
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
python src/hsi_model/training_script_fixed.py \
    --config-name config \
    data_dir=/path/to/ARAD_1K
```

### Memory-Optimized Training
```bash
python src/hsi_model/train_optimized.py \
    --config-name config \
    data_dir=/path/to/ARAD_1K
```

### Multi-GPU Training
```bash
python -m torch.distributed.run --nproc_per_node=4 \
    src/hsi_model/training_script_fixed.py \
    --config-name config \
    data_dir=/path/to/ARAD_1K
```

---

## Customize Training

Override any parameter from the config:

```bash
python src/hsi_model/training_script_fixed.py \
    data_dir=/path/to/ARAD_1K \
    batch_size=16 \
    epochs=500 \
    optimizer.generator_lr=1e-4 \
    optimizer.discriminator_lr=5e-5 \
    lambda_adversarial=0.2 \
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
1. Reduce batch size: `batch_size=8`
2. Use gradient accumulation: `gradient_accumulation_steps=4`
3. Use memory-optimized trainer: `train_optimized.py`

### "Padding error at dimension 3"
**Solution:** This is already fixed in the code. Ensure you're using the latest version.

---

## Key Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 20 | Batch size for training |
| `val_batch_size` | 1 | Batch size for validation |
| `patch_size` | 128 | Size of image patches |
| `epochs` | 300 | Number of training epochs |
| `generator_lr` | 0.0002 | Generator learning rate |
| `discriminator_lr` | 0.00005 | Discriminator learning rate |
| `lambda_rec` | 1.0 | Reconstruction loss weight |
| `lambda_adversarial` | 0.1 | Adversarial loss weight |
| `lambda_perceptual` | 0.1 | Perceptual loss weight |
| `lambda_sam` | 0.05 | SAM loss weight |
| `mixed_precision` | true | Use automatic mixed precision |
| `n_critic` | 1 | Discriminator steps per generator step |

See `src/configs/config.yaml` for all parameters.

---

## Verify Installation

Test that everything is working:

```bash
python -c "
import sys
sys.path.insert(0, 'src')
from hsi_model.models import NoiseRobustCSWinModel
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
- Enable `gradient_accumulation_steps=2` for larger effective batch size

### For Lower Memory:
- Use `train_optimized.py` instead of `training_script_fixed.py`
- Reduce `batch_size=8`
- Increase `gradient_accumulation_steps=4`
- Set `memory_mode=lazy` in config

### For Better Quality:
- Increase `epochs=500`
- Tune loss weights (`lambda_*`)
- Experiment with `sinkhorn_epsilon` and `sinkhorn_iters`
- Enable `use_r1_regularization=true`

---

## Troubleshooting

**Get detailed logs:**
```bash
HYDRA_FULL_ERROR=1 python src/hsi_model/training_script_fixed.py ...
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
python src/hsi_model/training_script_fixed.py \
    --config-name config \
    batch_size=16 \
    epochs=500 \
    optimizer.generator_lr=1e-4 \
    lambda_adversarial=0.15

# 3. Monitor (in another terminal)
tail -f experiments/run1/logs/training.log

# 4. Check checkpoints
ls -lh experiments/run1/checkpoints/

# Training will auto-save checkpoints and can be resumed if interrupted
```

---

## Help & Support

- **Full Documentation:** See `README.md`
- **Verification Report:** See `CSWIN_VERIFICATION_REPORT.md`
- **Model Audit:** See `MODEL_AUDIT_REPORT.md` (in repo root)
- **Bug Fixes:** See `FIXES_APPLIED.md` (in repo root)

---

**Ready to train!** 🚀

Just run:
```bash
python src/hsi_model/training_script_fixed.py \
    --config-name config \
    data_dir=/path/to/ARAD_1K
```
