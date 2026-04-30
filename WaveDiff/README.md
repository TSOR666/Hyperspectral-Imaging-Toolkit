# Latent Diffusion Models for Hyperspectral Image Reconstruction

This repository hosts the research code for reconstructing hyperspectral images (HSI) from RGB observations with latent diffusion models enhanced by multi-scale wavelet processing. The implementation builds neural architectures for forward diffusion, reverse denoising, and adaptive spectral refinement that target high-fidelity, physically consistent reconstructions on benchmarks such as ARAD-1K.

## Key Features
- **Latent diffusion backbones** tailored to predict 31-band hyperspectral cubes from natural RGB inputs.
- **Wavelet-enhanced denoisers** including adaptive thresholding modules for frequency-aware noise suppression.
- **Optimal-transport diffusion sampling (DPM-OT)** with cached schedule statistics for numerically stable generation.
- **Comprehensive utilities** for spectral loss computation, curriculum masking, visualization, and metric reporting.

## Repository Layout
```
WaveDiff/
├── configs/               # Example JSON configuration files
├── diffusion/             # Diffusion schedules, samplers, and helper utilities
├── losses/                # Spectral, spatial, and wavelet-domain loss functions
├── models/                # Model definitions wrapping diffusion + refinement modules
├── modules/               # Building blocks (encoders, denoisers, attention, etc.)
├── transforms/            # Wavelet transforms and adaptive thresholding operators
├── utils/                 # Masking, metrics, logging, visualization helpers
├── train.py               # CLI entry-point for supervised training
├── inference.py           # CLI entry-point for reconstruction/evaluation
├── demo.py                # Minimal inference demo for a single RGB input
├── QUICK_START.md         # Step-by-step setup walkthrough
└── README.md              # Model-architecture focused documentation
```

The top-level `README.md` (this file) gives a project overview. Module-level details live in the subdirectory documentation.

## Architecture Summary

- Backbones: Latent diffusion models specialized for 31‑band HSI reconstruction.
- Wavelets: Standard, learnable, and adaptive wavelet transforms for multi‑scale spectral structure; optional adaptive thresholding.
- Refinement: Spectral refinement head and optional pixel‑space refinement to reduce artifacts after denoising.
- Schedules: `diffusion/noise_schedule.py` with DPM‑OT samplers (`diffusion/dpm_ot.py`, `enhanced_dpm_ot.py`).

See `WaveDiff/README.md` for module‑level details (encoders, decoders, denoisers, attention).

## Installation
1. Clone the repository and move into the project root:
   ```bash
   git clone https://github.com/yourusername/hsi-wavelet-diffusion.git
   cd hsi-wavelet-diffusion/WaveDiff
   ```
2. (Optional) Create and activate a Python environment (Python ≥ 3.9 recommended).
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Preparing Data
Training and evaluation expect paired RGB/HSI samples arranged as:
```
<data_root>/
├── train/
│   ├── RGB/    # 8-bit RGB images (png/jpg)
│   └── HSI/    # Corresponding hyperspectral cubes (.npy or .mat)
├── val/
│   ├── RGB/
│   └── HSI/
└── test/
    ├── RGB/
    └── HSI/
```
Each RGB filename must match its hyperspectral counterpart. Datasets such as [ARAD-1K](https://www.cs.ubc.ca/labs/imager/tr/arad1k/) follow this convention.

## Training
Launch supervised training with the provided CLI:
```bash
cd WaveDiff
python train.py \
    --model_type adaptive_wavelet \
    --train_dir data/ARAD1K/train \
    --val_dir data/ARAD1K/val \
    --batch_size 8 \
    --num_epochs 100
```
All CLI arguments are documented in `python train.py --help`. For reproducible experiments you can also pass a JSON config file:
```bash
python train.py --config configs/example_config.json
```

Checkpoints and visualizations are timestamped and saved under `checkpoints/` and `visualizations/` respectively.

### Configuration (configs/example_config.json)

- Core: `model_type` (base/wavelet/adaptive_wavelet), `latent_dim`, `timesteps`, `image_size`.
- Optim: `batch_size`, `learning_rate`, `min_lr`, `weight_decay`, `max_grad_norm`, `num_epochs`, `num_workers`.
- Loss: `diffusion_loss_weight`, `l1_loss_weight`, `cycle_loss_weight`, `wavelet_loss_weight`.
- Data: `train_dir`, `val_dir`, `resume_from_checkpoint`, `checkpoint_dir`, `visualization_dir`.
- Masking/Curriculum: `use_masking`, `mask_strategy`, `curriculum_strategies`, `initial_mask_ratio`, `final_mask_ratio`.
- Thresholding: `threshold_method`, `init_threshold`, `trainable_threshold`.

## Inference & Evaluation
Reconstruct hyperspectral cubes for a single RGB image:
```bash
cd WaveDiff
python inference.py \
    --checkpoint checkpoints/<run_id>/final_model.pt \
    --image path/to/rgb.png \
    --output_dir results/
```

To process an entire directory and compute spectral metrics when ground truth is available:
```bash
python inference.py \
    --checkpoint checkpoints/<run_id>/final_model.pt \
    --input_dir data/ARAD1K/test/RGB \
    --output_dir results/
```
The script automatically looks for matching HSI cubes next to each RGB file and produces false-color visualizations, spectral plots, and quantitative reports whenever ground truth data is available.

## Additional Resources
- `WaveDiff/README.md` documents the architectural components and includes qualitative results.
- `WaveDiff/QUICK_START.md` provides a fast setup checklist for new users.
- Example notebooks and scripts can be extended to integrate new datasets or sampling schedules.

## Citation
If this repository is useful in your research, please cite it appropriately:
```
@software{hsi_wavelet_ldm,
  title        = {Latent Diffusion Models for Hyperspectral Image Reconstruction},
  author       = {Thierry Silvio Claude Soreze},
  year         = {2024},
  url          = {https://github.com/TSOR666/hsi-wavelet-diffusion}
}
```

## License
This project is released under the MIT License. See `LICENSE` for details.
