# HSI Wavelet Diffusion - Quick Start Guide

This guide will help you get started with the Hyperspectral Image (HSI) Wavelet Diffusion model for RGB to HSI reconstruction.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/hsi-wavelet-diffusion.git
   cd hsi-wavelet-diffusion
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

The model expects data in the following structure:
```
data/
  ├── ARAD1K/
      ├── train/
      │   ├── RGB/
      │   │   ├── img1.png
      │   │   ├── img2.png
      │   │   └── ...
      │   └── HSI/
      │       ├── img1.mat
      │       ├── img2.mat
      │       └── ...
      ├── val/
      │   ├── RGB/
      │   │   └── ...
      │   └── HSI/
      │       └── ...
      └── test/
          ├── RGB/
          │   └── ...
          └── HSI/
              └── ...
```

Each RGB image should have a corresponding HSI file with the same base filename.

## Training

To train a model, use the `train.py` script:

```bash
python train.py --model_type wavelet --train_dir data/ARAD1K/train --val_dir data/ARAD1K/val --batch_size 8 --num_epochs 100
```

Available model types:
- `base`: Standard Latent Diffusion Model
- `wavelet`: Wavelet-enhanced Latent Diffusion Model
- `adaptive_wavelet`: Adaptive Wavelet Thresholding Latent Diffusion Model

## Inference

To run inference on a single image:

```bash
python inference.py --checkpoint checkpoints/best_model.pt --image path/to/rgb_image.png --output_dir results
```

To process a directory of images:

```bash
python inference.py --checkpoint checkpoints/best_model.pt --input_dir path/to/rgb_images --output_dir results
```

For adaptive threshold models:

```bash
python inference.py --checkpoint checkpoints/adaptive_model.pt --image path/to/rgb_image.png --output_dir results --sampling_steps 20
```

Use `--no_adaptive_threshold` to disable adaptive processing when evaluating baseline behavior.

## Evaluating Results

To evaluate the model performance against ground truth HSI:

```bash
python inference.py --checkpoint checkpoints/best_model.pt --input_dir path/to/test/RGB --output_dir evaluation_results
```

The script will automatically look for ground truth HSI data in a sibling `HSI` directory.

## Example Configuration

You can also use a JSON configuration file for training:

```bash
python train.py --config configs/example_config.json
```

See `configs/example_config.json` for an example configuration file.

## Model Selection Guide

- **Base Model**: Good general performance, fastest training time
- **Wavelet Model**: Better detail preservation and multi-scale feature extraction
- **Adaptive Wavelet Model**: Best performance for noisy inputs, adaptive to different frequency components

## Visualizing Results

The inference script automatically generates visualizations including:
- False color RGB representations
- Band visualizations
- Spectral signature comparisons (when ground truth is available)
- Error maps (when ground truth is available)

