# src/hsi_model/constants.py
"""
Global constants for the HSI reconstruction project.

Centralizes magic numbers and configuration defaults to improve maintainability.
"""

# ============================================
# ARAD-1K Dataset Constants
# ============================================
ARAD1K_CROP_HEIGHT = 226  # MST++ evaluation protocol
ARAD1K_CROP_WIDTH = 256   # MST++ evaluation protocol
ARAD1K_FULL_HEIGHT = 482  # Typical full image height
ARAD1K_FULL_WIDTH = 512   # Typical full image width
ARAD1K_NUM_BANDS = 31     # Hyperspectral bands
ARAD1K_RGB_CHANNELS = 3   # RGB channels

# ============================================
# Model Architecture Constants
# ============================================
DEFAULT_BASE_CHANNELS = 64
DEFAULT_NUM_HEADS = 4
DEFAULT_SPLIT_SIZE = 7
DEFAULT_MLP_RATIO = 2.0
DEFAULT_NORM_GROUPS = 8

# Generator
GENERATOR_CLAMP_RANGE = 10.0
GENERATOR_RESIDUAL_SCALE = 0.1

# Discriminator
DISCRIMINATOR_BASE_DIM = 64
DISCRIMINATOR_OUTPUT_CLAMP = 10.0
DISCRIMINATOR_ATTENTION_CLAMP = 50.0

# ============================================
# Training Constants
# ============================================
DEFAULT_BATCH_SIZE = 20
DEFAULT_PATCH_SIZE = 128
DEFAULT_STRIDE = 8
DEFAULT_NUM_WORKERS = 2
DEFAULT_WARMUP_STEPS = 2000
DEFAULT_GRADIENT_CLIP_NORM = 1.0

# Learning rates
DEFAULT_GENERATOR_LR = 2e-4
DEFAULT_DISCRIMINATOR_LR = 5e-5

# Loss weights
DEFAULT_LAMBDA_REC = 1.0
DEFAULT_LAMBDA_PERCEPTUAL = 0.1
DEFAULT_LAMBDA_ADVERSARIAL = 0.1
DEFAULT_LAMBDA_SAM = 0.05

# ============================================
# Sinkhorn Algorithm Constants
# ============================================
DEFAULT_SINKHORN_EPSILON = 0.1
DEFAULT_SINKHORN_ITERATIONS = 50
SINKHORN_EPS_STABILITY = 1e-9

# ============================================
# Regularization Constants
# ============================================
DEFAULT_R1_GAMMA = 10.0
R1_APPLY_FREQUENCY = 16  # Apply R1 every N iterations

# ============================================
# Numerical Stability Constants
# ============================================
EPSILON_SMALL = 1e-12  # For normalizations
EPSILON_MEDIUM = 1e-8  # For divisions
EPSILON_LARGE = 1e-6   # For SAM calculations
CHARBONNIER_EPSILON = 1e-3

# Attention stability
ATTENTION_TEMPERATURE_INIT = 1.0
ATTENTION_CLAMP_VALUE = 50.0

# ============================================
# Memory Management Constants
# ============================================
H5PY_CACHE_SIZE_MB = 4  # Reduced from default 64MB
DEFAULT_CACHE_SIZE_GB = 4.0
OMP_NUM_THREADS = 2

# ============================================
# Validation/Testing Constants
# ============================================
VALIDATION_CENTER_CROP_START_H = 128  # MST++ protocol
VALIDATION_CENTER_CROP_START_W = 128  # MST++ protocol
VALIDATION_CENTER_CROP_END_H = -128
VALIDATION_CENTER_CROP_END_W = -128

MAX_LOSS_VALUE = 100.0  # Safety clamp

# ============================================
# Color Conversion Constants
# ============================================
WAVELENGTH_MIN = 400  # nm
WAVELENGTH_MAX = 700  # nm

# CIE 1931 color matching function parameters
CMF_RED_PARAMS = (599.8, 33.0, 0.264)    # peak, width, amplitude
CMF_GREEN_PARAMS = (549.1, 57.0, 0.323)
CMF_BLUE_PARAMS = (445.8, 33.0, 0.272)

# ============================================
# Logging Constants
# ============================================
LOG_EVERY_N_ITERATIONS = 20
VALIDATE_EVERY_N_ITERATIONS = 1000
SAVE_CHECKPOINT_EVERY_N_ITERATIONS = 5000

# ============================================
# File System Constants
# ============================================
DEFAULT_CHECKPOINT_DIR = "./artifacts/checkpoints"
DEFAULT_LOG_DIR = "./artifacts/logs"
DEFAULT_DATA_DIR = "./data/ARAD_1K"

CHECKPOINT_BEST_NAME = "best_model.pth"
CHECKPOINT_LATEST_NAME = "latest_checkpoint.pth"
CHECKPOINT_KEEP_COUNT = 5  # Number of recent checkpoints to keep

# ============================================
# Device Constants
# ============================================
PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True,max_split_size_mb:128"
