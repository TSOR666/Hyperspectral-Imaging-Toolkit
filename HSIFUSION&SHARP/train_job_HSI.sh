#!/bin/sh

#=======================
#     LSF DIRECTIVES - A100 OPTIMIZED
#=======================

# A100 GPU queue (updated for better hardware)
#BSUB -q gpua100

# Job name with timestamp for tracking
#BSUB -J mstpp_optimized

# CPU cores optimized for A100 workflow
#BSUB -n 8

# All cores on the same machine for optimal memory bandwidth
#BSUB -R "span[hosts=1]"

# 1 A100 GPU in exclusive mode for maximum performance
#BSUB -gpu "num=1:mode=exclusive_process:mps=no"

# Extended time limit for 300 epochs with larger batch size
#BSUB -W 72:00

# Increased RAM for MST++ with A100 optimizations (batch_size=20)
#BSUB -R "rusage[mem=32GB]"

# Email notifications
#BSUB -u etienne.rozoy@gmail.com
#BSUB -B -N

# Logs with timestamp
#BSUB -o logs/HSIFUSION_log_%J.out
#BSUB -e logs/HSIFUSION_log_%J.err

#=======================
#     ENVIRONMENT SETUP
#=======================

# Clear any existing modules
module purge

# Load optimized CUDA and cuDNN for A100
module load cuda/11.8
module load cudnn/v8.6.0.163-prod-cuda-11.X

# Activate Python environment
source /zhome/25/d/221980/miniconda3/etc/profile.d/conda.sh
conda activate hsifusion
# Set environment variables for optimal A100 performance
export CUDA_LAUNCH_BLOCKING=1
export TORCH_CUDNN_V8_API_ENABLED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_USE_CUDA_DSA=1

# Enable tensor core optimizations
export NVIDIA_TF32_OVERRIDE=1

# Set memory allocation strategy
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Enable optimized data loading
export OMP_NUM_THREADS=8

# Display full error traces
export HYDRA_FULL_ERROR=1

#=======================
#     PRE-TRAINING CHECKS
#=======================

echo "=== System Information ==="
nvidia-smi
echo ""
echo "=== GPU Memory Before Training ==="
nvidia-smi --query-gpu=memory.total,memory.free,memory.used --format=csv
echo ""
echo "=== Environment Variables ==="
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo ""

 

#=======================
#     OPTIMIZED TRAINING
#=======================

echo "=== Starting HSIFusion Training ==="
echo "Timestamp: $(date)"
echo "Using HSIFusion Lightning Pro configuration"
echo ""

# Run training with optimized configuration
  

python hsifusion_training.py --model_size base --data_root ./dataset --output_dir ./experiments/hsifusion

#python test_setup_script.py
#python apply_all_fixes.py
#python optimized_train_script.py --model hsifusion --model_size base


# Capture exit code
TRAIN_EXIT_CODE=$?

#=======================
#     POST-TRAINING ANALYSIS
#=======================

echo ""
echo "=== Training Completed ==="
echo "Exit code: $TRAIN_EXIT_CODE"
echo "Timestamp: $(date)"
echo ""

echo "=== GPU Memory After Training ==="
nvidia-smi --query-gpu=memory.total,memory.free,memory.used --format=csv
echo ""

echo "=== Final GPU Status ==="
nvidia-smi
echo ""

# Check if training was successful
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "=== Training completed successfully! ==="
    
    # Run quick validation on best model if available
    if [ -f "checkpoints/arad1k/best.pth" ]; then
        echo "=== Running validation on best model ==="
        python robust_test.py \
            --checkpoint checkpoints/arad1k/best.pth \
            --data_dir ./ARAD_1K_Efficient \
            --output_dir ./validation_results \
            --max_samples 10 \
            --visualize
    fi
    
else
    echo "=== Training failed with exit code $TRAIN_EXIT_CODE ==="
    
    # Save debug information
    echo "=== Saving debug information ==="
    if [ -f "checkpoints/arad1k/emergency.pth" ]; then
        echo "Emergency checkpoint saved"
    fi
    
    # Show last few lines of log for quick diagnosis
    echo "=== Last 20 lines of training log ==="
    if [ -f "logs/training.log" ]; then
        tail -20 logs/training.log
    fi
fi

echo ""
echo "=== Job completed at $(date) ==="

