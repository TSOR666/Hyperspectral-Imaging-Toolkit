#!/bin/sh
#
# LSF submission for the MSWR-Net v2.1.2 MST++-faithful baseline run.
#
# Trains FROM SCRATCH with MRAE-only loss, matching MST++'s recipe so the
# ARAD-1K MRAE result is a clean comparison against the MST++ baseline.
# See mswr_v2/configs/experiments/baseline_mstpp.yaml for the recipe details.
#
# Update the "EDIT ME" lines for your account and submit with:
#   bsub < mswr_v2/scripts/submit_mstpp_baseline.sh

#=======================
#     LSF DIRECTIVES
#=======================
#BSUB -q gpua100
#BSUB -J mswr_mstpp
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process:mps=no"
#BSUB -W 48:00
#BSUB -R "rusage[mem=32GB]"
# EDIT ME — email notifications
#BSUB -u your.email@example.com
#BSUB -B -N
#BSUB -o logs/mswr_mstpp_%J.out
#BSUB -e logs/mswr_mstpp_%J.err

#=======================
#     PATHS — EDIT ME
#=======================
REPO_DIR=/work3/paulgob/mswr_v2
DATA_ROOT=/work3/paulgob/dataset

#=======================
#     ENVIRONMENT
#=======================
module purge
module load cuda/11.8
module load cudnn/v8.6.0.163-prod-cuda-11.X

# EDIT ME — conda env path
source /zhome/XX/X/XXXXXX/miniconda3/etc/profile.d/conda.sh
conda activate mswr_v2

# Allocator tuning recommended by mswr_v2/README.md
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export OMP_NUM_THREADS=8
export NVIDIA_TF32_OVERRIDE=1
export TORCH_CUDNN_V8_API_ENABLED=1
# NOTE: do not set CUDA_LAUNCH_BLOCKING=1 — it serializes the GPU and tanks throughput.

cd "${REPO_DIR}"

#=======================
#     PRE-FLIGHT
#=======================
echo "=== Host: $(hostname)  ===  Time: $(date) ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

#=======================
#     TRAIN (from scratch — no --pretrained_model_path)
#=======================
echo "=== Launching MST++-faithful baseline ==="
python train_mswr_v212_logging.py \
  --config configs/experiments/baseline_mstpp.yaml \
  --data_root       "${DATA_ROOT}" \
  --log_base        "${REPO_DIR}/experiments/logs" \
  --checkpoint_base "${REPO_DIR}/experiments/checkpoints"

TRAIN_EXIT_CODE=$?

echo ""
echo "=== Train exit code: ${TRAIN_EXIT_CODE} ==="
echo "=== Job completed at $(date) ==="
exit ${TRAIN_EXIT_CODE}
