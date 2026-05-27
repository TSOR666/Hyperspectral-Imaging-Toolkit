#!/bin/sh
#
# LSF submission for the MSWR-Net v2.1.2 MRAE-focused warm-restart fine-tune.
#
# Submits the recipe in configs/mswr_finetune_mrae.yaml against the stripped
# best checkpoint. Before submitting, run scripts/strip_checkpoint.py once to
# produce the weights-only file referenced below.
#
# Update the four "EDIT ME" lines for your account/paths and submit with:
#   bsub < mswr_v2/scripts/submit_finetune_mrae.sh

#=======================
#     LSF DIRECTIVES
#=======================
#BSUB -q gpua100
#BSUB -J mswr_ft_mrae
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process:mps=no"
#BSUB -W 12:00
#BSUB -R "rusage[mem=32GB]"
# EDIT ME — email notifications
#BSUB -u your.email@example.com
#BSUB -B -N
#BSUB -o logs/mswr_ft_mrae_%J.out
#BSUB -e logs/mswr_ft_mrae_%J.err

#=======================
#     PATHS — EDIT ME
#=======================
# Repo path on the cluster
REPO_DIR=/work3/paulgob/mswr_v2
# Dataset root with Train_RGB / Train_Spec / split_txt / Valid_*
DATA_ROOT=/work3/paulgob/dataset
# Source best_model.pth from the previous run
SRC_CKPT=/work3/paulgob/mswr_v2_first/experiments/checkpoints/2026-05-22_04-54-38-524507/best_model.pth
# Output for the stripped weights-only checkpoint
STRIPPED_CKPT=${REPO_DIR}/experiments/checkpoints/best_model_weights_only.pth

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

if [ ! -f "${SRC_CKPT}" ]; then
  echo "ERROR: SRC_CKPT not found: ${SRC_CKPT}" >&2
  exit 2
fi

mkdir -p "$(dirname "${STRIPPED_CKPT}")"

#=======================
#     STRIP CHECKPOINT
#=======================
# Idempotent: only strips if the stripped file is missing or older than the source.
if [ ! -f "${STRIPPED_CKPT}" ] || [ "${SRC_CKPT}" -nt "${STRIPPED_CKPT}" ]; then
  echo "=== Stripping checkpoint to weights only ==="
  python scripts/strip_checkpoint.py \
    --input  "${SRC_CKPT}" \
    --output "${STRIPPED_CKPT}"
else
  echo "=== Stripped checkpoint up-to-date: ${STRIPPED_CKPT} ==="
fi

#=======================
#     TRAIN
#=======================
echo "=== Launching fine-tune ==="
python train_mswr_v212_logging.py \
  --config configs/mswr_finetune_mrae.yaml \
  --data_root       "${DATA_ROOT}" \
  --log_base        "${REPO_DIR}/experiments/logs" \
  --checkpoint_base "${REPO_DIR}/experiments/checkpoints" \
  --pretrained_model_path "${STRIPPED_CKPT}"

TRAIN_EXIT_CODE=$?

echo ""
echo "=== Train exit code: ${TRAIN_EXIT_CODE} ==="
echo "=== Job completed at $(date) ==="
exit ${TRAIN_EXIT_CODE}
