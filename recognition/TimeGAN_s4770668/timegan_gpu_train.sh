#!/bin/bash
#SBATCH --job-name=timegan-train
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=timegan_train_%j.out
#SBATCH --error=timegan_train_%j.err

set -euo pipefail

source "$HOME/miniconda3/bin/activate"
conda activate torch

cd "$HOME/comp3710-timegan/PatternAnalysis-2026/recognition/TimeGAN_s4770668"

nvidia-smi

python -u - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable in this allocation.")

print("GPU:", torch.cuda.get_device_name(0))
x = torch.ones(2, 3, device="cuda")
print("GPU tensor check:", x.sum().item())
PY

python -u train_joint.py \
  --data-dir "$HOME/comp3710-timegan/data/raw" \
  --checkpoint checkpoints/supervisor_cpu_01/supervisor_best.pt \
  --epochs 10 \
  --device cuda \
  --run-name "joint_gpu_resource_${SLURM_JOB_ID}"
