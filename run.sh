#!/usr/bin/env bash
# Run the OpenES experiment on four Brax environments with Hydra multirun.
# Set CUDA_VISIBLE_DEVICES to choose GPUs and N_JOBS to limit concurrent jobs.
# Additional arguments are forwarded to Hydra, for example seed=42,43.
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source .venv/bin/activate

exec python train.py --multirun \
  hydra/launcher=joblib \
  "hydra.launcher.n_jobs=${N_JOBS:-4}" \
  agent=openes \
  env=brax/ant,brax/hopper,brax/walker2d \
  "$@"
