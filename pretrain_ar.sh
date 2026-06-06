#!/bin/bash
#SBATCH --job-name=transformer_pretrain
#SBATCH -A [YOUR_ACCOUNT_ALLOCATION]
#SBATCH --qos=embers                      # Standard production high-priority queue on Phoenix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12                 # Requests the 12-core parallel processing sweet spot
#SBATCH --gres=gpu:A100:1                  # Explicitly requests 1x NVIDIA A100 GPU node
#SBATCH --mem=64G                          # Allocates 64GB of system RAM memory
#SBATCH --time=01:00:00                     # Sets a 3-hour wall-time safety boundary
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=vyadav68@gatech.edu

module load anaconda3
conda activate polymers

# Ensure log directory exists
mkdir -p logs checkpoints/pretrain

# Force all temporary file operations to build inside high-throughput scratch space
if [ -n "$SCRM" ]; then
    export SCRATCH_DIR="$SCRM"
elif [ -n "$SCRATCH" ]; then
    export SCRATCH_DIR="$SCRATCH"
fi

# Asynchronous GPU kernel optimization 
export CUDA_LAUNCH_BLOCKING=0
unset TORCH_CUDNN_BENCHMARK                # Stripped to stabilize text-generation layers

# Core Thread-Splitting Strategy:
# Grants your live GFN2-xTB baseline loops 6 parallel threads for geometry optimizations,
# leaving 6 cores completely open for causal transformer tracking and MCTS graph management.
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6

cd $SLURM_SUBMIT_DIR
export HF_HOME="/storage/scratch1/2/vyadav68/.hf_cache"

python pretrain_ar.py
