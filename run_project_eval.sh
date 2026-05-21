#!/bin/bash
#SBATCH --job-name=ER_project                # Job name
#SBATCH --output=logs/ER_project_%j.out      # Standard output file (%j will be replaced by the job ID)
#SBATCH --error=logs/ER_project_%j.err       # Standard error file
#SBATCH --nodes=1                                # Number of nodes requested
#SBATCH --ntasks=1                               # Number of tasks requested
#SBATCH --cpus-per-task=64                       # Number of CPU cores per task (adjust based on your parallel environment)
#SBATCH --mem=32G                                # Memory allocation
#SBATCH --time=00:20:00                          # Expected runtime (hours:minutes:seconds)
#SBATCH --partition=academic                     # Partition to submit to academic partition
#SBATCH --account=micro-515                      # Account name

export MUJOCO_GL=egl

mkdir -p logs

# Use env directly (robust)
# PYTHON=/home/serillon/miniconda3/envs/evorob/bin/python
source .venv/bin/activate

# echo "Python used:"
# $PYTHON -c "import sys; print(sys.executable)"

# # Optional debug
# $PYTHON -c "import PIL; print(PIL.__version__)"

# Run
$PYTHON final_project_test.py --best_dir_path results/2026-05-21_16-13-44_final_test