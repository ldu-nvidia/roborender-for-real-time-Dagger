#!/bin/bash
#SBATCH --partition=viscam
#SBATCH --account=viscam
#SBATCH --job-name=mkprev
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/svl/u/ravenh/fast_roborender/make_previews_%j.log
set -e
export HOME=/sailhome/ravenh
cd /svl/u/ravenh/fast_roborender
exec roborender-acceleration/.venv/bin/python make_previews.py
