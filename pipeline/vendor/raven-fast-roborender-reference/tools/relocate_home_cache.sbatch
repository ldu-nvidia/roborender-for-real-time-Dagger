#!/bin/bash
#SBATCH --job-name=relocate_cache
#SBATCH --account=viscam
#SBATCH --partition=sc-loprio
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --output=/svl/u/ravenh/fast_roborender/relocate_cache_%j.log
set -euo pipefail
DEST=/vision/u/ravenh/home_cache
mkdir -p "$DEST"
for NAME in openpi huggingface; do
  SRC=/sailhome/ravenh/.cache/$NAME
  [ -L "$SRC" ] && { echo "$NAME already relocated"; continue; }
  [ -d "$SRC" ] || { echo "$NAME missing"; continue; }
  echo "copying $NAME ..."
  rsync -a --info=progress2 "$SRC/" "$DEST/$NAME/"
  SRC_N=$(find "$SRC" -type f | wc -l); DST_N=$(find "$DEST/$NAME" -type f | wc -l)
  echo "$NAME files: src=$SRC_N dst=$DST_N"
  if [ "$SRC_N" -eq "$DST_N" ]; then
    rm -rf "$SRC" && ln -s "$DEST/$NAME" "$SRC" && echo "$NAME relocated + symlinked"
  else
    echo "MISMATCH for $NAME - left original in place"; exit 1
  fi
done
df -h /sailhome/ravenh | tail -1
