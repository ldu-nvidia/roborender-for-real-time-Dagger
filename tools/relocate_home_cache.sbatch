#!/bin/bash
#SBATCH --job-name=relocate_cache
#SBATCH --account=general_cs_infra
#SBATCH --partition=backfill_block1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --output=/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/logs/relocate_cache_%j.log
set -euo pipefail
STAGE_ROOT=${ROBORENDER_STAGE_ROOT:-/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale}
DEST=$STAGE_ROOT/runtime/cache/home
: "${SOURCE_HOME:?SOURCE_HOME must be the home directory whose caches will be copied}"
mkdir -p "$DEST"
for NAME in openpi huggingface; do
  SRC=$SOURCE_HOME/.cache/$NAME
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
df -h "$SOURCE_HOME" | tail -1
