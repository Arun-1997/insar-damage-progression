#!/bin/bash
# ─────────────────────────────────────────────────────────────
# Phase 1 — Batch SNAP preprocessing
# Runs insar_preprocessing.xml for all pre/post scene pairs
#
# Usage:
#   chmod +x run_snap_preprocessing.sh
#   ./run_snap_preprocessing.sh
#
# Requirements:
#   SNAP installed: https://step.esa.int/main/download/snap-download/
#   Set SNAP_HOME below to your SNAP installation path
#
# Runtime: ~20–40 min per pair on a laptop (CPU-bound)
# ─────────────────────────────────────────────────────────────

set -e  # exit on any error

SNAP_HOME="/usr/local/snap"       # adjust to your SNAP install path
GPT="$SNAP_HOME/bin/gpt"
GRAPH="./insar_preprocessing.xml"

RAW_DIR="./data/raw_slc"
OUT_DIR="./data/processed"
mkdir -p "$OUT_DIR"

MASTER=$(ls "$RAW_DIR/pre_event/"*.zip | head -1)

# ── all post-event scenes ──────────────────────────────────────
POST_SCENES=(
    "post_01"
    "post_02"
    "post_03"
    "post_04"
    "post_05"
    "post_06"
    "post_07"
    "post_08"
)

echo "=================================================="
echo " SNAP InSAR Batch Preprocessing"
echo " Master: $MASTER"
echo "=================================================="

for SCENE in "${POST_SCENES[@]}"; do
    SLAVE=$(ls "$RAW_DIR/$SCENE/"*.zip 2>/dev/null | head -1)

    if [ -z "$SLAVE" ]; then
        echo "⚠  Skipping $SCENE — no .zip file found"
        continue
    fi

    OUTPUT="$OUT_DIR/${SCENE}"
    LOGFILE="$OUT_DIR/${SCENE}.log"

    echo ""
    echo "── Processing: $SCENE ──────────────────────────"
    echo "   Slave:  $SLAVE"
    echo "   Output: $OUTPUT"
    echo "   Log:    $LOGFILE"

    # Run SNAP GPT
    "$GPT" "$GRAPH" \
        -Pmaster="$MASTER" \
        -Pslave="$SLAVE" \
        -Poutput="$OUTPUT" \
        -c 4G \            # cache size — increase if you have RAM
        -q 4 \             # number of parallel tiles
        2>&1 | tee "$LOGFILE"

    if [ $? -eq 0 ]; then
        echo "   ✓ Done: $SCENE"
    else
        echo "   ✗ FAILED: $SCENE — check $LOGFILE"
    fi
done

echo ""
echo "=================================================="
echo " Preprocessing complete."
echo " GeoTIFFs saved to: $OUT_DIR"
echo " Next: run python build_stack.py"
echo "=================================================="
