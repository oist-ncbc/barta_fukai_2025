#!/bin/bash

conditions=(
    "hebb"
    "hebb_smooth_rate"
    "rate"
    )
patterns=(200 400 600)
run_id=$$
simulation_id="$(date +%Y%m%d_%H%M%S)_$run_id"

TMP_DIR="tmp/simulation_$simulation_id"
mkdir -p "$TMP_DIR"
git archive --format=tar HEAD | tar -x -C "$TMP_DIR"
cd "$TMP_DIR"

for condition in "${conditions[@]}"; do
    for num in "${patterns[@]}"; do
        mkdir -p logs

        ./run_scripts/run_all_deigo.sh "$num" "$condition" "$simulation_id" &
    done
done