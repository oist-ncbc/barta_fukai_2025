#!/bin/bash

systems=(
    "hebb_recharge_minus_strong_tr2.0"
    "hebb_recharge_minus_strong_tr2.1"
    "hebb_recharge_minus_strong_tr2.2"
    "hebb_recharge_minus_strong_tr2.3"
    "hebb_recharge_minus_strong_tr2.4"
    "hebb_recharge_minus_strong_tr2.5"
    "hebb_recharge_minus_strong_tr2.6"
    "hebb_recharge_minus_strong_tr2.7"
    "hebb_recharge_plus_strong_tr1.5"
    "hebb_recharge_plus_strong_tr1.6"
    "hebb_recharge_plus_strong_tr1.7"
    "hebb_recharge_plus_strong_tr1.8"
    "hebb_recharge_plus_strong_tr1.9"
    "hebb_recharge_plus_strong_tr2.0"
    "hebb_recharge_plus_strong_tr2.1"
    "hebb_recharge_plus_strong_tr2.2"
    "hebb_recharge_plus_strong_tr2.3"
    "hebb_recharge_plus_strong_tr2.4"
    "hebb_recharge_plus_strong_tr2.5")
patterns=(1000 1200 1400 1600 1800 2000)
run_id=$$
simulation_id="$(date +%Y%m%d_%H%M%S)_$run_id"

TMP_DIR="tmp/simulation_$simulation_id"
mkdir -p "$TMP_DIR"
git archive --format=tar HEAD | tar -x -C "$TMP_DIR"
cd "$TMP_DIR"

for system in "${systems[@]}"; do
    for num in "${patterns[@]}"; do
        mkdir -p logs

        bash run_scripts/run_notrain_deigo.sh "$num" "$system" "$simulation_id" &
    done
done