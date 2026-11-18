#!/usr/bin/env bash
# Whole sleep study after the data and source models exist (data.sleepedf, train_sleep.py):
# tuning on SC-mid only, then every method on all targets for 5 seeds, then tables + figure.
# Seeds run 3 + 2 at a time: 5 at once runs out of memory on a 6 GB GPU.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONWARNINGS=ignore

python study_sleep_tta.py --tune
python study_sleep_tta.py --tune-balanced

run_seed() { python study_sleep_tta.py --seeds "$1" --out "outputs/sleep_tta_s$1.json" > "outputs/sleep_tta_s$1.log" 2>&1; }
run_seed 0 & run_seed 1 & run_seed 2 & wait
run_seed 3 & run_seed 4 & wait

python summarize_sleep.py > /dev/null
python plot_sleep.py
echo "done: outputs/sleep_summary.md, figures/sleep_results.png"
