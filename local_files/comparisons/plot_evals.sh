#!/usr/bin/env bash
for f in skill_score*.csv; do
    # echo "$f{f%.csv}.png"
    python plot_evals_save_regions.py --csv_path "$f" --output_path "./plot_regions/${f%.csv}.png"
done