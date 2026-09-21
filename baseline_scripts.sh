#!/usr/bin/env bash

models=(DecisionTree RandomForest SVM XGBoost LightGBM MLP)
tasks=(Cartridge Step Solvent Ratio)

failed=()

for m in "${models[@]}"; do
  for t in "${tasks[@]}"; do
    echo "Running $m $t"
    if ! python baseline.py "$m" "$t" --data_dir data --output_dir output/baseline_output; then
      echo "[FAIL] $m $t"
      failed+=("$m $t")
    fi
  done
done

if [ ${#failed[@]} -gt 0 ]; then
  echo "Failed jobs:"
  printf ' - %s\n' "${failed[@]}"
  exit 1
fi

echo "All baseline jobs finished. Computing 5 overall metrics..."
python compute_baseline_overall.py --output_dir output/baseline_output --models "$(IFS=,; echo "${models[*]}")"
echo "All done."