#!/usr/bin/env bash
# Regenerates docs/figures/example_old_students/ from the previous project's artifacts.
#   bash scripts/analysis/examples/run_old_students_example.sh DEV_REPO   (run on a compute node)
# DEV_REPO is the previous project's checkout holding runs/ and state/ (not distributed: per-paper data).
set -euo pipefail
main() {
  local dev=${1:?usage: $0 DEV_REPO}
  local here; here=$(cd "$(dirname "$0")/../../.." && pwd)
  local out=$here/docs/figures/example_old_students
  local py=${PY:-python}
  cd "$here"
  "$py" scripts/analysis/collect_results.py --systems scripts/analysis/examples/old_students.json \
    --runs-dir "$dev/runs" --state-dir "$dev/state" --fact-dirs factcheck --common-papers --out-dir "$out"
  "$py" scripts/analysis/collect_results.py --systems scripts/analysis/examples/old_prefcc_verdict_check.json \
    --runs-dir "$dev/runs" --state-dir "$dev/state" --out-dir "$out/verdict_check"
  "$py" scripts/analysis/pareto.py "$out/results.json" --out-dir "$out/pareto" --costs gpu_s_per_paper,params_b,train_gpu_hours \
    --recall-floor -0.05 --load 100000
  "$py" scripts/analysis/train_resources.py main k2 q35 q35prefcc q35prefccs2 --runs-dir "$dev/runs" \
    --out-json "$out/train_resources.json" --out-md "$out/train_resources.md"
  "$py" scripts/analysis/learning_curves.py main k2 q35 q35prefcc q35prefccs2 --runs-dir "$dev/runs" \
    --out-dir "$out/curves" --label "main=A Qwen3-4B" --label "k2=B K2-Horizon-7B" --label "q35=C Qwen3.5-9B" \
    --label "q35prefcc=prefcc seed 1" --label "q35prefccs2=prefcc seed 2"
}
main "$@"
