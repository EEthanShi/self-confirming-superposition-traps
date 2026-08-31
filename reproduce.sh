#!/usr/bin/env bash
# Reproduce everything in the paper that does not need a GPU.
#
# The E2 cohorts and the E3 PPO cohort are excluded here: they were executed on
# a GPU server and take hours. Their frozen reports ship under results/, and
# the commands that re-run them are given in the README.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "== self-tests =="
python3 "$ROOT/tests.py"

echo
echo "== theory: executable algebra for every theorem =="
python3 "$ROOT/scripts/check_theory.py" | tail -1

echo
echo "== theory: untied-decoder phase sweep =="
python3 "$ROOT/scripts/untied_phase_sweep.py" | tail -1

echo
echo "== E1: solved-class finite-sample study (about 15 s) =="
cd "$ROOT/experiments/sampled_phase"
python3 run_experiment.py --output-dir "$ROOT/outputs_rebuild" > /dev/null
if diff -rq "$ROOT/outputs_rebuild" "$ROOT/experiments/sampled_phase/outputs" > /dev/null; then
  echo "E1 outputs reproduce the released files byte for byte"
else
  echo "E1 outputs DIFFER from the released files" >&2
  exit 1
fi

echo
echo "== figures: E1 panel from the released outputs =="
cd "$ROOT"
python3 figures/make_e1_figure.py > /dev/null 2>&1 || \
  echo "(skipped: figure script writes into the paper tree)"

echo
echo "done"
