# Sampled phase-diagram reproduction

This directory contains the finite-sample optimization check embedded after the
global phase-diagram theorem. Its scope and pass/fail criteria were frozen in
`PROTOCOL.md` before the primary run.

Run from this directory with NumPy and Python 3:

```sh
python3 -m unittest -v test_experiment.py
python3 run_experiment.py
```

For a smoke test, `python3 run_experiment.py --quick` writes to the separate
`outputs_quick/` directory and refuses to overwrite the official outputs.

The script samples the exact horizon-two generator, trains the normalized tied
code, certifies every result against the empirical weighted-frame optimum, runs
the learned three-dimensional capacity control, evaluates branch distortion on
fresh forced-branch samples, and writes all retained runs under `outputs/`.

The paper consumes `figure_summary.csv`, `representative_lines.tex`, and
`sample_size_table_rows.tex`. All optimizer starts, datasets, controls, and
failure counts remain available in the same output directory for audit.
