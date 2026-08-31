# self-confirming-superposition-traps

Code and released results for the paper *Self-Confirming Superposition Traps
in Reinforcement Learning* (under review).

A reinforcement learning agent trains its representation on the data its own
policy collects. When capacity is scarce, the optimal code stores rarely
co-activated features on shared directions, and the policy itself decides
which features are rare. A policy that avoids an action therefore makes the
features carrying that action cheap to collide, the collision lowers the value
the deployed controller assigns to that action, and the avoidance feeds
itself. The paper defines this configuration through four conditions, which it
calls **allocation**, **reversal**, **feedback**, and **suboptimality**, solves
a horizon-two model in which all four hold at every global optimum, and then
measures the loop at three increasing distances from that solved class.

This repository holds the executable algebra behind every theorem, the three
experiments, the frozen preregistrations and gate tables they were judged
against, and the raw results behind every number in the paper.

The theory checks and E1 need only NumPy and SciPy. E2 figures and the E3 PPO
cohort additionally need Matplotlib and PyTorch.

## Quick start

Verify the whole repository, which re-derives the theorems numerically and
runs every unit test (about 20 seconds, no GPU):

```bash
python3 tests.py
```

Reproduce every CPU result in the paper, including a byte-for-byte rebuild of
the E1 outputs (about one minute, no GPU):

```bash
bash reproduce.sh
```

Check the theorems on their own. The first script evaluates the closed forms
of the phase diagram, the envelope identities, the finite-rate inertia and the
untied-decoder endpoints against direct computation; the second sweeps the
untied decoder across noise levels:

```bash
python3 scripts/check_theory.py
python3 scripts/untied_phase_sweep.py
```

Rerun E1, the solved-class finite-sample study, which fits 50 datasets at each
of 23 occupancies from eight starts and certifies every fit against an
independent optimum certificate:

```bash
cd experiments/sampled_phase && python3 run_experiment.py
```

Rerun the E2 closed-loop grids and the E3 PPO cohort. These were executed on a
GPU server and take hours, so their frozen reports ship under `results/`:

```bash
cd experiments/closed_loop
python3 run_final_core.py            # basins, separator, time-scale series
python3 run_final_interventions.py   # entropy, visitation floor, replay, orthogonality
python3 run_final_neural.py          # the learned-representation bridge

cd ../ppo
python3 -m e3.final_cohort           # 100 seeds per arm, standard PPO
```

Rebuild the paper's figures from released results:

```bash
python3 figures/make_e1_figure.py
python3 figures/make_figs_final.py
```

## Layout

| Path | Content |
| --- | --- |
| `scripts/check_theory.py` | executable algebra for every theorem, printing `ALL_CHECKS_PASSED` |
| `scripts/untied_phase_sweep.py` | untied-decoder sweep across noise levels |
| `experiments/sampled_phase/` | E1, the solved-class finite-sample study, with its frozen `PROTOCOL.md` and released `outputs/` |
| `experiments/closed_loop/e2/` | the closed loop: solved model, two-timescale dynamics with intervention hooks, cohort statistics, and the learned-representation bridge |
| `experiments/closed_loop/run_final_*.py` | the three frozen E2 runs behind the paper's numbers |
| `experiments/ppo/e3/` | E3: the two-step environment, a line-auditable PPO with GAE, the frozen evaluators, and the final cohort |
| `preregistration/` | every frozen specification and gate table, including the two E3 pilots that failed their gates |
| `results/` | raw reports behind every number, with `E2_RESULTS.md` as the reading guide |
| `figures/` | figure scripts and the phase-sweep data they consume |
| `tests/` | the E2 analytic tests and the E3 unit tests, driven by `tests.py` |

## Integrity notes

Every experiment was frozen before execution, and the frozen documents ship in
`preregistration/` next to the gate tables that judged them. Registered
failures are kept rather than removed. E2's orthogonality gate fixed a single
initialization and asked when that point escaped, so it measured membership of
one point rather than the width of a basin, and its verdict stands as a
failure; the development follow-up that measures the width it should have
tested is reported separately. E2's return gate in the constrained bridge arm
compared returns at a balanced reference occupancy, where the constrained
class is provably symmetric between endpoints, so it tested a quantity the
theory itself sets to zero, and the confirmatory return chain stays open. Two
E3 pilots failed their frozen gates before the final cohort and both receipts
are retained.

The E1 rebuild is byte-for-byte on the pinned environment, which
`reproduce.sh` checks. E2 and E3 ran on a GPU server with fixed seeds and
common random numbers across paired arms; their reports are released rather
than regenerated here, because the cohorts take hours.

What the evidence supports is stated at its own tier. The solved class is
proved. E1 measures the static allocation-to-gap link, E2 measures the closed
loop and the intervention thresholds inside the solved class, and E3
establishes that a trap consistent with the mechanism exists in a controlled
standard PPO stack. Prevalence across deep RL practice is not claimed, and the
uniqueness of the cross-talk mediator remains open.

## License

MIT. See `LICENSE`.
