# Frozen protocol: sampled optimization of the tied phase diagram

Protocol version: 1.0  
Frozen before the primary run: 2026-08-09  
Primary seed: 20260809

## Question and scope

The experiment asks whether ordinary finite-sample gradient optimization of the
exact tied reconstruction objective recovers the population geometry proved in
Theorem 2. It is an explanatory optimization check inside the solved model. It
is not evidence that self-confirming superposition traps are prevalent in deep
RL, and it does not test the actor dynamics.

## Data generator and loss

For each occupancy `p`, sample `N = 4096` independent episodes using the
branch/pair generator in Section 3 and an allowed standard-normal feature
distribution:

- `B ~ Bernoulli(p)`;
- conditional on `B=S`, choose pair `13` or `23` with probability `1/2`;
- conditional on `B=E`, choose pair `12` or `13` with probability `1/2`;
- sample active feature values independently from `Normal(0,1)`;
- use `sigma = 0` in the primary experiment.

In the tied model, nonzero `sigma` adds the same representation-independent
constant to each branch loss, so this choice does not change the optimized
geometry or control contrast.

The optimized loss is the sum of squared reconstruction errors over the two
active targets only. The code and tied readout are exactly those in the paper.
No full-three-output loss, regularizer, raw-state skip, or analytic population
solution is supplied to the optimizer.

For numerical efficiency, the script reduces the fixed raw dataset to its three
exact sufficient coefficients after first checking, on raw sampled batches and
random codes, that the unreduced reconstruction loss and gradient agree with
the reduced empirical objective to `1e-10`.

## Primary two-dimensional run

- Occupancies: the uniform grid `{0, 0.05, ..., 1}` together with the two exact
  population thresholds `p_-` and `p_+`.
- Independent datasets per occupancy: `50`.
- Independent optimizer initializations per dataset: `8`.
- Gauge: `v3=(1,0)`; optimize line angles `alpha,beta ~ Uniform(0,pi)` for
  `v1=(cos(alpha),sin(alpha))` and `v2=(cos(beta),sin(beta))`.
- Optimizer: full-batch gradient descent on the finite-sample reconstruction
  risk, fixed learning rate `0.05`, at most `20000` steps, float64, no momentum,
  no weight decay, and no initialization filtering.
- Stopping rule: projected gradient infinity norm below `1e-10`.
- Every run is retained. The exact empirical global optimum is computed from
  the independently proved weighted-frame solution and used only after
  training as an optimization certificate.

The population curves are never used for model selection. Because the paper's
theory assumes a global representation best response, the main learned marker
for each dataset uses the lowest empirical-loss result among its eight fixed
initializations. The single-start success rate and all failed starts are also
reported.

## Errors and summaries

The experiment keeps two errors separate:

- optimization error: trained empirical loss minus the certified empirical
  global optimum;
- sampling error: certified empirical Gram matrix minus the population Gram
  matrix at the same occupancy.

The main figure reports the median and 10--90 percent dataset-dispersion interval
across the 50 independent datasets. It uses squared Gram entries and
`D=g12-g23`, which are invariant to line signs and common rotations. The three
displayed line geometries at `p=0.2,0.5,0.8` are medoids under squared-Gram
distance, never visually selected seeds.

## Three-dimensional capacity control

At `p in {0.2,0.5,0.8}`, use the same datasets, seed convention, number of
initializations, optimization budget, tied loss, and unit-column constraint,
but learn three vectors in `R^3` by projected gradient descent on the product of
spheres. Endpoints are excluded because an absent pair makes its angle
unidentified by the empirical risk.

## Held-out control distortion

For the learned two- and three-dimensional codes, estimate `D_S` and `D_E` on
independent forced-branch samples and compare `D_E-D_S` with `g12-g23`. For the
illustrative control-gap check use the pre-fixed base gap `delta=0.5`.

## Sample-size robustness

Without rerunning the optimizer, draw 100 independent datasets for each

`N in {256,1024,4096,16384}` and
`p in {0.2,0.35,p_-,0.5,p_+,0.65,0.8}`.

Use the certified empirical optimum to isolate sampling variation. Report
medians and 10--90 percent intervals without fitting or claiming a convergence
rate.

## Frozen pass/fail criteria

1. Raw and reduced sampled losses and gradients agree within `1e-10`.
2. At least 99 percent of datasets have a best-of-eight optimization error below
   `1e-8`; the single-start rate is reported separately.
3. The median maximum squared-Gram distance from the certified empirical
   optimum is below `0.02`.
4. At `p=0.2`, at least 95 percent of datasets have
   `g12>0.99`, `g13<0.01`, and `g23<0.01`; the symmetric condition holds at
   `p=0.8` for `g23`.
5. At `p=0.2` and `delta=0.5`, at least 95 percent of learned two-dimensional
   codes reverse the branch gap.
6. In the three-dimensional control, at least 95 percent of datasets have
   `max(g12,g13,g23)<0.01` and retain a positive branch gap for `delta=0.5`.

No threshold, occupancy, seed, or run may be changed or discarded after the
primary results are observed. A failed criterion is reported as a failed
criterion rather than repaired post hoc.
