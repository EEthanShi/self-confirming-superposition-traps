"""Frozen statistics for the cohort. Rules fixed before execution:
- fraction trapped per (arm, delta, floor, p0) over seeds, failed runs
  excluded and counted;
- boundary: scan the p0 grid ascending, take the LAST index i with
  frac[i] >= 0.5 > frac[i+1]; linear interpolation to 0.5; all-below ->
  -inf sentinel, all-above -> +inf sentinel;
- CI: percentile bootstrap over seeds, B=2000, PCG64(123456), the SAME
  resample index matrix reused across arms and floor levels (paired);
- paired floor shift: boundary(eps=0.05) - boundary(eps=0.15) per bootstrap
  replicate, percentile interval.
"""
import numpy as np

BOOT_B, BOOT_SEED = 2000, 123456


def frac_curve(records, p0_grid):
    out = []
    for v in p0_grid:
        rs = [r for r in records if abs(r["p0"] - v) < 1e-9 and not r["failed"]]
        out.append(np.mean([r["basin"] == 0 for r in rs]) if rs else np.nan)
    return np.array(out)


def boundary(frac, p0_grid):
    idx = None
    for i in range(len(p0_grid) - 1):
        if frac[i] >= 0.5 > frac[i + 1]:
            idx = i
    if idx is None:
        if np.all(frac < 0.5):
            return -np.inf
        if np.all(frac >= 0.5):
            return np.inf
        return np.nan
    t = (0.5 - frac[idx + 1]) / (frac[idx] - frac[idx + 1])
    return float(p0_grid[idx + 1] - t * (p0_grid[idx + 1] - p0_grid[idx]))


def boot_indices(n_seeds):
    rng = np.random.Generator(np.random.PCG64(BOOT_SEED))
    return rng.integers(0, n_seeds, size=(BOOT_B, n_seeds))


def boot_boundaries(records, p0_grid, seeds, idxmat):
    seeds = list(seeds)
    table = {}
    for r in records:
        table.setdefault((r["seed"], round(r["p0"], 6)), r)
    bs = []
    for row in idxmat:
        chosen = [seeds[i] for i in row]
        frac = []
        for v in p0_grid:
            vals = [1 - table[(s, round(v, 6))]["basin"]
                    for s in chosen if (s, round(v, 6)) in table
                    and not table[(s, round(v, 6))]["failed"]]
            frac.append(np.mean(vals) if vals else np.nan)
        bs.append(boundary(np.array(frac), p0_grid))
    return np.array(bs)


def ci(arr, lo=2.5, hi=97.5):
    """Percentile CI plus the non-finite fraction. A cell whose bootstrap
    boundaries are non-finite in more than 5% of replicates is UNRESOLVED:
    survivor-filtered percentiles would fake precision (external review)."""
    fin = arr[np.isfinite(arr)]
    bad = 1.0 - len(fin) / max(len(arr), 1)
    if len(fin) == 0:
        return [float("nan")] * 2, bad
    return [float(np.percentile(fin, lo)), float(np.percentile(fin, hi))], bad
