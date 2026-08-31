"""E3 competence pilot: complete runner + mechanical gate reducer + receipt.
GATED: may run only after the preregistration passes external static review.
Usage:  python -m e3.pilot          (from the bundle root)
Writes out/e3_pilot_report.json with per-run records, three-signal verdicts
(mechanical), and a receipt. No checkpoint/resume (runs are minutes)."""
import json, os, socket, platform, time
import numpy as np
from multiprocessing import Pool

SIGMA, DELTA = 0.30, 0.3
SEEDS = list(range(5000, 5020))
FREE_UPDATES = 400


def _pin_threads():
    """Operational: one intra-op thread per worker. Prevents 20x24-thread
    oversubscription (first launch aborted at load ~400/24 cores; the report
    was never produced and no results were read; scientific content and
    seeds unchanged) and strengthens determinism."""
    import torch, os
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"


def forced_cell(args):
    _pin_threads()
    from .ppo import run_training
    from . import evaluators as ev
    seed, k, branch = args
    net, logs, S = run_training(k_proj=k, delta=DELTA, seed=seed,
                                updates=200, forced_branch=branch,
                                sigma_obs=SIGMA)
    return dict(kind="forced", seed=seed, k=k, forced=int(branch),
                competence=ev.competence(net, S["eval"], sigma_obs=SIGMA),
                xtalk=ev.crosstalk(net, S["eval"], sigma_obs=SIGMA),
                gram=ev.probe_gram(net, S["eval"], sigma_obs=SIGMA))


def free_cell(args):
    _pin_threads()
    from .ppo import run_training
    from . import evaluators as ev
    seed, k, p0 = args
    net, logs, S = run_training(k_proj=k, delta=DELTA, seed=seed, p0=p0,
                                updates=FREE_UPDATES, sigma_obs=SIGMA)
    cut = logs[-1]["update"] * 3 // 4          # last quarter of actual run
    tail = [l["p_E"] for l in logs if l["update"] >= cut]
    return dict(kind="free", seed=seed, k=k, p0=p0,
                p_E_end=logs[-1]["p_E"],
                tail_drift=float(max(tail) - min(tail)),
                xtalk=ev.crosstalk(net, S["eval"], sigma_obs=SIGMA),
                gram=ev.probe_gram(net, S["eval"], sigma_obs=SIGMA),
                gap=ev.forced_gap(net, S["eval"], sigma_obs=SIGMA,
                                  delta=DELTA),
                ret_det=ev.deployment_return(net, S["eval"], sigma_obs=SIGMA,
                                             delta=DELTA),
                ret_smp=ev.deployment_return(net, S["eval"], sigma_obs=SIGMA,
                                             delta=DELTA, sampled=True))


def _exact_binom_ci(flags, alpha=0.05):
    """Pre-frozen two-sided Clopper-Pearson interval over seed-level runs.
    All-success at n=5 gives lower bound 0.478 < 0.5, so the P3 gate
    implies n >= 6 successes minimum (stated in the prereg)."""
    from scipy.stats import beta
    n = len(flags)
    if n == 0:
        return None
    k = int(sum(flags))
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return [lo, hi]


def gates(recs):
    # --- P1: competence, PER CONDITION (k x trained branch), primary =
    # the branch the model was actually trained on; cross-branch secondary ---
    p1_conds, p1_secondary = {}, {}
    for k in (2, 64):
        for br, name in ((0, "S"), (1, "E")):
            sel = [r for r in recs if r["kind"] == "forced" and r["k"] == k
                   and r["forced"] == br]
            oks = [r["competence"][name]["ratio_to_oracle"] >= 0.85
                   for r in sel]
            other = "E" if name == "S" else "S"
            p1_conds[f"k{k}_{name}"] = dict(
                passed=bool(sum(oks) >= 18 and len(oks) >= 20),
                detail=f"{sum(oks)}/{len(oks)}")
            p1_secondary[f"k{k}_{name}_cross"] = float(np.mean(
                [r["competence"][other]["ratio_to_oracle"] for r in sel]))                 if sel else None
    p1 = dict(passed=all(c["passed"] for c in p1_conds.values()),
              conditions=p1_conds, cross_branch_secondary=p1_secondary)

    # --- P2: two-basin indication with two-sided control and stability ---
    def sel_free(k, p0):
        return [r for r in recs if r["kind"] == "free" and r["k"] == k
                and r["p0"] == p0]
    lows, highs = sel_free(2, 0.1), sel_free(2, 0.9)
    cl, ch = sel_free(64, 0.1), sel_free(64, 0.9)
    by_seed = lambda rs: {r["seed"]: r["p_E_end"] for r in rs}
    hl, ll = by_seed(highs), by_seed(lows)
    paired = [hl[s] - ll[s] for s in sorted(set(hl) & set(ll))]
    split = float(np.mean(paired)) if paired else float("nan")
    rng = np.random.default_rng(43)
    pa = np.array(paired)
    bs = pa[rng.integers(0, len(pa), size=(2000, len(pa)))].mean(axis=1)
    split_ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    chl, cll = by_seed(ch), by_seed(cl)
    cpaired = [chl[s] - cll[s] for s in sorted(set(chl) & set(cll))]
    csplit = float(np.mean(cpaired)) if cpaired else float("nan")
    drift_ok = all(r["tail_drift"] <= 0.1
                   for r in lows + highs + cl + ch)
    p2 = dict(passed=bool(split >= 0.5 and abs(csplit) <= 0.15 and drift_ok),
              split=split, split_ci=split_ci, control_split=csplit,
              drift_ok=drift_ok)

    # --- P3: cross-talk mediator, BOTH sides, seed-unit CIs
    # (theory-inspired operational analogue, not a direct theorem quantity) ---
    trap_low = [r for r in lows if r["p_E_end"] < 0.5]
    esc_high = [r for r in highs if r["p_E_end"] > 0.5]
    fl = [r["xtalk"]["x12"] > r["xtalk"]["x23"] for r in trap_low]
    fh = [r["xtalk"]["x23"] > r["xtalk"]["x12"] for r in esc_high]
    ci_l, ci_h = _exact_binom_ci(fl), _exact_binom_ci(fh)
    p3 = dict(passed=bool(len(fl) >= 5 and len(fh) >= 5
                          and np.mean(fl) >= 0.9 and np.mean(fh) >= 0.9
                          and ci_l[0] > 0.5 and ci_h[0] > 0.5),
              low=dict(n=len(fl), frac=float(np.mean(fl)) if fl else None,
                       ci=ci_l),
              high=dict(n=len(fh), frac=float(np.mean(fh)) if fh else None,
                        ci=ci_h))
    return dict(P1=p1, P2=p2, P3=p3,
                all_pass=bool(p1["passed"] and p2["passed"] and p3["passed"]))


def make_receipt(t0, recs, jobs):
    """Binds the run to the deployed artifact: reads deploy_stamp.json
    (written by deploy.sh from the real git state), re-hashes the deployed
    prereg to detect post-deploy tampering, and fingerprints raw output."""
    import hashlib
    stamp = json.load(open("deploy_stamp.json"))
    spec_now = hashlib.sha256(
        open("E3_PREREGISTRATION.deployed.md", "rb").read()).hexdigest()
    assert spec_now == stamp["spec_sha256"], "deployed spec drifted from stamp"
    raw = json.dumps(recs, sort_keys=True).encode()
    return dict(stamp=stamp, spec_sha256_at_runtime=spec_now,
                raw_output_sha256=hashlib.sha256(raw).hexdigest(),
                n_jobs=len(jobs), n_records=len(recs),
                n_failed=sum(1 for r in recs if r.get("failed")),
                seeds=sorted({r["seed"] for r in recs}),
                host=socket.gethostname(),
                python=platform.python_version(), numpy=np.__version__,
                started_unix=t0, finished_unix=time.time())


def expected_jobs():
    jobs_f = [(s, k, b) for s in SEEDS for k in (2, 64) for b in (0, 1)]
    jobs_r = [(s, k, p0) for s in SEEDS for k in (2, 64)
              for p0 in (0.1, 0.5, 0.9)]
    return jobs_f, jobs_r


def admission(report_path="out/e3_pilot_report.json"):
    """Pre-run gate. Runs BEFORE any Pool, seed read, or training. Exits on:
    missing stamp; dirty deploy; spec or any E3 payload hash mismatch;
    inexact file census; pre-existing report (refuse overwrite)."""
    import hashlib
    if not os.path.exists("deploy_stamp.json"):
        raise SystemExit("ADMISSION: deploy_stamp.json missing")
    stamp = json.load(open("deploy_stamp.json"))
    if stamp.get("dirty_files", 1) != 0:
        raise SystemExit("ADMISSION: deployed from dirty worktree")
    def sha(f):
        return hashlib.sha256(open(f, "rb").read()).hexdigest()
    if sha("E3_PREREGISTRATION.deployed.md") != stamp["spec_sha256"]:
        raise SystemExit("ADMISSION: deployed spec hash mismatch")
    payload = stamp.get("payload_sha256", {})
    expected_files = sorted(payload)
    actual = sorted(
        os.path.join(d, f) for d in ("e3", "tests") if os.path.isdir(d)
        for f in os.listdir(d)
        if f.endswith(".py") and (d == "e3" or f == "test_e3.py"))
    if expected_files != actual:
        raise SystemExit(f"ADMISSION: payload census mismatch "
                         f"{expected_files} vs {actual}")
    for f, h in payload.items():
        if sha(f) != h:
            raise SystemExit(f"ADMISSION: payload hash mismatch {f}")
    jf, jr = expected_jobs()
    jobs = jf + jr
    if len(jobs) != 200 or len(set(jobs)) != 200:
        raise SystemExit("ADMISSION: job census not exactly 200 unique")
    if os.path.exists(report_path):
        raise SystemExit("ADMISSION: report exists; refusing overwrite")
    return stamp, jf, jr


if __name__ == "__main__":
    stamp, jobs_f, jobs_r = admission()
    t0 = time.time()
    with Pool(20) as pool:
        recs = pool.map(forced_cell, jobs_f) + pool.map(free_cell, jobs_r)
    rep = dict(frozen="E3_PREREGISTRATION.md", records=recs,
               gates=gates(recs),
               receipt=make_receipt(t0, recs, jobs_f + jobs_r))
    os.makedirs("out", exist_ok=True)
    tmp = "out/.e3_pilot_report.tmp"
    json.dump(rep, open(tmp, "w"))
    os.replace(tmp, "out/e3_pilot_report.json")     # atomic
    print(json.dumps(rep["gates"], indent=1))
