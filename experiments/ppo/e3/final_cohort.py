"""E3 FINAL cohort per E3_FINAL_FROZEN.md (same-commit freeze). GATED:
requires external static review PASS before execution. Usage:
python -m e3.final_cohort"""
import json, os, socket, platform, time, hashlib
import numpy as np
from multiprocessing import Pool
from scipy.stats import beta

SIGMA, EPS = 0.30, 0.05
CFG = dict(delta=0.15, rls=0.3)
REF = 0.23511118022080546          # rank-2 constructive reference, sigma=.30


def _pin():
    import torch
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"


def cell(args):
    _pin()
    from .ppo import run_training
    from . import evaluators as ev
    tag, seed, k, q0, rls = args
    net, logs, S = run_training(k_proj=k, delta=CFG["delta"], seed=seed,
                                p0=q0, updates=400, sigma_obs=SIGMA,
                                floor_eps=EPS, root_lr_scale=rls,
                                log_every=1)
    cut = logs[-1]["update"] * 3 // 4
    tail = [l["p_E"] for l in logs if l["update"] >= cut]
    comp = ev.competence(net, S["eval"], sigma_obs=SIGMA)
    return dict(tag=tag, seed=seed, k=k, q0=q0, rls=rls,
                p_E_end=logs[-1]["p_E"],
                tail_drift=float(max(tail) - min(tail)),
                mse_S=comp["S"]["mse"], mse_E=comp["E"]["mse"],
                xtalk=ev.crosstalk(net, S["eval"], sigma_obs=SIGMA),
                gap=ev.forced_gap(net, S["eval"], sigma_obs=SIGMA,
                                  delta=CFG["delta"]),
                ret=ev.deployment_return(net, S["eval"], sigma_obs=SIGMA,
                                         delta=CFG["delta"], floor_eps=EPS))


def expected_jobs():
    J = []
    for s in range(6000, 6100):
        J.append(("k2_low", s, 2, 0.05, CFG["rls"]))
        J.append(("k64_low", s, 64, 0.05, CFG["rls"]))
    for s in range(6100, 6200):
        J.append(("k2_high", s, 2, 0.9, CFG["rls"]))
        J.append(("k64_high", s, 64, 0.9, CFG["rls"]))
    for s in range(6000, 6030):
        J.append(("comparator_low", s, 2, 0.05, 1.0))
    return J


def admission_final(report_path="out/e3_final_report.json"):
    """Same five refusals as pilot admission, final census (430 unique)."""
    if not os.path.exists("deploy_stamp.json"):
        raise SystemExit("ADMISSION: deploy_stamp.json missing")
    stamp = json.load(open("deploy_stamp.json"))
    if stamp.get("dirty_files", 1) != 0:
        raise SystemExit("ADMISSION: deployed from dirty worktree")
    def sha(f):
        return hashlib.sha256(open(f, "rb").read()).hexdigest()
    if sha("E3_PREREGISTRATION.deployed.md") != stamp["spec_sha256"]:
        raise SystemExit("ADMISSION: deployed spec hash mismatch")
    if "final_spec_sha256" not in stamp:
        raise SystemExit("ADMISSION: stamp lacks final_spec_sha256")
    if sha("E3_FINAL_FROZEN.deployed.md") != stamp["final_spec_sha256"]:
        raise SystemExit("ADMISSION: deployed FINAL spec hash mismatch")
    payload = stamp.get("payload_sha256", {})
    actual = sorted(
        os.path.join(d, f) for d in ("e3", "tests") if os.path.isdir(d)
        for f in os.listdir(d)
        if f.endswith(".py") and (d == "e3" or f == "test_e3.py"))
    if sorted(payload) != actual:
        raise SystemExit("ADMISSION: payload census mismatch")
    for f, h in payload.items():
        if sha(f) != h:
            raise SystemExit(f"ADMISSION: payload hash mismatch {f}")
    jobs = expected_jobs()
    if len(jobs) != 430 or len(set(jobs)) != 430:
        raise SystemExit("ADMISSION: job census not exactly 430 unique")
    if os.path.exists(report_path):
        raise SystemExit("ADMISSION: report exists; refusing overwrite")
    return stamp, jobs


def cp(flags, alpha=0.05):
    n, k = len(flags), int(sum(flags))
    if n == 0:
        return None
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return [lo, hi]


def _boot_mean(x, seed0):
    rng = np.random.default_rng(seed0)
    x = np.asarray(x, dtype=float)
    bs = x[rng.integers(0, len(x), size=(2000, len(x)))].mean(axis=1)
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def gates(recs):
    R = {t: [r for r in recs if r["tag"] == t] for t in
         ("k2_low", "k2_high", "k64_low", "k64_high", "comparator_low")}
    low, high = R["k2_low"], R["k2_high"]
    # F1: ENDPOINT-CONDITIONED validity (twelfth review): the agent must have
    # learned the task on the branch it actually occupies. Trapped-low check
    # S; escaped-high check E; CP lower > 0.8 each side; other branches
    # reported only. (Init-defined low-S gating conflicted with F2: S-MSE is
    # a trap indicator, so 95/100 had only ~34% power at p=.933.)
    trap_low = [r for r in low if r["p_E_end"] < 0.5]
    esc_high = [r for r in high if r["p_E_end"] > 0.5]
    fS = [r["mse_S"] <= 1.10 * REF for r in trap_low]
    fE = [r["mse_E"] <= 1.10 * REF for r in esc_high]
    c1s, c1e = cp(fS), cp(fE)
    f1 = dict(passed=bool(c1s and c1e and c1s[0] > 0.8 and c1e[0] > 0.8),
              trapped_S=dict(n=len(fS), ok=int(sum(fS)), ci=c1s),
              escaped_E=dict(n=len(fE), ok=int(sum(fE)), ci=c1e),
              reported=dict(
                  low_S_mean=float(np.mean([r["mse_S"] for r in low])),
                  low_E_mean=float(np.mean([r["mse_E"] for r in low])),
                  high_S_mean=float(np.mean([r["mse_S"] for r in high])),
                  high_E_mean=float(np.mean([r["mse_E"] for r in high]))),
              reference=REF)
    tl = [r["p_E_end"] < 0.5 for r in low]
    c2 = cp(tl)
    f2 = dict(passed=bool(c2[0] > 0.8), frac=float(np.mean(tl)), ci=c2)
    eh = [r["p_E_end"] > 0.5 for r in high]
    c3 = cp(eh)
    f3 = dict(passed=bool(c3[0] > 0.8), frac=float(np.mean(eh)), ci=c3)
    t_lo = sum(r["p_E_end"] < 0.5 for r in R["k64_low"])
    t_hi = sum(r["p_E_end"] < 0.5 for r in R["k64_high"])
    f4 = dict(passed=bool(t_lo <= 1 and t_hi <= 1),
              k64_low_trapped=t_lo, k64_high_trapped=t_hi)
    trap = [r for r in low if r["p_E_end"] < 0.5]
    esc = [r for r in high if r["p_E_end"] > 0.5]
    c5t = cp([r["xtalk"]["x12"] > r["xtalk"]["x23"] for r in trap])
    c5e = cp([r["xtalk"]["x23"] > r["xtalk"]["x12"] for r in esc])
    f5 = dict(passed=bool(c5t and c5e and c5t[0] > 0.5 and c5e[0] > 0.5),
              trap=dict(n=len(trap), ci=c5t), esc=dict(n=len(esc), ci=c5e))
    c6t = cp([r["gap"]["gap_with_delta"] < 0 for r in trap])
    c6e = cp([r["gap"]["gap_with_delta"] > 0 for r in esc])
    f6 = dict(passed=bool(c6t and c6e and c6t[0] > 0.5 and c6e[0] > 0.5),
              trap=dict(n=len(trap), ci=c6t), esc=dict(n=len(esc), ci=c6e))
    ret = {t: {r["seed"]: r["ret"]["deterministic_return"] for r in R[t]}
           for t in R}
    lo_pairs = sorted(set(ret["k2_low"]) & set(ret["k64_low"]))
    hi_pairs = sorted(set(ret["k2_high"]) & set(ret["k64_high"]))
    rank_lo = [ret["k64_low"][s] - ret["k2_low"][s] for s in lo_pairs]
    rank_hi = [ret["k64_high"][s] - ret["k2_high"][s] for s in hi_pairs]
    basin = (float(np.mean(list(ret["k2_high"].values())))
             - float(np.mean(list(ret["k2_low"].values()))))
    ci_a = _boot_mean(rank_lo, 71)
    b_hi = np.array(list(ret["k2_high"].values()))
    b_lo = np.array(list(ret["k2_low"].values()))
    rng = np.random.default_rng(72)
    bs = (b_hi[rng.integers(0, len(b_hi), size=(2000, len(b_hi)))].mean(1)
          - b_lo[rng.integers(0, len(b_lo), size=(2000, len(b_lo)))].mean(1))
    ci_b = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    did = [np.mean(rank_lo) - np.mean(rank_hi)]
    rngd = np.random.default_rng(73)
    rl, rh = np.array(rank_lo), np.array(rank_hi)
    bsd = (rl[rngd.integers(0, len(rl), size=(2000, len(rl)))].mean(1)
           - rh[rngd.integers(0, len(rh), size=(2000, len(rh)))].mean(1))
    ci_c = [float(np.percentile(bsd, 2.5)), float(np.percentile(bsd, 97.5))]
    f7 = dict(passed=bool(ci_a[0] > 0 and ci_b[0] > 0 and ci_c[0] > 0),
              rank_gap_low=dict(mean=float(np.mean(rank_lo)), ci=ci_a),
              basin_gap=dict(mean=basin, ci=ci_b),
              did=dict(mean=float(did[0]), ci=ci_c))
    tail_ok = all(r["tail_drift"] <= 0.1 for r in recs)
    comp_sec = dict(trap_frac=float(np.mean(
        [r["p_E_end"] < 0.5 for r in R["comparator_low"]])))
    allp = all(g["passed"] for g in (f1, f2, f3, f4, f5, f6, f7)) and tail_ok
    return dict(F1=f1, F2=f2, F3=f3, F4=f4, F5=f5, F6=f6, F7=f7,
                tail_ok=tail_ok, comparator_secondary=comp_sec,
                all_pass=bool(allp))


if __name__ == "__main__":
    stamp, jobs = admission_final()
    t0 = time.time()
    with Pool(20) as pool:
        recs = pool.map(cell, jobs)
    raw = json.dumps(recs, sort_keys=True).encode()
    rep = dict(frozen="E3_FINAL_FROZEN.md", records=recs, gates=gates(recs),
               receipt=dict(stamp=stamp,
                            raw_output_sha256=hashlib.sha256(raw).hexdigest(),
                            n_jobs=len(jobs), n_records=len(recs),
                            host=socket.gethostname(),
                            python=platform.python_version(),
                            numpy=np.__version__,
                            started_unix=t0, finished_unix=time.time()))
    os.makedirs("out", exist_ok=True)
    tmp = "out/.f.tmp"
    json.dump(rep, open(tmp, "w"))
    os.replace(tmp, "out/e3_final_report.json")
    print(json.dumps(rep["gates"], indent=1))
