"""E3 Pilot-v2 per E3_PILOT2_FROZEN.md (same-commit freeze). Admission-gated,
branch-separated competence, both-side mediator/gap, CRN-paired capacity
return deficit, tail stability. Usage: python -m e3.pilot2"""
import json, os, socket, platform, time, hashlib
import numpy as np
from multiprocessing import Pool
from scipy.stats import beta

SIGMA, EPS = 0.30, 0.05
MAIN = dict(delta=0.15, rls=0.3)
SECD = dict(delta=0.30, rls=0.1)
REF = None  # set in main from env


def _pin():
    import torch
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"


def cell(args):
    _pin()
    from .ppo import run_training
    from . import evaluators as ev
    tag, cfg, seed, k, q0, rls = args
    net, logs, S = run_training(k_proj=k, delta=cfg["delta"], seed=seed,
                                p0=q0, updates=400, sigma_obs=SIGMA,
                                floor_eps=EPS, root_lr_scale=rls,
                                log_every=1)
    cut = logs[-1]["update"] * 3 // 4
    tail = [l["p_E"] for l in logs if l["update"] >= cut]
    comp = ev.competence(net, S["eval"], sigma_obs=SIGMA)
    return dict(tag=tag, delta=cfg["delta"], rls=rls, seed=seed, k=k, q0=q0,
                p_E_end=logs[-1]["p_E"],
                tail_drift=float(max(tail) - min(tail)),
                mse_S=comp["S"]["mse"], mse_E=comp["E"]["mse"],
                xtalk=ev.crosstalk(net, S["eval"], sigma_obs=SIGMA),
                gap=ev.forced_gap(net, S["eval"], sigma_obs=SIGMA,
                                  delta=cfg["delta"]),
                ret_det=ev.deployment_return(net, S["eval"], sigma_obs=SIGMA,
                                             delta=cfg["delta"],
                                             floor_eps=EPS),
                ret_smp=ev.deployment_return(net, S["eval"], sigma_obs=SIGMA,
                                             delta=cfg["delta"],
                                             floor_eps=EPS, sampled=True))


def expected_jobs():
    J = []
    for s in range(5200, 5230):
        J.append(("main_low", MAIN, s, 2, 0.05, MAIN["rls"]))
        J.append(("main_ctrl", MAIN, s, 64, 0.05, MAIN["rls"]))
        J.append(("comparator_low", MAIN, s, 2, 0.05, 1.0))
    for s in range(5230, 5260):
        J.append(("main_high", MAIN, s, 2, 0.9, MAIN["rls"]))
    for s in range(5260, 5275):
        J.append(("sec_low", SECD, s, 2, 0.05, SECD["rls"]))
        J.append(("sec_ctrl", SECD, s, 64, 0.05, SECD["rls"]))
    for s in range(5275, 5290):
        J.append(("sec_high", SECD, s, 2, 0.9, SECD["rls"]))
    return J


def cp(flags, alpha=0.05):
    n, k = len(flags), int(sum(flags))
    if n == 0:
        return None
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return [lo, hi]


def gates(recs, ref):
    R = {t: [r for r in recs if r["tag"] == t]
         for t in ("main_low", "main_high", "main_ctrl", "comparator_low")}
    low, high, ctrl = R["main_low"], R["main_high"], R["main_ctrl"]
    fav = [min(r["mse_S"], r["mse_E"]) for r in low]
    disfav = [max(r["mse_S"], r["mse_E"]) for r in low]
    q1n = sum(f <= 1.10 * ref for f in fav)
    q1 = dict(passed=bool(q1n >= 28), detail=f"{q1n}/{len(fav)}",
              favored_mean=float(np.mean(fav)),
              disfavored_mean=float(np.mean(disfav)), reference=ref)
    tl = [r["p_E_end"] < 0.5 for r in low]
    q2ci = cp(tl)
    q2 = dict(passed=bool(q2ci[0] > 0.8), frac=float(np.mean(tl)), ci=q2ci)
    eh = [r["p_E_end"] > 0.5 for r in high]
    q3ci = cp(eh)
    q3 = dict(passed=bool(q3ci[0] > 0.8), frac=float(np.mean(eh)), ci=q3ci)
    ct = sum(r["p_E_end"] < 0.5 for r in ctrl)
    q4 = dict(passed=bool(ct <= 1), trapped=ct, n=len(ctrl))
    trap = [r for r in low if r["p_E_end"] < 0.5]
    esc = [r for r in high if r["p_E_end"] > 0.5]
    m_t = [r["xtalk"]["x12"] > r["xtalk"]["x23"] for r in trap]
    m_e = [r["xtalk"]["x23"] > r["xtalk"]["x12"] for r in esc]
    c5t, c5e = cp(m_t), cp(m_e)
    q5 = dict(passed=bool(c5t and c5e and c5t[0] > 0.5 and c5e[0] > 0.5),
              trap=dict(n=len(m_t), ci=c5t), esc=dict(n=len(m_e), ci=c5e))
    g_t = [r["gap"]["gap_with_delta"] < 0 for r in trap]
    g_e = [r["gap"]["gap_with_delta"] > 0 for r in esc]
    c6t, c6e = cp(g_t), cp(g_e)
    q6 = dict(passed=bool(c6t and c6e and c6t[0] > 0.5 and c6e[0] > 0.5),
              trap=dict(n=len(g_t), ci=c6t), esc=dict(n=len(g_e), ci=c6e))
    k2 = {r["seed"]: r["ret_det"]["deterministic_return"] for r in low}
    k64 = {r["seed"]: r["ret_det"]["deterministic_return"] for r in ctrl}
    deficit = np.array([k64[s] - k2[s] for s in sorted(set(k2) & set(k64))])
    rng = np.random.default_rng(4242)
    bs = deficit[rng.integers(0, len(deficit),
                              size=(2000, len(deficit)))].mean(axis=1)
    q7ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    q7 = dict(passed=bool(q7ci[0] > 0), mean=float(deficit.mean()), ci=q7ci)
    drift_ok = all(r["tail_drift"] <= 0.1 for r in recs
                   if r["tag"].startswith(("main", "comparator")))
    comparator = dict(trap_frac=float(np.mean(
        [r["p_E_end"] < 0.5 for r in R["comparator_low"]])))
    sec = {t: float(np.mean([r["p_E_end"] < 0.5
                             for r in recs if r["tag"] == t]))
           for t in ("sec_low", "sec_high", "sec_ctrl")}
    allp = all(q["passed"] for q in (q1, q2, q3, q4, q5, q6, q7)) and drift_ok
    return dict(Q1=q1, Q2=q2, Q3=q3, Q4=q4, Q5=q5, Q6=q6, Q7=q7,
                tail_ok=drift_ok, comparator_secondary=comparator,
                secondary_config=sec, all_pass=bool(allp))


def admission2(report_path="out/e3_pilot2_report.json"):
    """Pilot-v2 admission: same five refusals as pilot.admission, own census
    (165 unique jobs). No monkeypatching of audited gates."""
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
        raise SystemExit("ADMISSION: payload census mismatch")
    for f, h in payload.items():
        if sha(f) != h:
            raise SystemExit(f"ADMISSION: payload hash mismatch {f}")
    jobs = expected_jobs()
    keys = [(j[0], j[2], j[3], j[4], j[5]) for j in jobs]
    if len(jobs) != 165 or len(set(keys)) != 165:
        raise SystemExit("ADMISSION: job census not exactly 165 unique")
    if os.path.exists(report_path):
        raise SystemExit("ADMISSION: report exists; refusing overwrite")
    return stamp, jobs


if __name__ == "__main__":
    from . import env as E
    stamp, jobs = admission2()
    ref = E.rank2_reference_mse(SIGMA)
    t0 = time.time()
    with Pool(20) as pool:
        recs = pool.map(cell, jobs)
    raw = json.dumps(recs, sort_keys=True).encode()
    rep = dict(frozen="E3_PILOT2_FROZEN.md", records=recs,
               gates=gates(recs, ref),
               receipt=dict(stamp=stamp,
                            raw_output_sha256=hashlib.sha256(raw).hexdigest(),
                            n_jobs=len(jobs), n_records=len(recs),
                            host=socket.gethostname(),
                            python=platform.python_version(),
                            numpy=np.__version__,
                            started_unix=t0, finished_unix=time.time()))
    os.makedirs("out", exist_ok=True)
    tmp = "out/.p2.tmp"
    json.dump(rep, open(tmp, "w"))
    os.replace(tmp, "out/e3_pilot2_report.json")
    print(json.dumps(rep["gates"], indent=1))
