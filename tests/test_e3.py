import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
from e3 import env
from e3.ppo import (TrunkPolicy, run_training, floor_pi_E, make_streams,
                    state_hash)


def test_env_distributions():
    rng = [np.random.default_rng(i) for i in (1, 2)]
    b = (np.random.default_rng(3).random(50000) < 0.3).astype(int)
    mask, Z, sens = env.draw_batch(50000, b, *rng)
    assert sens.shape[1] == env.SENS_DIM
    for br, pairs in env.PAIR_SETS.items():
        sel = b == br
        for (p, q) in pairs:
            f = ((mask[sel][:, p] == 1) & (mask[sel][:, q] == 1)).mean()
            assert 0.45 < f < 0.55


def test_analytic_oracle():
    """Simulate the analytic-optimal responder; its empirical masked MSE must
    match 2 s2/(1+s2)."""
    s = 0.30
    rng = np.random.default_rng(5)
    b = np.zeros(400000, dtype=int)
    mask, Z, sens = env.draw_batch(len(b), b, rng, rng, sigma_obs=s)
    a = sens[:, :3] / (1 + s ** 2)
    mse = (((a - Z) ** 2) * mask).sum(1).mean()
    assert abs(mse - env.oracle_recon_mse(s)) < 2e-3
    assert abs(env.oracle_recon_mse(0.30) - 0.16514) < 1e-4


def test_theory_floor_mapping():
    logits = torch.tensor([[0.0, 100.0], [0.0, -100.0]])
    p = floor_pi_E(logits, 0.05)
    assert abs(p[0].item() - 0.95) < 1e-4
    assert abs(p[1].item() - 0.05) < 1e-4


def test_sensory_no_bypass_and_branch_bypass():
    net = TrunkPolicy(2, init_rng=np.random.default_rng(0))
    sens = torch.randn(4, env.SENS_DIM)
    bo = torch.zeros(4, 2); bo[:, 1] = 1.0
    h = torch.tanh(net.enc(sens))
    null = h - h @ net.proj
    # sensory null-space perturbation cannot reach the heads...
    o1 = net.trunk(torch.cat([h @ net.proj, bo], -1))
    o2 = net.trunk(torch.cat([(h + 3 * null) @ net.proj, bo], -1))
    assert torch.allclose(o1, o2, atol=1e-6)
    # ...but branch identity can (bypass exists exactly for the context)
    bo2 = torch.zeros(4, 2); bo2[:, 0] = 1.0
    o3 = net.trunk(torch.cat([h @ net.proj, bo2], -1))
    assert not torch.allclose(o1, o3, atol=1e-4)


def test_forced_mode_freezes_root_actor():
    seed = 11
    net, _, _ = run_training(k_proj=2, seed=seed, updates=3, batch=256,
                             minibatch=128, forced_branch=1)
    S = make_streams(seed)
    torch.manual_seed(int(S["init"].integers(2 ** 31)))
    ref = TrunkPolicy(2, init_rng=S["init"])
    with torch.no_grad():
        ref.root_head.bias.copy_(torch.tensor(
            [0.0, float(np.log(0.5 / 0.5))]))
    # root head must be untouched by forced training up to value-shared trunk
    assert torch.allclose(net.root_head.weight, ref.root_head.weight)
    assert torch.allclose(net.root_head.bias, ref.root_head.bias)


def test_bound_streams_full_state_hash():
    torch.manual_seed(111); np.random.seed(222)
    n1, l1, _ = run_training(k_proj=2, seed=7, updates=4, batch=256,
                             minibatch=128, log_every=2)
    torch.manual_seed(999); np.random.seed(888)
    n2, l2, _ = run_training(k_proj=2, seed=7, updates=4, batch=256,
                             minibatch=128, log_every=2)
    assert l1 == l2
    assert state_hash(n1) == state_hash(n2)


def _fake_records(*, bad_high_reversal=False, one_cond_17=False,
                  csplit_neg=False, control_unstable=False):
    recs = []
    for k in (2, 64):
        for br, name in ((0, "S"), (1, "E")):
            for i, s in enumerate(range(5000, 5020)):
                ratio = 0.9
                if one_cond_17 and k == 2 and br == 0 and i < 3:
                    ratio = 0.5                      # 17/20 in one condition
                comp = {"S": dict(ratio_to_oracle=ratio, mse=0.2,
                                  oracle_mse=0.165),
                        "E": dict(ratio_to_oracle=ratio, mse=0.2,
                                  oracle_mse=0.165)}
                recs.append(dict(kind="forced", seed=s, k=k, forced=br,
                                 competence=comp,
                                 xtalk=dict(x12=0.2, x13=0.1, x23=0.05),
                                 gram={}))
    for k in (2, 64):
        for p0 in (0.1, 0.5, 0.9):
            for s in range(5000, 5020):
                if k == 2:
                    pe = 0.02 if p0 == 0.1 else (0.98 if p0 == 0.9 else 0.5)
                else:
                    pe = (0.05 if csplit_neg else 0.9) if p0 == 0.9 else \
                         (0.95 if csplit_neg else 0.9) if p0 == 0.1 else 0.9
                x = dict(x12=0.3, x13=0.1, x23=0.05)
                if p0 == 0.9 and k == 2:
                    x = dict(x12=0.3, x13=0.1, x23=0.05) if bad_high_reversal \
                        else dict(x12=0.05, x13=0.1, x23=0.3)
                recs.append(dict(kind="free", seed=s, k=k, p0=p0, p_E_end=pe,
                                 tail_drift=0.5 if (control_unstable and
                                                    k == 64) else 0.02,
                                 xtalk=x, gram={}, gap={}, ret_det={},
                                 ret_smp={}))
    return recs


def test_gate_reducer_synthetic():
    from e3.pilot import gates
    g = gates(_fake_records())
    assert g["all_pass"], g
    assert not gates(_fake_records(bad_high_reversal=True))["P3"]["passed"]
    assert not gates(_fake_records(one_cond_17=True))["P1"]["passed"]
    assert not gates(_fake_records(csplit_neg=True))["P2"]["passed"]
    assert not gates(_fake_records(control_unstable=True))["P2"]["passed"]


def test_pilot_schema_e2e():
    """1-seed micro run through the real cells: schema only, no science."""
    from e3 import pilot
    f = pilot.forced_cell.__wrapped__ if hasattr(pilot.forced_cell,
                                                 "__wrapped__") else None
    import e3.pilot as P
    import e3.ppo as ppo
    orig = ppo.run_training
    def tiny(**kw):
        kw.update(updates=1, batch=64, minibatch=32)
        return orig(**kw)
    ppo.run_training = tiny
    try:
        r1 = P.forced_cell((5000, 2, 0))
        r2 = P.free_cell((5000, 2, 0.1))
    finally:
        ppo.run_training = orig
    for r, keys in ((r1, ("competence", "xtalk", "gram")),
                    (r2, ("p_E_end", "tail_drift", "xtalk", "gap",
                          "ret_det", "ret_smp"))):
        for k in keys:
            assert k in r, k


def test_exact_binom_ci():
    from e3.pilot import _exact_binom_ci
    ci = _exact_binom_ci([1] * 5)
    assert abs(ci[0] - 0.025 ** 0.2) < 1e-6 and ci[1] == 1.0
    assert ci[0] < 0.5                       # n=5 all-success cannot pass
    ci6 = _exact_binom_ci([1] * 6)
    assert ci6[0] > 0.5                      # n=6 all-success can
    ci_mixed = _exact_binom_ci([1] * 18 + [0] * 2)
    assert 0.5 < ci_mixed[0] < 0.9 < ci_mixed[1] <= 1.0


def test_admission_refusals(tmp_path=None):
    import tempfile, json, hashlib, shutil
    from e3 import pilot
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as td:
        os.chdir(td)
        try:
            # missing stamp
            try:
                pilot.admission(); raise AssertionError("no refusal")
            except SystemExit as e:
                assert "missing" in str(e)
            # build a consistent fake deployment
            os.makedirs("e3"); os.makedirs("tests"); os.makedirs("out")
            open("e3/x.py", "w").write("A = 1\n")
            open("tests/test_e3.py", "w").write("# t\n")
            open("E3_PREREGISTRATION.deployed.md", "w").write("SPEC\n")
            def sha(f):
                return hashlib.sha256(open(f, "rb").read()).hexdigest()
            stamp = {"commit": "c", "dirty_files": 0,
                     "spec_sha256": sha("E3_PREREGISTRATION.deployed.md"),
                     "manifest_sha256": "m",
                     "payload_sha256": {"e3/x.py": sha("e3/x.py"),
                                        "tests/test_e3.py":
                                        sha("tests/test_e3.py")}}
            json.dump(stamp, open("deploy_stamp.json", "w"))
            out = pilot.admission()
            assert out[0]["commit"] == "c"
            # dirty stamp refused
            stamp["dirty_files"] = 3
            json.dump(stamp, open("deploy_stamp.json", "w"))
            try:
                pilot.admission(); raise AssertionError("no refusal")
            except SystemExit as e:
                assert "dirty" in str(e)
            stamp["dirty_files"] = 0
            json.dump(stamp, open("deploy_stamp.json", "w"))
            # tampered SPEC refused (implementation checks it; test was
            # missing this case per final audit)
            open("E3_PREREGISTRATION.deployed.md", "w").write("SPEC2\n")
            try:
                pilot.admission(); raise AssertionError("no refusal")
            except SystemExit as e:
                assert "spec hash mismatch" in str(e)
            open("E3_PREREGISTRATION.deployed.md", "w").write("SPEC\n")
            # tampered payload refused
            open("e3/x.py", "w").write("A = 2\n")
            try:
                pilot.admission(); raise AssertionError("no refusal")
            except SystemExit as e:
                assert "hash mismatch" in str(e)
            open("e3/x.py", "w").write("A = 1\n")
            # census mismatch refused (extra file)
            open("e3/y.py", "w").write("B = 1\n")
            try:
                pilot.admission(); raise AssertionError("no refusal")
            except SystemExit as e:
                assert "census" in str(e)
            os.remove("e3/y.py")
            # existing report refused
            open("out/e3_pilot_report.json", "w").write("{}")
            try:
                pilot.admission(); raise AssertionError("no refusal")
            except SystemExit as e:
                assert "refusing overwrite" in str(e)
        finally:
            os.chdir(cwd)


def test_exact_job_census():
    from e3.pilot import expected_jobs
    jf, jr = expected_jobs()
    assert len(jf) == 80 and len(jr) == 120
    assert len(set(jf + jr)) == 200


def test_pilot2_census():
    from e3.pilot2 import expected_jobs
    J = expected_jobs()
    keys = [(j[0], j[2], j[3], j[4], j[5]) for j in J]
    assert len(J) == 165 and len(set(keys)) == 165
    from collections import Counter
    c = Counter(j[0] for j in J)
    assert c == {"main_low": 30, "main_ctrl": 30, "comparator_low": 30,
                 "main_high": 30, "sec_low": 15, "sec_ctrl": 15,
                 "sec_high": 15}


def test_final_census_and_admission_refusals():
    from e3.final_cohort import expected_jobs, admission_final
    J = expected_jobs()
    assert len(J) == 430 and len(set(J)) == 430
    from collections import Counter
    c = Counter(j[0] for j in J)
    assert c == {"k2_low": 100, "k64_low": 100, "k2_high": 100,
                 "k64_high": 100, "comparator_low": 30}
    import tempfile, json as js, hashlib
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as td:
        os.chdir(td)
        try:
            try:
                admission_final(); raise AssertionError("no refusal")
            except SystemExit as e:
                assert "missing" in str(e)
            os.makedirs("e3"); os.makedirs("tests"); os.makedirs("out")
            open("e3/x.py", "w").write("A = 1\n")
            open("tests/test_e3.py", "w").write("# t\n")
            open("E3_PREREGISTRATION.deployed.md", "w").write("SPEC\n")
            open("E3_FINAL_FROZEN.deployed.md", "w").write("FINAL\n")
            def sha(f):
                return hashlib.sha256(open(f, "rb").read()).hexdigest()
            st = {"commit": "c", "dirty_files": 0,
                  "spec_sha256": sha("E3_PREREGISTRATION.deployed.md"),
                  "final_spec_sha256": sha("E3_FINAL_FROZEN.deployed.md"),
                  "payload_sha256": {"e3/x.py": sha("e3/x.py"),
                                     "tests/test_e3.py":
                                     sha("tests/test_e3.py")}}
            js.dump(st, open("deploy_stamp.json", "w"))
            assert admission_final()[0]["commit"] == "c"
            for mut, needle in (
                (lambda: st.update(dirty_files=2), "dirty"),
                (lambda: open("E3_FINAL_FROZEN.deployed.md", "w").write("X"),
                 "FINAL spec hash"),
                (lambda: open("e3/x.py", "w").write("A = 9\n"),
                 "hash mismatch"),
                (lambda: open("e3/y.py", "w").write("B\n"), "census"),
            ):
                mut(); js.dump(st, open("deploy_stamp.json", "w"))
                try:
                    admission_final(); raise AssertionError("no refusal")
                except SystemExit as e:
                    assert needle in str(e), (needle, str(e))
                # restore
                st["dirty_files"] = 0
                open("E3_FINAL_FROZEN.deployed.md", "w").write("FINAL\n")
                open("e3/x.py", "w").write("A = 1\n")
                if os.path.exists("e3/y.py"):
                    os.remove("e3/y.py")
                js.dump(st, open("deploy_stamp.json", "w"))
            open("out/e3_final_report.json", "w").write("{}")
            try:
                admission_final(); raise AssertionError("no refusal")
            except SystemExit as e:
                assert "refusing overwrite" in str(e)
        finally:
            os.chdir(cwd)


def test_final_gates_synthetic():
    from e3.final_cohort import gates, REF
    def rec(tag, seed, pe, mS=0.24, mE=0.74, x12=0.3, x23=0.05, g=-0.2, r=-0.25):
        return dict(tag=tag, seed=seed, k=2, q0=0.05, rls=0.3, p_E_end=pe,
                    tail_drift=0.02, mse_S=mS, mse_E=mE,
                    xtalk=dict(x12=x12, x13=0.1, x23=x23),
                    gap=dict(gap_with_delta=g, gap_no_delta=g - 0.15),
                    ret=dict(p_E=pe, deterministic_return=r))
    def build(trap_n=95, minflip=False):
        recs = []
        for i, s in enumerate(range(6000, 6100)):
            trapped = i < trap_n
            mS = 0.24 if not (minflip and not trapped) else 0.30
            mE = 0.74 if not (minflip and not trapped) else 0.20
            # pilot-v2 observed structure: escaped-low runs abandon S, so
            # their S-MSE degrades (twelfth review: synthetic tests must
            # replicate observed structure, not idealized structure)
            mS_row = mS if trapped else 0.85
            mE_row = mE if trapped else 0.24
            recs.append(rec("k2_low", s, 0.02 if trapped else 0.98, mS=mS_row,
                            mE=mE_row, x12=0.3 if trapped else 0.05,
                            x23=0.05 if trapped else 0.3,
                            g=-0.2 if trapped else 0.8,
                            r=-0.258 if trapped else -0.127))
            recs.append(rec("k64_low", s, 0.98, r=-0.026))
        for s in range(6100, 6200):
            # escaped-high occupies E: E-MSE good, S degraded (observed str.)
            recs.append(rec("k2_high", s, 0.98, mS=0.85, mE=0.24,
                            x12=0.05, x23=0.3, g=0.8, r=-0.05))
            recs.append(rec("k64_high", s, 0.98, r=-0.02))
        for s in range(6000, 6030):
            recs.append(rec("comparator_low", s, 0.5, r=-0.1))
        return recs
    g = gates(build(trap_n=95))
    assert g["F2"]["passed"] and g["all_pass"], g["F2"]
    g88 = gates(build(trap_n=88))          # below n=100 threshold (needs 89)
    assert not g88["F2"]["passed"]
    g89 = gates(build(trap_n=89))
    assert g89["F2"]["passed"]
    # F1 endpoint-conditioned: trapped judged on S, escaped-high on E;
    # escaped-low rows with degraded S must NOT sink F1 (the old init-defined
    # gate would have failed here: only 89/100 S-competent at trap_n=89)
    g89b = gates(build(trap_n=89))
    assert g89b["F1"]["passed"] and g89b["F2"]["passed"]
    assert g89b["F1"]["trapped_S"]["ok"] == 89
    # and a trapped run with bad S-MSE does sink F1
    recs_bad = build(trap_n=95)
    nbad = 0
    for r_ in recs_bad:
        if r_["tag"] == "k2_low" and r_["p_E_end"] < 0.5 and nbad < 25:
            r_["mse_S"] = 0.9; nbad += 1
    assert not gates(recs_bad)["F1"]["passed"]
    # DiD must fail if low and high rank gaps are equal
    recs = build(trap_n=95)
    for r_ in recs:
        if r_["tag"] == "k64_high":
            # RAISE k64_high so rank_gap_high ~= rank_gap_low -> DiD <= 0
            r_["ret"]["deterministic_return"] = r_["ret"][
                "deterministic_return"] + 0.25
    assert not gates(recs)["F7"]["passed"]
