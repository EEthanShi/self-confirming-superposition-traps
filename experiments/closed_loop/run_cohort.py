"""Cohort execution per E2_COHORT_FROZEN.md (frozen in the same commit)."""
import sys, json, time, os, socket, platform
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, ".")
from e2.cohort import run_cell, ARMS

tg = json.load(open("cohort_targets.json"))
GRIDS = {float(k): v for k, v in tg["grids"].items()}
DELTAS = [0.15, 0.30, 0.60]
BAND = [0.40, 0.45, 0.50, 0.55]
SEEDS_A = list(range(2000, 2050))
SEEDS_X = list(range(2500, 2530))

def cell(args):
    return run_cell(*args[:6], floor_eps=args[6])

def receipt(tag, t0):
    return dict(tag=tag, host=socket.gethostname(),
                python=platform.python_version(), numpy=np.__version__,
                git_head=os.environ.get("GIT_HEAD", "unset"),
                started_unix=t0, finished_unix=time.time())

if __name__ == "__main__":
    t0 = time.time()
    jobs = []
    for di, d in enumerate(DELTAS):
        for pi, p0 in enumerate(GRIDS[d]):
            for s in SEEDS_A:
                for arm in ARMS:
                    for eps in (0.05, 0.15):
                        jobs.append((arm, d, p0, di, pi, s, eps))
    with Pool(22) as pool:
        recs = pool.map(cell, jobs)
    json.dump(dict(frozen="E2_COHORT_FROZEN.md", block="A", records=recs,
                   receipt=receipt("confirmatory blockA", t0)),
              open("out/cohort_blockA.json", "w"))
    print("blockA done", len(recs), round(time.time()-t0), "s", flush=True)
    t1 = time.time()
    jobs = []
    for di, d in enumerate(BAND):
        gr = np.round(np.linspace(0.15, 0.55, 11), 4)
        for pi, p0 in enumerate(gr):
            for s in SEEDS_X:
                for arm in ARMS:
                    jobs.append((arm, d, float(p0), 10+di, pi, s, 0.05))
    with Pool(22) as pool:
        recs = pool.map(cell, jobs)
    json.dump(dict(frozen="E2_COHORT_FROZEN.md", block="exploratory",
                   records=recs, receipt=receipt("band exploratory", t1)),
              open("out/cohort_exploratory.json", "w"))
    print("COHORT ALL DONE", len(recs), round(time.time()-t0), "s", flush=True)
