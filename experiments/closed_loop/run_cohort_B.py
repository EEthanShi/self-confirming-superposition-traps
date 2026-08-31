"""Block B execution per E2_COHORT_B_FROZEN.md (same-commit freeze)."""
import sys, json, time, os, socket, platform
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, ".")
from e2.cohort import run_cell, ARMS

tg = json.load(open("cohort_targets_B.json"))
GRIDS = {float(k): v for k, v in tg["grids"].items()}
SEEDS = list(range(3000, 3050))

def cell(a):
    return run_cell(*a[:6], floor_eps=a[6])

if __name__ == "__main__":
    t0 = time.time()
    jobs = []
    for di, d in enumerate((0.15, 0.30, 0.60)):
        for pi, p0 in enumerate(GRIDS[d]):
            for s in SEEDS:
                for arm in ARMS:
                    for eps in (0.05, 0.25):
                        jobs.append((arm, d, p0, di, pi, s, eps))
    print("cells:", len(jobs), flush=True)
    with Pool(22) as pool:
        recs = pool.map(cell, jobs)
    json.dump(dict(frozen="E2_COHORT_B_FROZEN.md", block="B", records=recs,
                   receipt=dict(host=socket.gethostname(),
                                python=platform.python_version(),
                                numpy=np.__version__,
                                git_head=os.environ.get("GIT_HEAD", "unset"),
                                started_unix=t0, finished_unix=time.time())),
              open("out/cohort_blockB.json", "w"))
    print("BLOCK B DONE", len(recs), round(time.time()-t0), "s", flush=True)
