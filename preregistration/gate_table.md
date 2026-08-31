# Gate table (rebuilt from raw)

G1 dim3: 779/779 -> PASS
G2 (m=20 grid, 33 excluded): 746/746 -> PASS
G3 m=80: max|err|=0.0044 -> PASS
G4 d=0.2: tau*=1.724 PASS; sinks PASS
G4 d=0.5: tau*=1.393 PASS; sinks PASS
G4 d=0.8: tau*=1.115 PASS; sinks PASS
G5 eps=0.05: |err|=0.0003 -> PASS
G5 eps=0.1: |err|=0.0001 -> PASS
G5 eps=0.15: |err|=0.0006 -> PASS
G5 eps=0.2: |err|=0.0006 -> PASS
G6 d=0.3: |lam*-crit|=0.1690 -> FAIL (registered); endpoint-gap subcheck PASS (low 1.1e-16, high 6.7e-16)
G6 d=0.6: |lam*-crit|=0.1007 -> FAIL (registered); endpoint-gap subcheck FAIL (low 1.2e+00, high 1.2e-15)
G7 pbar=0.2: |err|=0.0122 -> PASS
G7 pbar=0.5: |err|=0.0046 -> PASS
G6' d=0.3: mono=True, lam_hat=2.824 vs crit=2.798, s_end=0.0052 -> PASS (append-only; does not overwrite G6)
G6' d=0.6: mono=True, lam_hat=1.526 vs crit=1.523, s_end=0.0046 -> PASS (append-only; does not overwrite G6)
repeated-solve control (observed): 42/42 agreement
