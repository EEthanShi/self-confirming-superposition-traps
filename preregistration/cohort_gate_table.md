# Cohort gate table (rebuilt from raw)

failed runs: 0/9900
C1 fullrank3 trapped 0.0000 -> PASS
C2 constr2 d=0.15 eps=0.05: b=0.4720 CI[0.4695,0.4756] pred 0.4779 -> PASS
C2 constr2 d=0.15 eps=0.15: b=0.4683 CI[0.4669,0.4699] pred 0.4715 -> PASS
C3 unconstr2 d=0.15 eps=0.05: b=0.4876 CI[0.4806,0.4927] pred 0.4897 -> PASS
C3 unconstr2 d=0.15 eps=0.15: b=0.4835 CI[0.4727,0.4906] pred 0.4867 -> PASS
C4 constr2 d=0.15: shift CI[0.0016,0.0069] pred 0.0063 -> DIRECTION-ONLY GATE (frozen tol wider than signal) direction+
C4 unconstr2 d=0.15: shift CI[0.0007,0.0106] pred 0.003 -> DIRECTION-ONLY GATE (frozen tol wider than signal) direction+
C2 constr2 d=0.3 eps=0.05: b=0.4490 CI[0.4477,0.4509] pred 0.4561 -> PASS
C2 constr2 d=0.3 eps=0.15: b=0.4468 CI[0.4452,0.4480] pred 0.4436 -> PASS
C3 unconstr2 d=0.3 eps=0.05: b=0.4717 CI[0.4663,0.4753] pred 0.4794 -> PASS
C3 unconstr2 d=0.3 eps=0.15: b=0.4595 CI[0.4499,0.4684] pred 0.4736 -> PASS
C4 constr2 d=0.3: shift CI[0.0009,0.0042] pred 0.0125 -> DIRECTION-ONLY GATE (frozen tol wider than signal) direction+ [CI excludes point pred]
C4 unconstr2 d=0.3: shift CI[0.0047,0.0207] pred 0.0059 -> DIRECTION-ONLY GATE (frozen tol wider than signal) direction+
C2 constr2 d=0.6 eps=0.05: b=0.4185 CI[0.4169,0.4207] pred 0.4154 -> PASS
C2 constr2 d=0.6 eps=0.15: b=0.3892 CI[0.3879,0.3901] pred 0.3913 -> PASS
C3 unconstr2 d=0.6 eps=0.05: b=0.3126 CI[0.3106,0.3150] pred 0.3247 -> DEV-CHECK (consistent)
C3 unconstr2 d=0.6 eps=0.15: UNRESOLVED (grid-edge; 99% non-finite bootstrap)
C4 constr2 d=0.6: shift CI[0.0274,0.0315] pred 0.0242 -> DIRECTION-ONLY GATE (frozen tol wider than signal) direction+ [CI excludes point pred]
C4 unconstr2 d=0.6: shift CI[0.0387,0.0423] pred 0.0501 -> DEV-CHECK direction+ [CI excludes point pred]
exploratory band (outcome-informed, no gates):
  d=0.4: constr2=0.44999999999999996  unconstr2=0.45399999999999996  fullrank3=edge
  d=0.45: constr2=0.44  unconstr2=0.44  fullrank3=edge
  d=0.5: constr2=0.42157894736842105  unconstr2=0.43266666666666664  fullrank3=edge
  d=0.55: constr2=0.4124  unconstr2=0.385  fullrank3=edge
