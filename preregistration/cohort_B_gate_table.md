# Block B gate table (rebuilt from raw)

records 35100, failed 0
B1 capacity control trapped 0.0000 -> PASS
B2 constr2 d=0.15 e=0.05: b=0.4786 err CI [-0.0005,+0.0013] margin ±0.02 -> PASS
B2 constr2 d=0.15 e=0.25: b=0.4598 err CI [-0.0023,-0.0002] margin ±0.02 -> PASS
B3 unconstr2 d=0.15 e=0.05: b=0.4822 err CI [-0.0096,+0.0077] margin ±0.03 -> PASS
B3 unconstr2 d=0.15 e=0.25: b=0.4762 err CI [-0.0100,+0.0158] margin ±0.03 -> PASS
B4 constr2 d=0.15: shift CI [0.0177,0.0209] (pred 0.0177, secondary) -> PASS
B4 unconstr2 d=0.15: shift CI [0.0001,0.0198] (pred 0.0083, secondary) -> PASS
B2 constr2 d=0.3 e=0.05: b=0.4561 err CI [-0.0022,+0.0005] margin ±0.02 -> PASS
B2 constr2 d=0.3 e=0.25: b=0.4182 err CI [-0.0050,+0.0000] margin ±0.02 -> PASS
B3 unconstr2 d=0.3 e=0.05: b=0.4668 err CI [-0.0150,-0.0014] margin ±0.03 -> PASS
B3 unconstr2 d=0.3 e=0.25: b=0.4480 err CI [-0.0230,+0.0006] margin ±0.03 -> PASS
B4 constr2 d=0.3: shift CI [0.0345,0.0399] (pred 0.0351, secondary) -> PASS
B4 unconstr2 d=0.3: shift CI [0.0046,0.0311] (pred 0.0164, secondary) -> PASS
B2 constr2 d=0.6 e=0.05: b=0.4124 err CI [-0.0043,-0.0014] margin ±0.02 -> PASS
B2 constr2 d=0.6 e=0.25: b=0.3433 err CI [-0.0065,-0.0029] margin ±0.02 -> PASS
B3 unconstr2 d=0.6 e=0.05: b=0.3153 err CI [-0.0109,-0.0056] margin ±0.03 -> PASS
B3 unconstr2 d=0.6 e=0.25: b=0.1859 err CI [-0.0014,+0.0038] margin ±0.03 -> PASS
B4 constr2 d=0.6: shift CI [0.0670,0.0716] (pred 0.0676, secondary) -> PASS
B4 unconstr2 d=0.6: shift CI [0.1269,0.1344] (pred 0.1403, secondary) -> PASS
B5 mediator: trapped g12>g23 1.000, escaped g23>g12 0.999 -> PASS
B6 gap link: trapped gap<0 0.992, escaped gap>0 1.000 -> PASS
B7 constr2 d=0.15: ret(trap)-ret(escape) CI hi +0.0022 -> FAIL
B7 unconstr2 d=0.15: ret(trap)-ret(escape) CI hi -0.0061 -> PASS
B7 constr2 d=0.3: ret(trap)-ret(escape) CI hi +0.0011 -> FAIL
B7 unconstr2 d=0.3: ret(trap)-ret(escape) CI hi -0.0147 -> PASS
B7 constr2 d=0.6: ret(trap)-ret(escape) CI hi +0.0013 -> FAIL
B7 unconstr2 d=0.6: ret(trap)-ret(escape) CI hi -0.0191 -> PASS
--- stratified re-report (delta x arm x eps, seed-clustered; supersedes pooled B5-B7 for interpretation, registered verdicts kept) ---
  constr2 d=0.15 e=0.05: n_trap=987/n_esc=963 B5 trap 1.000/esc 1.000  B6 trap 1.000/esc 1.000  B7 ret diff CI [-0.0017,+0.0028]
  constr2 d=0.15 e=0.25: n_trap=497/n_esc=1453 B5 trap 1.000/esc 1.000  B6 trap 1.000/esc 1.000  B7 ret diff CI [-0.0023,+0.0026]
  unconstr2 d=0.15 e=0.05: n_trap=1138/n_esc=812 B5 trap 1.000/esc 1.000  B6 trap 1.000/esc 1.000  B7 ret diff CI [-0.0020,+0.0022]
  unconstr2 d=0.15 e=0.25: n_trap=913/n_esc=1037 B5 trap 1.000/esc 1.000  B6 trap 1.000/esc 1.000  B7 ret diff CI [-0.0009,+0.0028]
  constr2 d=0.3 e=0.05: n_trap=991/n_esc=959 B5 trap 1.000/esc 1.000  B6 trap 1.000/esc 1.000  B7 ret diff CI [-0.0033,+0.0016]
  constr2 d=0.3 e=0.25: n_trap=298/n_esc=1652 B5 trap 1.000/esc 1.000  B6 trap 1.000/esc 1.000  B7 ret diff CI [-0.0047,+0.0024]
  unconstr2 d=0.3 e=0.05: n_trap=1175/n_esc=775 B5 trap 1.000/esc 1.000  B6 trap 1.000/esc 1.000  B7 ret diff CI [-0.0030,+0.0015]
  unconstr2 d=0.3 e=0.25: n_trap=767/n_esc=1183 B5 trap 1.000/esc 1.000  B6 trap 1.000/esc 1.000  B7 ret diff CI [-0.0019,+0.0017]
  constr2 d=0.6 e=0.05: n_trap=1664/n_esc=286 B5 trap 1.000/esc 1.000  B6 trap 1.000/esc 1.000  B7 ret diff CI [-0.0051,+0.0010]
  constr2 d=0.6 e=0.25: n_trap=1114/n_esc=836 B5 trap 1.000/esc 1.000  B6 trap 1.000/esc 1.000  B7 ret diff CI [-0.0028,+0.0022]
  unconstr2 d=0.6 e=0.05: n_trap=738/n_esc=1212 B5 trap 1.000/esc 0.999  B6 trap 0.999/esc 1.000  B7 ret diff CI [-0.0012,+0.0025]
  unconstr2 d=0.6 e=0.25: n_trap=280/n_esc=1670 B5 trap 1.000/esc 0.990  B6 trap 0.718/esc 1.000  B7 ret diff CI [+0.0056,+0.0104]
