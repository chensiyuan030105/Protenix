"""P007: the structure benchmark -- five numbers and one sanity flag.

P009's held-out metric is the denoising loss, and its input carries a noised
copy of the answer (P004 section 2.18, fifth part).  It cannot say whether the
model predicts the next frame.  This package makes the model sample:

    oneshot.py   256 fixed held-out windows, 5 samples each, RMSD / lDDT
                 against the true target frame, with the persistence, static
                 and cross-replica baselines on the same windows
    rollout.py   8 history frames in, 100 steps of 0.1 ns out, autoregressive
    metrics.py   the kernels, ported from the MD calibration unchanged
    report.py    three dt bins, the noise floor, per system, against the
                 thresholds frozen before any model was scored

Every number has an MD-measured ceiling and floor in
artifacts/reports/P007/thresholds.json, and every number is read against the
`zero` / `zero-seed2` noise floor.  The metric set, the protocol and the
thresholds are settled in P007 sections 2-4; this code implements them and does
not reopen them.
"""
