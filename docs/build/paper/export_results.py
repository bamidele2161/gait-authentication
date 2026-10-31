"""Export every 9-sensor result (problem definition + proposed solution) to docs/results_9sensor/.

Reads saved result folders only; runs no experiments.
Run from the repository root: PYTHONPATH=. venv/bin/python docs/build/paper/export_results.py
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[3]
RG = ROOT / "src" / "results" / "robust_gait"
OUT = ROOT / "docs" / "results_9sensor"
OUT.mkdir(exist_ok=True)
NINE = "sacrum+left_leg+right_leg+left_wrist+right_wrist+left_foot+right_foot+head+sternum_late-median_C0.3_"
THREE = "_probes-st_fatigue-dt_control-dt_fatigue"
COND = ["st_control", "st_fatigue", "dt_control", "dt_fatigue"]

SYSTEMS = [  # (id, classifier, augmentation, negatives, channels, protocol, folder stem)
    ("LR-hand", "Logistic regression", "hand-made", "ST-control + ST-fatigue", "raw axes", "3 conditions",
     "axes_mean_aug4a0.2w0.1_bg240_snorm_calsession2_minimax_g2_fuse10" + THREE),
    ("LR-drift", "Logistic regression", "learned drift", "ST-control + ST-fatigue", "raw axes", "3 conditions",
     "axes_mean_aug0a0.2w0.1_bg240_snorm_calsession2_minimax_g2_fuse10_drift4" + THREE),
    ("LR-hand+drift (FINAL)", "Logistic regression", "hand-made + learned drift", "ST-control + ST-fatigue", "raw axes",
     "3 conditions", "axes_mean_aug4a0.2w0.1_bg240_snorm_calsession2_minimax_g2_fuse10_drift4" + THREE),
    ("LR-hand, negatives ST-control only", "Logistic regression", "hand-made", "ST-control only", "raw axes",
     "3 conditions", "axes_mean_aug4a0.2w0.1_bg240_snorm_calsession2_minimax_g2_fuse10" + THREE + "_bgcond-st_control"),
    ("SVM-hand", "RBF-SVM", "hand-made", "ST-control + ST-fatigue", "raw axes", "3 conditions",
     "axes_mean_aug4a0.2w0.1_bg240_snorm_calsession2_minimax_g2_fuse10_svmC0.1g0.01" + THREE),
    ("SVM-drift", "RBF-SVM", "learned drift", "ST-control + ST-fatigue", "raw axes", "3 conditions",
     "axes_mean_aug0a0.2w0.1_bg240_snorm_calsession2_minimax_g2_fuse10_drift4_svmC0.1g0.01" + THREE),
    ("SVM-hand+drift", "RBF-SVM", "hand-made + learned drift", "ST-control + ST-fatigue", "raw axes", "3 conditions",
     "axes_mean_aug4a0.2w0.1_bg240_snorm_calsession2_minimax_g2_fuse10_drift4_svmC0.1g0.01" + THREE),
    ("LR-hand, raw axes (DT-only protocol)", "Logistic regression", "hand-made", "ST-control + ST-fatigue", "raw axes",
     "DT only", "axes_mean_aug4a0.2w0.1_bg240_snorm_calsession2_minimax_g2_fuse10"),
    ("LR-hand, magnitudes (DT-only protocol)", "Logistic regression", "hand-made", "ST-control + ST-fatigue",
     "magnitudes", "DT only", "mean_aug4a0.2w0.1_bg240_snorm_calsession2_minimax_g2_fuse10"),
]


def seed_folders(stem):
    base = RG / (NINE + stem)
    fs = [base, Path(str(base) + "_seed142"), Path(str(base) + "_seed242")]
    missing = [f for f in fs if not (f / "participant_metrics.csv").exists()]
    if missing:
        raise FileNotFoundError(missing)
    return fs


summary, per_seed, per_user = [], [], []
for sid, clf, aug, neg, ch, prot, stem in SYSTEMS:
    seeds = []
    for seed, f in zip((42, 142, 242), seed_folders(stem)):
        m = pd.read_csv(f / "participant_metrics.csv")
        m["seed"] = seed
        seeds.append(m)
    m = pd.concat(seeds)
    ps = m.groupby(["seed", "condition"])[["frr", "far", "eer"]].mean().mul(100).reset_index()
    for _, r in ps.iterrows():
        per_seed.append(dict(system=sid, seed=r.seed, condition=r.condition, FRR=r.frr, FAR=r.far, EER=r.eer))
    for c, g in ps.groupby("condition"):
        summary.append(dict(group="Proposed solution", system=sid, classifier=clf, augmentation=aug, negatives=neg,
                            channels=ch, protocol=prot, condition=c, decision="10 strides (~12 s)",
                            FRR=g.frr.mean(), FRR_min=g.frr.min(), FRR_max=g.frr.max(),
                            FAR=g.far.mean(), FAR_min=g.far.min(), FAR_max=g.far.max(),
                            EER=g.eer.mean(), EER_min=g.eer.min(), EER_max=g.eer.max(), seeds=3))
    pu = m.groupby(["claimed_participant_id", "condition"])[["frr", "far", "eer"]].mean().mul(100).reset_index()
    for _, r in pu.iterrows():
        per_user.append(dict(system=sid, user=r.claimed_participant_id, condition=r.condition,
                             FRR=r.frr, FAR=r.far, EER=r.eer))

b = pd.read_csv(ROOT / "src/results/multisensor_problem_definition_rerun/raw_axes_9imu_targetfar0.01/participant_metrics.csv")
for dec, g in b.groupby("decision"):
    sid = f"Problem-definition SVM ({dec} decisions)"
    for c, gc in g.groupby("condition"):
        summary.append(dict(group="Problem definition", system=sid, classifier="RBF-SVM (per person, grid-searched)",
                            augmentation="none", negatives="other 15 participants' ST-control",
                            channels="raw axes, 594 statistical features", protocol="per-person 60/20/20 split",
                            condition=c, decision="2 s window" if dec == "2s" else "10 windows (11 s)",
                            FRR=gc.frr.mean() * 100, FAR=gc.far.mean() * 100, EER=gc.eer.mean() * 100, seeds=1))
    for _, r in g.iterrows():
        per_user.append(dict(system=sid, user=r.participant_id, condition=r.condition,
                             FRR=r.frr * 100, FAR=r.far * 100, EER=r.eer * 100))

order = {c: i for i, c in enumerate(COND)}
S = pd.DataFrame(summary)
S["_o"] = S.condition.map(order)
grp = {"Problem definition": 0, "Proposed solution": 1}
S = S.sort_values(["group", "system", "_o"], key=lambda s: s.map(grp) if s.name == "group" else s).drop(columns="_o")
S.round(2).to_csv(OUT / "01_summary_all_systems.csv", index=False)

W = S.pivot_table(index=["group", "system"], columns="condition", values=["FRR", "FAR", "EER"]).round(2)
W = W.reindex(columns=[(m, c) for c in COND for m in ("FRR", "FAR", "EER") if (m, c) in W.columns])
W.columns = [f"{c} {m}" for m, c in W.columns]
W.to_csv(OUT / "02_summary_wide.csv")

pd.DataFrame(per_seed).round(2).to_csv(OUT / "03_per_seed.csv", index=False)
pd.DataFrame(per_user).round(2).to_csv(OUT / "04_per_user.csv", index=False)

pooled = pd.read_csv(HERE := Path(__file__).resolve().parent / "fig/experiments/summary_auc_eer.csv")
name = {"baseline_svm": "Problem-definition SVM (11s decisions)", "lr_hand": "LR-hand", "lr_drift": "LR-drift",
        "lr_both": "LR-hand+drift (FINAL)", "svm_hand": "SVM-hand", "svm_drift": "SVM-drift",
        "svm_both": "SVM-hand+drift"}
pooled["system"] = pooled.system.map(name)
pooled.rename(columns={"auc": "pooled_AUC", "pooled_eer": "pooled_EER", "frr": "pooled_FRR", "far": "pooled_FAR"}) \
    .round(4).to_csv(OUT / "05_pooled_auc_eer_seed42.csv", index=False)

PU = pd.DataFrame(per_user)
worst = PU.assign(W=np.maximum(PU.FRR, PU.FAR)).pivot_table(index=["user", "condition"], columns="system", values="W")
tests = []
pairs = [("LR-hand+drift (FINAL)", "Problem-definition SVM (11s decisions)"),
         ("SVM-hand+drift", "Problem-definition SVM (11s decisions)"),
         ("LR-hand+drift (FINAL)", "LR-hand"), ("LR-hand+drift (FINAL)", "SVM-hand+drift"),
         ("LR-hand", "LR-hand, negatives ST-control only")]
for a, bb in pairs:
    for c in ["st_fatigue", "dt_control", "dt_fatigue"]:
        x, y = worst.xs(c, level="condition")[a], worst.xs(c, level="condition")[bb]
        tests.append(dict(comparison=f"{a} vs {bb}", condition=c, mean_worst_A=x.mean(), mean_worst_B=y.mean(),
                          users_A_better=int((x < y).sum()), users_A_worse=int((x > y).sum()),
                          ties=int((x == y).sum()), wilcoxon_p=wilcoxon(x, y).pvalue))
pd.DataFrame(tests).round(4).to_csv(OUT / "06_paired_tests.csv", index=False)
print("written to", OUT)
for f in sorted(OUT.glob("*.csv")):
    print(" ", f.name, len(pd.read_csv(f)), "rows")
