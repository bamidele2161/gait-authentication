"""Numbers for the paired tests table and the DET figure (DT-fatigue).

Builds the inputs from the result folders, then runs the paired tests and the
DET plot:
  per_user.csv   config, user, recording, frr, far (percent; 0.1% FAR target,
                 logistic regression, nine IMUs)
  scores/<config>_DT-fatigue.npz   genuine, impostor (pooled 10-stride decision
                 scores of the 16 test users; higher = genuine), gen_fold,
                 imp_fold (fold of each score), thr_01, thr_1 (deployed
                 threshold per fold at the 0.1% and 1% targets)

The operating points are computed with each fold's own deployed threshold
(pooled over folds), so they are the pooled, not the per-user, FRR and FAR.

    PYTHONPATH=. venv/bin/python docs/build/paper/paired_tests_det.py OUTDIR
"""
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, wilcoxon

ROOT = Path("src/results")
RECORDING = {"st_fatigue": "ST-fatigue", "dt_control": "DT-control", "dt_fatigue": "DT-fatigue"}
COMPARISONS = [  # (A, B, metric, recording)
    ("conventional", "hc_cvae", "frr", "DT-fatigue"),
    ("conventional", "hc_cvae", "far", "DT-control"),
    ("hc", "hc_cvae", "frr", "DT-fatigue"),
    ("hc", "hc_cvae", "far", "ST-fatigue"),
    ("hc", "hc_bank", "frr", "DT-fatigue"),
    ("hc_ridge", "hc_cvae", "frr", "DT-fatigue"),
    ("hc_bank", "hc_cvae", "frr", "DT-fatigue"),
    ("hc_bank", "hc_cvae", "far", "ST-fatigue"),
]
COLORS = {"conventional": "#52514e", "hc": "#1baf7a", "hc_bank": "#eb6834", "hc_cvae": "#2a78d6"}
plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.linewidth": 0.6,
                     "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})


def robust_dirs():
    """{(config, target): folder} for the proposed pipeline (LR, nine IMUs, all conditions)."""
    out = {}
    for f in glob.glob(str(ROOT / "robust_gait" / "*" / "config.json")):
        c = json.loads(Path(f).read_text()); v = c["verifier"]
        if v["classifier"] != "logreg" or len(v["locations"]) != 9 or not v["enrolment_augmentation"]:
            continue
        if tuple(c.get("drift_conditions", ("st_fatigue", "dt_control", "dt_fatigue"))) != (
                "st_fatigue", "dt_control", "dt_fatigue") or c.get("calibration_conditions"):
            continue
        source = c["drift_model"] if v["drift_augmentation"] else "none"
        name = {"none": "hc", "bank": "hc_bank", "cvae": "hc_cvae", "ridge": "hc_ridge"}[source]
        out[name, round(c["target_far"], 4)] = Path(f).parent
    for t in (0.001, 0.01):
        out["conventional", t] = ROOT / "multisensor_problem_definition_stride" / f"lr_proposed_protocol_far{t:g}_seed42"
    return out


def build_inputs(work):
    dirs = robust_dirs()
    rows = []
    for (name, t), d in dirs.items():
        if t != 0.001:
            continue
        m = pd.read_csv(d / "participant_metrics.csv")
        if "decision" in m:
            m = m[m.decision == "10strides"]
        for _, r in m.iterrows():
            rows.append({"config": name, "user": r.claimed_participant_id,
                         "recording": RECORDING[r.condition], "frr": 100 * r.frr, "far": 100 * r.far})
    pd.DataFrame(rows).to_csv(work / "per_user.csv", index=False)
    (work / "scores").mkdir(exist_ok=True)
    for name in ("conventional", "hc", "hc_bank", "hc_cvae"):
        pieces = {"gen": [], "imp": [], "gf": [], "if": []}
        thr = {}
        for t in (0.001, 0.01):
            z = np.load(dirs[name, t] / "scores.npz")
            prefix = "10strides|" if name == "conventional" else ""
            tkey = "threshold|10strides|{}" if name == "conventional" else "threshold|{}"
            thr[t] = np.array([float(z[tkey.format(f)][0]) for f in range(4)])
            if t != 0.001:
                continue
            for key in z.files:
                if key.startswith("threshold") or not key.startswith(prefix) or not key.endswith("dt_fatigue"):
                    continue
                fold, claimed, owner, _ = key[len(prefix):].split("|")
                kind = ("gen", "gf") if claimed == owner else ("imp", "if")
                pieces[kind[0]].append(z[key]); pieces[kind[1]].append(np.full(len(z[key]), int(fold)))
        np.savez(work / "scores" / f"{name}_DT-fatigue.npz",
                 genuine=np.concatenate(pieces["gen"]), impostor=np.concatenate(pieces["imp"]),
                 gen_fold=np.concatenate(pieces["gf"]), imp_fold=np.concatenate(pieces["if"]),
                 thr_01=thr[0.001], thr_1=thr[0.01])


def paired(df, a, b, metric, rec, n_boot=10_000, seed=0):
    x = df[(df.config == a) & (df.recording == rec)].set_index("user")[metric]
    y = df[(df.config == b) & (df.recording == rec)].set_index("user")[metric]
    d = (y - x.reindex(y.index)).to_numpy()          # B minus A, per user
    assert len(d) == 16 and not np.isnan(d).any()
    rng = np.random.default_rng(seed)
    boot = rng.choice(d, (n_boot, len(d))).mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    k = int((d != 0).sum())
    p = wilcoxon(d, zero_method="wilcox").pvalue if k > 0 else 1.0
    return d.mean(), lo, hi, k, p


def holm(p):
    p = np.asarray(p); order = np.argsort(p); adj = np.empty_like(p); m = len(p); running = 0
    for i, j in enumerate(order):
        running = max(running, (m - i) * p[j]); adj[j] = min(1.0, running)
    return adj


def table(per_user_csv):
    df = pd.read_csv(per_user_csv)
    rows = [paired(df, *c) for c in COMPARISONS]
    padj = holm([r[4] for r in rows])
    out = []
    for c, r, pa in zip(COMPARISONS, rows, padj):
        print(f"{c[0]:>12} -> {c[1]:<8} {c[2].upper()} {c[3]:<10} "
              f"{r[0]:+.1f} [{r[1]:+.1f}, {r[2]:+.1f}]  k={r[3]:2d}  p={pa:.3g}"
              f"  (min possible p={2.0**(1-r[3]) if r[3] else 1:.3g})")
        out.append({"A": c[0], "B": c[1], "metric": c[2], "recording": c[3], "mean_diff": r[0],
                    "ci_lo": r[1], "ci_hi": r[2], "k": r[3], "p_raw": r[4], "p_holm": pa})
    return pd.DataFrame(out)


def det_points(gen, imp):
    thr = np.unique(np.concatenate([gen, imp]))
    frr = np.array([(gen < t).mean() for t in thr])   # accept if score >= threshold
    far = np.array([(imp >= t).mean() for t in thr])
    eer_i = np.argmin(np.abs(frr - far))
    return far, frr, (frr[eer_i] + far[eer_i]) / 2


def det_figure(scores_dir, configs=("conventional", "hc", "hc_bank", "hc_cvae"),
               labels=("Conventional", "Hand-crafted only", "+ drift bank", "+ CVAE drift"),
               out="det_dtfatigue.pdf"):
    clip = lambda v: norm.ppf(np.clip(v, 1e-4, 1 - 1e-4))
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    points = []
    for cfg, lab in zip(configs, labels):
        z = np.load(Path(scores_dir) / f"{cfg}_DT-fatigue.npz")
        far, frr, eer = det_points(z["genuine"], z["impostor"])
        line, = ax.plot(clip(far), clip(frr), color=COLORS[cfg], lw=1.2,
                        label=f"{lab} (EER {100*eer:.1f}%)")
        for key, fill in (("thr_01", True), ("thr_1", False)):
            thr = z[key]   # one deployed threshold per fold
            op_far = (z["impostor"] >= thr[z["imp_fold"]]).mean()
            op_frr = (z["genuine"] < thr[z["gen_fold"]]).mean()
            points.append((cfg, key, 100 * op_frr, 100 * op_far))
            ax.plot(clip(op_far), clip(op_frr), "o", ms=4, color=line.get_color(),
                    mfc=line.get_color() if fill else "white", mew=1.0, zorder=3)
    ticks = np.array([0.001, 0.01, 0.05, 0.2, 0.5])
    ax.set_xticks(clip(ticks)); ax.set_xticklabels([f"{100*t:g}" for t in ticks])
    ax.set_yticks(clip(ticks)); ax.set_yticklabels([f"{100*t:g}" for t in ticks])
    ax.set_xlim(clip(5e-4), clip(0.7)); ax.set_ylim(clip(5e-4), clip(0.7))
    ax.set_xlabel("FAR (%)"); ax.set_ylabel("FRR (%)"); ax.legend(fontsize=6, frameon=False)
    ax.grid(alpha=.3, lw=0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout(); fig.savefig(out); fig.savefig(str(out).rsplit(".", 1)[0] + ".png", dpi=200)
    for p in points:
        print(f"{p[0]:>12} {'0.1%' if p[1] == 'thr_01' else '1%':>4} target: pooled FRR {p[2]:.1f}%  FAR {p[3]:.2f}%")


if __name__ == "__main__":
    outdir = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    work = outdir / "paired_det_inputs"
    work.mkdir(parents=True, exist_ok=True)
    build_inputs(work)
    table(work / "per_user.csv").to_csv(outdir / "paired_tests.csv", index=False)
    det_figure(work / "scores", out=outdir / "det_dtfatigue.pdf")
