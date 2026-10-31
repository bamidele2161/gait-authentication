"""Share of each test user's real drift lying in the span of the 36 learned drift vectors."""
import numpy as np, pandas as pd
exec(open(__file__.replace("subspace.py", "drift_accuracy.py")).read().split("rows = []")[0])
rows = []
rng = np.random.default_rng(0)
for fold in create_outer_folds(people, 4):
    dev = list(fold.development_participants)
    bank = R.drift_bank(dev, data, exp)
    for u in fold.evaluation_participants:
        enrol = data[u, "st_control"].strides
        keep = [NAMES.index(l) for l in R.working_locations(enrol, dev, data, exp)]
        bg = np.vstack([feats(data[p, c].strides, keep) for p in dev for c in R.TRAINING_CONDITIONS])
        mu, sd = bg.mean(0), bg.std(0) + 1e-9
        f = lambda s: (feats(s, keep) - mu) / sd
        st = f(enrol).mean(0)
        B = np.stack([f(enrol + v).mean(0) - st for v in bank]).T          # (features, 36)
        Q, _ = np.linalg.qr(B)
        Qr, _ = np.linalg.qr(rng.standard_normal(B.shape))                  # random 36-dim baseline
        for c in LATER:
            a = f(data[u, c].strides).mean(0) - st
            rows.append({"condition": c, "user": u,
                         "captured_by_learned_span": np.sum((Q.T @ a) ** 2) / np.sum(a ** 2),
                         "captured_by_random_span": np.sum((Qr.T @ a) ** 2) / np.sum(a ** 2)})
df = pd.DataFrame(rows)
print(df.groupby("condition")[["captured_by_learned_span", "captured_by_random_span"]].agg(["mean", "min", "max"]).round(3))
df.to_csv(__file__.replace("subspace.py", "subspace.csv"), index=False); print("dims:", B.shape)
