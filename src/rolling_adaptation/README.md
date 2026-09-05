# Secure rolling ST-fatigue adaptation

This experiment tests whether recovery during an ST-fatigue recording makes a
fixed temporary verifier outdated. It compares that fixed verifier with a
rolling verifier on identical evaluation intervals.

The default schedule is 30 seconds initial trusted update, 20 seconds initial
calibration, then 50 seconds evaluation followed by a 20-second trusted refresh
(10 seconds update and 10 seconds calibration). The rolling model retains at
most 60 recent trusted windows. Refresh windows never contribute to evaluation.

The experiment also reports `fixed_two_stage` and `rolling_two_stage`. These
methods require agreement from a complementary spectral verifier whose joint
threshold is calibrated against the worst development impostor identity.

`rolling_spectral_rescue` keeps the fixed verifier's acceptances and uses
rolling-plus-spectral agreement only to rescue fixed-model rejections. This
cannot increase FRR relative to the fixed verifier; the complementary gate is
used to limit the additional FAR.

```bash
source venv/bin/activate
python3 -m src.rolling_adaptation.run_experiment --fold 1
```
