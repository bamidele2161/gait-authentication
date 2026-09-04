# Two-stage state-specific verification

The primary verifier uses the existing 88 statistical features. A complementary
verifier uses 48 cadence, spectral-energy, entropy, centroid, and periodicity
features. After trusted current-state adaptation, a probe is accepted only when
both verifiers accept it. Their thresholds are selected jointly on trusted
calibration gait and development-participant impostors.

```bash
source venv/bin/activate
python3 -m src.two_stage_verification.run_experiment --fold 1
```
