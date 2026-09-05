# Two-stage state-specific verification

The primary verifier uses the existing 88 statistical features. A complementary
verifier uses 48 cadence, spectral-energy, entropy, centroid, and periodicity
features. After trusted current-state adaptation, a probe is accepted only when
both verifiers accept it. Their thresholds are selected jointly on trusted
calibration gait and development-participant impostors.

The `two_stage_group_robust` method strengthens calibration: every development
impostor identity must satisfy the FAR constraint separately. This prevents a
dangerous identity from being hidden by the average of many easy impostors.

```bash
source venv/bin/activate
python3 -m src.two_stage_verification.run_experiment --fold 1
```
