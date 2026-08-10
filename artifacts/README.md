# Artifacts

Local, versioned model packages produced by:

```bash
python -m fraud_risk.train_final
```

## Layout

```text
artifacts/
└── xgb-transformed-v1/
    ├── model.joblib       # fitted XGB Transformed preprocessing + classifier
    ├── calibrator.joblib  # fitted Platt (sigmoid) calibrator
    └── metadata.json      # version, threshold, features, temporal ranges, library versions
```

## Git policy

Generated binary artifacts (`*.joblib`) and `metadata.json` are **not committed**. Only this README is tracked. Rebuild locally after clone with `python -m fraud_risk.train_final` (requires the PaySim CSV under `data/raw/`).

## Frozen portfolio model

`xgb-transformed-v1` is the frozen Task 4 / Task 5 deployment model (XGB Transformed + Platt calibration + threshold `0.044`).
