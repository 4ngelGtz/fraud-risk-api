# Artifacts

Local, versioned model packages produced by training or CI helpers.

## Layout

```text
artifacts/
└── xgb-transformed-v1/      # path expected by Docker / FRAUD_MODEL_DIR
    ├── model.joblib
    ├── calibrator.joblib
    └── metadata.json
```

The on-disk directory name for Docker is always `xgb-transformed-v1/`. Which package lives there depends on how it was generated:

| Package | How created | `metadata.model_version` | Purpose |
| --- | --- | --- | --- |
| **Real portfolio model** | `python -m fraud_risk.train_final` (needs PaySim under `data/raw/`) | `xgb-transformed-v1` | Local deployment / evaluation |
| **CI smoke package** | `python scripts/create_ci_smoke_artifact.py` | `ci-smoke-v1` | Ephemeral serving-contract smoke tests on CI runners only |

`ci-smoke-v1` is **not** a fraud model. It must never be used for evaluation or deployment. It exists only so CI can build and start the API without PaySim or the real binary.

## Git policy

Generated binary artifacts (`*.joblib`) and `metadata.json` are **not committed**. Only this README is tracked.

- Rebuild the real package locally: `python -m fraud_risk.train_final`
- CI regenerates the synthetic smoke package on the runner before `docker build`; those binaries are never pushed to Git

For Docker, whichever package is present under `artifacts/xgb-transformed-v1/` is copied into the image at build time.

## Frozen portfolio model

`xgb-transformed-v1` is the frozen Task 4 / Task 5 deployment model (XGB Transformed + Platt calibration + threshold `0.044`).
