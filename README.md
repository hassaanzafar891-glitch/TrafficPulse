# TrafficPulse

Synthetic traffic-speed forecasting, reliability experiments and historical replay
with a FastAPI backend and React dashboard. This is a sanitized source-only export,
not live traffic, a real-city deployment or independently validated performance.

The implementation includes persistence, seasonal historical average, direct
XGBoost horizons at +15/+30/+60 minutes, simulated input faults, split-conformal
calibration and a joint reliability controller. Synthetic diagnostics and calibration
diagnostics are not independent final-test results. Switched/fallback intervals
remain unavailable; coverage under faults/distribution shift is not guaranteed.

## Source-only boundary

No datasets, sensor metadata, trained models, generated reports, MLflow databases,
DVC cache, receipts, credentials or private Git history are included. Model/API/
controller source and the API Contract1.1 schemas are unchanged. See
[export review](docs/PUBLIC_EXPORT_REVIEW.md) and [exact manifest](EXPORT_MANIFEST.json).
The policy-only evaluation document omits real-data audit observations while
preserving synthetic workflow settings and leakage-safety rules.

## Artifact-free checks

Python3.12, Node22.18 and Git are the CI toolchain. In a later authorized setup,
create a virtual environment and install the existing pinned test requirements:

```text
python -m venv .venv
python -m pip install -r configs/ci-requirements.txt
python scripts/validate_ci.py
python scripts/run_ci_checks.py
cd frontend
npm ci --ignore-scripts --no-audit --no-fund
npm test
npm run build
```

Activate/use the virtual environment interpreter before installing or running
Python commands. The backend runner selects lightweight mocked/synthetic tests,
not the full artifact-dependent suite or model training. Git must be initialized by
a separately approved publication step before the Git-ignore test can run locally.
GitHub checkout provides that context in hosted CI. Hosted Actions has not yet
been verified. Official release tags and full commit SHAs were verified for
checkout v7.0.1, setup-python v7.1.0 and setup-node v7.1.0; all declare Node24.
Hosted Linux execution and repository settings remain review gates.

## Dashboard requirements

A fresh source checkout cannot serve predictions: it requires an independently
approved eleven-file synthetic artifact bundle and three generated provenance
reports, supplied out of band. No artifact download, training or reconstruction
is automatic. Missing files fail startup or produce503, never invented forecasts.
The inventory contains only paths/sizes/hashes. The DVC foundation is snapshot/
integrity tooling, not full training reproduction. No remote is configured here.

See [Docker setup](docs/DOCKER.md) for exact runtime prerequisites. When those
files are supplied and a build is approved, the dashboard runs on
http://127.0.0.1:8081. Backend port8000 is unpublished. Local source startup remains
`python scripts/run_backend.py` and, separately, `npm run dev` in frontend/.
The serving dependencies exceed the artifact-free CI set; the Dockerfile derives
the existing pinned CPU serving requirements without installing MLflow or DVC.
Do not run full experiment launchers just to obtain missing artifacts: some legacy
entry points generate a full synthetic stream and are outside the current
no-final-test-access validation scope.

## Publication limitations

No source-code license is selected in this staging export; no dataset license or
permission is implied. Verify release licensing and dependency notices before
publication/reuse. This directory has no Git repository, commits or remote yet.
