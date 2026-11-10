# Docker synthetic replay

Two CPU-only Linux containers: Node22 builds React; non-root Nginx serves static
assets and proxies /api and /health to the Python3.12 FastAPI backend. One Uvicorn
worker and four numerical threads; total runtime CPU quota4. Source layout stays
/app/src/trafficpulse. Model/controller logic and API Contract1.1 are unchanged.

Frontend binds only127.0.0.1:8081 on the host and8080 in the container. Backend
port8000 is not published and backend joins only the internal demo bridge. Frontend
joins demo and the normal web bridge so Docker can publish its localhost port.
No backend CORS change or browser absolute API URL is required.

## Required files, deliberately not exported

Compose binds exactly the11 paths listed in configs/reviewed_synthetic_artifacts.json
under /app/artifacts with matching relative suffixes, plus these3 reports under
/app/reports/generated:

- phase2b_synthetic_xgboost.json
- phase3b_synthetic_uncertainty.json
- phase3c_synthetic_controller.json

All14 binds are individual/read-only and use create_host_path:false. Obtain a
separately reviewed synthetic runtime bundle before startup; source alone cannot
produce predictions. The existing DVC snapshot does not include these3 reports.
Do not download from an assumed public remote or regenerate artifacts automatically.
The entry point verifies artifact sizes/hashes; existing API provenance checks remain.

## Later authorized startup

With Docker Desktop Linux containers on Windows, or an appropriate Linux Docker
engine, and the approved runtime files in place:

```text
docker compose -f compose.yaml config --quiet
docker compose -f compose.yaml build
docker compose -f compose.yaml up -d
docker compose -f compose.yaml ps
```

Open http://127.0.0.1:8081. GET /health checks backend readiness; /nginx-health
checks static-server liveness. Use `docker compose -f compose.yaml down` to stop
this stack. Configure a four-CPU build budget separately: runtime CPU quotas do
not throttle image builds. Builds download public dependencies; preparation of
this export did not build/run containers.

The existing local stack successfully built and passed health, metadata, HTML/JS
and one missing-input replay fixture smoke check after the web-bridge correction.
The fixture matched its existing expected joint fallback/input_unavailable decision
and suppressed intervals. This is a bounded synthetic integration smoke test, not
full predictive equivalence, real-world performance or independent final testing.

Backend uses xgboost-cpu at the exact manifest XGBoost version and libgomp1, with
sklearn/SciPy pins from the existing environment manifest. It does not install
NCCL, MLflow or DVC. Base tags and some transitive dependencies remain mutable;
image digests/Linux locking and hosted rebuilds need further review.
Deny-all Docker context allowlists exclude secrets, datasets, credentials, Git,
environments, DVC/MLflow stores, receipts and runtime reports/artifacts. Never
remove exclusions or mount the entire project to make startup succeed.
