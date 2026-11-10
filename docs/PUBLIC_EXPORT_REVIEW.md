# Public export review

Status: BLOCKED for publication pending separate review/approval. This source
staging directory has no Git history, commits, remotes or uploaded payloads.
The existing research repository and private remote were not changed.

## Included boundary

91 selected source/configuration/documentation/placeholder files, plus this report
and EXPORT_MANIFEST.json:93 files total. The manifest records relative path, byte
size, SHA-256, category and copy/sanitization status for every other92 files. It
explicitly excludes its own hash to avoid a self-referential digest.

Algorithm, controller, API Contract1.1 schemas, frontend, tests, selected scripts
copies are byte-identical to their reviewed sources. Public CI action pins and
their validator were subsequently updated together to verified Node24 releases. No model
performance, independent final-test, live-traffic or real-world validation claims
are made. Local Docker smoke evidence is described narrowly in docs/DOCKER.md.

## Exclusions

No original Git metadata/author history, private operational ledgers/master plan,
real-dataset audit report/documentation/statistics, raw data, sensor IDs/coordinates,
credentials, .dvc/config.local, DVC cache, MLflow records, runtime models/reports,
upload receipts, screenshots, installed environments or node_modules are included.
Real-data audit readers and historical-run metadata upload tools/tests are omitted;
they are not required by the public CI selection or serving imports.

The synthetic artifact inventory contains names/sizes/hashes only, not weights,
observations or decision arrays. It is retained for source integrity tests and
explicit future artifact verification. No DVC lockfile is copied: its original
policy fingerprint would be stale after privacy redaction. dvc.yaml remains a
snapshot/integrity foundation and retains its original Windows-local commands,
not a portable full-training DAG.

## Sanitized derivatives

configs/evaluation_protocol.json has a distinct public policy-only identifier.
Real dataset identity, checksum, observed counts, timestamps, instantiated split
endpoints and prior audit-access disclosure were removed. Train-only fit endpoints
are expressed as train-only scope, and the dataset-sized fault selector as
floor(rate * sensor_count). Outer/validation split methodology and operational
window, masking, model-selection, calibration, fault grid/seeds, controller and
resource rules remain. Every field read by the synthetic fault/calibration/
controller scripts is preserved exactly. This is not a replacement for the private
frozen instantiation and does not change the original research evaluation files.

.env.example uses relative placeholder directories. The offline readiness policy
keeps default-deny flags and empty destination placeholders but removes a private
historical run identifier. The version-inventory path needed by CI/Docker retains
only the unchanged sklearn/SciPy supplemental serving pins. Public README and
policy/Docker documentation are newly authored without personal paths or private
operational chronology. Intentional fictitious path literals remain in security
tests; they are not personal filesystem records.

## Reference and availability review

All dependencies listed by the CI DVC-foundation validator exist in the export;
no excluded data-audit file is required. Workflow/package/dependency manifests and
frontend source are present. Docker context allowlists include the serving inputs.

Intentional unavailable references: the11 runtime artifact paths in the inventory
and3 reports named in docs/DOCKER.md and compose.yaml. Artifact-dependent API/
model tests and experiment entry points need those files and optional dependencies.
They are outside lightweight CI. Source-only startup cannot serve predictions.
No remote URL/credentials or automatic artifact download is supplied, and training
is not an approved workaround for missing files. Fresh-machine demo recovery needs
a separately approved synthetic bundle including all3 reports.

The optional local tracking adapter imports MLflow only when explicitly invoked;
it is not exercised as a remote integration by CI. Local DVC wrappers remain
default-deny and do not configure storage. No cloud deployment is included.

## Validation

Static workflow validation passed, as did public pipeline dependency closure.
33 selected artifact-free backend checks passed under a socket/protected-path
guard, four-thread caps and a1800-second bound; no XGBoost/MLflow/DVC client module
was imported. The remaining Git-ignore subprocess test needs an actual Git
checkout and is deferred because Git initialization is forbidden at this stage.
Equivalent representative ignore paths were checked statically. The unchanged
hosted workflow still selects all34 tests; GitHub checkout supplies Git context.
Frontend tests/build were not executed in this source-only export; no installed
packages were copied or installed. Frontend package/lockfile roots agree.

No confirmed provider token/private key, personal home-directory path or original
workspace path was detected. No binary dataset/database/model payload is present.
This is scoped pattern/structure review, not a certified secret scan or proof
against all sensitive prose. EXPORT_MANIFEST.json supports review of exact bytes.

## Remaining publication gates

- Review and approve this exact export and policy-only derivative.
- Choose the source-code license and review dependency notices; no license grant
  or dataset redistribution permission is invented here.
- Official release tags, full commit SHAs and Node24 metadata were verified for
  actions/checkout v7.0.1, actions/setup-python v7.1.0 and actions/setup-node v7.1.0.
  The workflow and strict validator use the same pins. Linux hosted CI remains
  unverified. Docker base tags/transitive packages are not fully immutable locks.
- Choose an approved public commit identity only during a later authorized Git
  initialization. No existing private author history is copied.
- Review unavailable-artifact delivery separately from source publication.

No Git initialization, commit, remote, push, install, training, container build,
DVC/MLflow operation or data upload was performed while creating this export.
