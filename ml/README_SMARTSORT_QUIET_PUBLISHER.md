# Smart Sort – quiet, idempotent Kaggle model retrieval

This patch replaces only the GitHub Actions **retrieval/publisher workflow** and
`ml/cloud/validate_model.py`, and adds `ml/cloud/probe_kaggle_model.py`.
It does **not** modify the trained model, datasets, training script, or FastAPI app.

## Outcomes

- Kaggle pending/running/error (no completed model): workflow **green** (no candidate).
- Completed model, already reviewed under the same thresholds and policy: **green**;
  skip downloading and archiving again (review marker stored in GitHub Actions cache).
- New valid checkpoint fails any quality metric: **green**, model marked **REJECTED**;
  record every gate in the GitHub Actions run summary; archive candidate; no publish.
- Valid model passes all gates: **green**; archive candidate. A commit/prerelease is
  allowed **only** if `ENABLE_MODEL_AUTO_COMMIT=true` and the checkpoint differs.
- Missing/corrupt files, mismatched provenance/class labels, invalid weights, Kaggle
  auth/network errors, or actual publication errors: workflow **red**.

Cache keys reflect the candidate model, source commit, dataset, thresholds, current
model SHA, validator code, and auto-commit configuration, so changing these
causes a fresh review. GitHub may expire/evict caches; a subsequent recheck is
harmless and still should not fail purely because metrics are below threshold.

To manually review an already cached candidate, choose **Run workflow** and set
`force_recheck: true`. This downloads, archives, and validates again but does not
replace its stored review marker.

**Security:** The archive remains on GitHub Actions for 30 days. It does not make
mixed-license images public. Keep `ENABLE_MODEL_AUTO_COMMIT=false` until you have
independently tested and approved a model. Passing training-validation gates is
not proof of performance on the physical camera.

## Install from PowerShell

Run from the repository root on branch `hayden`. Extract this ZIP to a temporary
folder and copy only the following three paths into the repository:

- `.github/workflows/smartsort-kaggle-publish.yml`
- `ml/cloud/validate_model.py`
- `ml/cloud/probe_kaggle_model.py` (new)

Review the diff, commit only those three files, push to `hayden`, and merge into
`main` using `git merge --no-ff hayden -m "Merge quiet publisher fix"`. Avoid
`git add .` because your dataset ZIP and unrelated files are untracked.

## Local smoke tests (no Kaggle network call)

From the extracted ZIP folder:

`python -m unittest discover -s tests -v`

This uses a synthetic, randomly initialized 11-class MobileNetV2 checkpoint to
verify that a low-scoring, structurally valid candidate is rejected with exit
code zero and that structural/provenance failures remain errors.

## GitHub notification preferences

Green workflow runs can still generate notifications **if your GitHub settings
subscribe to all Actions results**. Change your GitHub notification settings to
**Only notify for failed workflows** if you want no email for successful
scheduled checks. True infrastructure failures will still surface as errors.
