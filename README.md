# Model Validation and Deploy Pipeline

This project implements a two-job GitHub Actions pipeline:

- **validate**
  - installs dependencies
  - runs `dvc pull`
  - runs `python train.py`
  - logs to MLflow using `MLFLOW_TRACKING_URI`
  - writes the current MLflow Run ID to `model_info.txt`
  - uploads `model_info.txt` as an artifact

- **deploy**
  - downloads `model_info.txt`
  - runs `python check_threshold.py`
  - fails if accuracy is below `0.85`
  - otherwise runs a mock Docker build and a real Docker build

## Secret required

Create this repository secret:

- `MLFLOW_TRACKING_URI`

Example:

`http://YOUR_MLFLOW_SERVER:5000`

## Important note for demos

The workflow still calls `dvc pull` as required. For quick demonstration, `train.py` also generates a fallback Iris dataset if `data/iris.csv` is missing. That lets you produce the required fail/success screenshots even before wiring a real DVC remote.

## How to force a failed or successful run

Use **Actions > model-validation-and-deploy > Run workflow**.

Set `force_accuracy` to one of these values:

- `0.80` → deploy job fails in `check_threshold.py`
- `0.95` → deploy job passes and completes the mock build
- leave it empty → use the real measured accuracy

## What screenshots to capture

### Failed run

Show:
- `validate` job green
- `deploy` job red
- open the `Check threshold` step and show `Accuracy below threshold. Pipeline will fail.`

### Successful run

Show:
- `validate` job green
- `deploy` job green
- open the `Mock Build` step and show `Building Docker image for Run ID: ...`
