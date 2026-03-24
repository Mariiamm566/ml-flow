import os
import sys
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

THRESHOLD = 0.85


def main() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        print("MLFLOW_TRACKING_URI is not set.")
        sys.exit(1)

    info_file = Path("model_info.txt")
    if not info_file.exists():
        print("model_info.txt not found.")
        sys.exit(1)

    run_id = info_file.read_text(encoding="utf-8").strip()
    if not run_id:
        print("Run ID is empty.")
        sys.exit(1)

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    run = client.get_run(run_id)
    accuracy = run.data.metrics.get("accuracy")

    if accuracy is None:
        print(f"No 'accuracy' metric found for run {run_id}")
        sys.exit(1)

    print(f"Run ID: {run_id}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Threshold: {THRESHOLD:.2f}")

    if float(accuracy) < THRESHOLD:
        print("Accuracy below threshold. Pipeline will fail.")
        sys.exit(1)

    print("Accuracy passed threshold. Deployment can continue.")


if __name__ == "__main__":
    main()
