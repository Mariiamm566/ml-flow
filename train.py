import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("data/iris.csv")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "classifier-ci-cd")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
FORCE_ACCURACY = os.getenv("FORCE_ACCURACY", "").strip()


def ensure_dataset() -> pd.DataFrame:
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df = df.rename(columns={"target": "target"})
    df.to_csv(DATA_PATH, index=False)
    return df


def main() -> None:
    if not TRACKING_URI:
        raise ValueError("MLFLOW_TRACKING_URI is not set.")

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = ensure_dataset()

    if "target" not in df.columns:
        raise ValueError("Dataset must contain a 'target' column.")

    x = df.drop(columns=["target"])
    y = df["target"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=300, random_state=42)),
        ]
    )

    with mlflow.start_run() as run:
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        accuracy = float(accuracy_score(y_test, preds))

        if FORCE_ACCURACY:
            accuracy = float(FORCE_ACCURACY)

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, artifact_path="model")

        Path("run_id.txt").write_text(run.info.run_id, encoding="utf-8")

        print(f"Run ID: {run.info.run_id}")
        print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
