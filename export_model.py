"""Export the champion model, the explainer and the config from the MLflow registry
into a self-contained folder (model_bundle/) that can be copied into a Docker image.

Run from the project root:   python export_model.py
"""
#ce code crée un dossier model_bundle/ contenant le modèle champion, l'explainer et le fichier de configuration à partir du registre MLflow. Il peut être exécuté depuis la racine du projet avec la commande : python export_model.py
import json
import os
import shutil
from pathlib import Path

import mlflow
from mlflow import MlflowClient

OUT = Path(os.getenv("BUNDLE_DIR", "model_bundle"))
ALIAS = "champion"


def main():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    client = MlflowClient()

    mv = client.get_model_version_by_alias("credit_scoring", ALIAS)
    ev = client.get_model_version_by_alias("credit_scoring_explainer", ALIAS)
    if mv.run_id != ev.run_id:
        raise SystemExit("Champion models come from different runs - fix the aliases first.")

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "model").mkdir(parents=True)
    (OUT / "explainer").mkdir(parents=True)

    mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/credit_scoring@{ALIAS}", dst_path=str(OUT / "model"))
    mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/credit_scoring_explainer@{ALIAS}", dst_path=str(OUT / "explainer"))

    config = mlflow.artifacts.load_dict(f"runs:/{mv.run_id}/config.json")
    config["model_version"] = str(mv.version)
    config["run_id"] = mv.run_id
    (OUT / "config.json").write_text(json.dumps(config, indent=2))

    print(f"Exported credit_scoring v{mv.version} (run {mv.run_id}) to {OUT.resolve()}")


if __name__ == "__main__":
    main()

