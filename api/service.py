import json
import math
import os
from pathlib import Path
 
import mlflow
import mlflow.sklearn
import pandas as pd
import shap
from mlflow import MlflowClient
 
from .schemas import MODEL_COLUMNS
 
MODEL_NAME = "credit_scoring"
EXPLAINER_NAME = "credit_scoring_explainer"
ALIAS = "champion"
 
 
class ModelService:
    """Loads the champion model + explainer once, at startup."""
 
    def __init__(self):
        model_dir = os.getenv("MODEL_DIR")
        if model_dir:
            config = self._load_from_bundle(Path(model_dir))      # Docker / production
        else:
            config = self._load_from_registry()                   # local development
 
        if set(config["input_features"]) != set(MODEL_COLUMNS):
            raise RuntimeError("API schema and trained model use different input columns.")
        self.threshold = float(config["threshold"])
        self.base_rate = float(config["base_rate"])
        self.version = str(config["model_version"])
 
        self.tree_explainer = shap.TreeExplainer(self.explain_pipe.named_steps["model"])
 
    def _load_from_registry(self) -> dict:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
        client = MlflowClient()
 
        mv = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
        ev = client.get_model_version_by_alias(EXPLAINER_NAME, ALIAS)
        if mv.run_id != ev.run_id:
            raise RuntimeError(
                "credit_scoring and credit_scoring_explainer champions come from different runs."
            )
 
        self.model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{ALIAS}")
        self.explain_pipe = mlflow.sklearn.load_model(f"models:/{EXPLAINER_NAME}@{ALIAS}")
 
        config = mlflow.artifacts.load_dict(f"runs:/{mv.run_id}/config.json")
        config["model_version"] = str(mv.version)
        return config
 
    def _load_from_bundle(self, path: Path) -> dict:
        """Load from a folder created by export_model.py (no MLflow database needed)."""
        self.model = mlflow.sklearn.load_model(str(path / "model"))
        self.explain_pipe = mlflow.sklearn.load_model(str(path / "explainer"))
        return json.loads((path / "config.json").read_text())
 
    # ------------------------------------------------------------------ inference
    def _risk_level(self, p: float) -> str:
        """low: below the average default rate | medium: below the threshold | high: rejected."""
        if p < self.base_rate:
            return "low"
        return "medium" if p < self.threshold else "high"
 
    def predict(self, df: pd.DataFrame) -> dict:
        p = float(self.model.predict_proba(df)[:, 1][0])
        return {
            "default_probability": round(p, 4),
            "risk_level": self._risk_level(p),
            "decision": "reject" if p >= self.threshold else "approve",
            "threshold": self.threshold,
            "model_version": self.version,
        }
 
    def explain(self, df: pd.DataFrame, top_k: int = 5) -> list[dict]:
        """Top-k SHAP contributions, computed on the uncalibrated XGBoost (log-odds)."""
        prepared = self.explain_pipe.named_steps["prep"].transform(df)
        shap_row = self.tree_explainer(prepared).values[0]
        contributions = []
        for name, raw, sv in zip(prepared.columns, prepared.iloc[0], shap_row):
            value = None if raw is None or math.isnan(float(raw)) else float(raw)
            contributions.append({
                "feature": name,
                "value": value,
                "shap_value": float(sv),
                "effect": "increases_risk" if sv > 0 else "decreases_risk",
            })
        contributions.sort(key=lambda c: abs(c["shap_value"]), reverse=True)
        return contributions[:top_k]
 
