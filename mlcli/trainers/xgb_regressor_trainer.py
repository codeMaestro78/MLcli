"""
XGBoost Regressor Trainer

XGBoost-based trainer for regression tasks.
"""

import numpy as np
import pickle
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

try:
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from mlcli.trainers.base_trainer import BaseTrainer
from mlcli.utils.registry import register_model
from mlcli.utils.metrics import compute_metrics

logger = logging.getLogger(__name__)


@register_model(
    name="xgb_regressor",
    description="XGBoost gradient boosting regressor",
    framework="xgboost",
    model_type="regression",
)
class XGBRegressorTrainer(BaseTrainer):
    """
    Trainer for XGBoost Regressor models.

    Gradient boosting with regularization for regression tasks.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize XGBoost Regressor trainer.

        Args:
            config: Configuration dictionary with model parameters
        """
        if not HAS_XGBOOST:
            raise ImportError("XGBoost is not installed. Install with: pip install xgboost")

        super().__init__(config)

        params = self.config.get("params", {})
        default_params = self.get_default_params()
        self.model_params = {**default_params, **params}

        logger.info(
            f"Initialized XGBRegressorTrainer with n_estimators={self.model_params.get('n_estimators', 100)}"
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Train XGBoost Regressor model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)

        Returns:
            Training history
        """
        logger.info(f"Training XGBoost Regressor on {X_train.shape[0]} samples")

        # Train model
        self.model = xgb.XGBRegressor(**self.model_params)

        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

        # Compute training metrics
        y_train_pred = self.model.predict(X_train)

        train_metrics = compute_metrics(y_train, y_train_pred, y_pred_proba=None, task="regression")

        # Feature importance
        feature_importance = self.model.feature_importances_.tolist()

        self.training_history = {
            "train_metrics": train_metrics,
            "feature_importance": feature_importance,
            "n_features": X_train.shape[1],
        }

        # Validation metrics
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            self.training_history["val_metrics"] = val_metrics

        self.is_trained = True
        logger.info(f"Training complete. R2: {train_metrics['r2']:.4f}")

        return self.training_history

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate XGBoost Regressor model.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Evaluation metrics (mse, rmse, mae, r2)
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        y_pred = self.model.predict(X_test)

        metrics = compute_metrics(y_test, y_pred, y_pred_proba=None, task="regression")

        logger.info(f"Evaluation - MSE: {metrics['mse']:.4f}, R2: {metrics['r2']:.4f}")

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input features

        Returns:
            Predicted values
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        Predict probabilities (not supported for regression).

        Args:
            X: Input features

        Returns:
            None (regression models don't have class probabilities)
        """
        return None

    def save(self, path: Path, formats: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Save the trained model.

        Args:
            path: Directory to save model
            formats: Save formats (pickle, joblib, json)

        Returns:
            Dictionary of format -> path
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Cannot save.")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        formats = formats or ["pickle"]
        saved_paths = {}

        for fmt in formats:
            if fmt == "pickle":
                model_path = path / "xgb_regressor.pkl"
                with open(model_path, "wb") as f:
                    pickle.dump(self.model, f)
                saved_paths["pickle"] = str(model_path)
                logger.info(f"Saved model to {model_path}")

            elif fmt == "joblib":
                model_path = path / "xgb_regressor.joblib"
                joblib.dump(self.model, model_path)
                saved_paths["joblib"] = str(model_path)
                logger.info(f"Saved model to {model_path}")

            elif fmt == "json":
                model_path = path / "xgb_regressor.json"
                self.model.save_model(model_path)
                saved_paths["json"] = str(model_path)
                logger.info(f"Saved model to {model_path}")

        return saved_paths

    def load(self, path: Path, format: str = "pickle") -> None:
        """
        Load a trained model.

        Args:
            path: Path to model file
            format: Model format
        """
        path = Path(path)

        if format == "pickle":
            with open(path, "rb") as f:
                self.model = pickle.load(f)
        elif format == "joblib":
            self.model = joblib.load(path)
        elif format == "json":
            self.model = xgb.XGBRegressor()
            self.model.load_model(path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        self.is_trained = True
        logger.info(f"Loaded model from {path}")

    @staticmethod
    def get_default_params() -> Dict[str, Any]:
        """Get default model parameters."""
        return {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0,
            "reg_lambda": 1,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        }
