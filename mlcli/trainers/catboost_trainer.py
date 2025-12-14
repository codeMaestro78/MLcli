"""
CatBoost Trainer

CatBoost-based trainer for gradient boosting classification and regression.
CatBoost handles categorical features automatically and is robust to overfitting.
"""

import numpy as np
import pickle
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, List
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
import logging

from mlcli.trainers.base_trainer import BaseTrainer
from mlcli.utils.registry import register_model
from mlcli.utils.metrics import compute_metrics

logger = logging.getLogger(__name__)


@register_model(
    name="catboost",
    description="CatBoost gradient boosting classifier/regressor",
    framework="catboost",
    model_type="classification",
)
class CatBoostTrainer(BaseTrainer):
    """
    Trainer for CatBoost models.

    CatBoost is a gradient boosting library that provides:
    - Automatic handling of categorical features
    - Robust to overfitting with ordered boosting
    - GPU training support
    - Fast prediction
    - Built-in cross-validation
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize CatBoost trainer.

        Args:
            config: Configuration dictionary with model parameters
        """
        super().__init__(config)

        params = self.config.get("params", {})
        default_params = self.get_default_params()
        self.model_params = {**default_params, **params}

        # Task type: classification or regression
        self.task_type = self.config.get("task_type", "classification")

        # Categorical feature indices (if any)
        self.cat_features = self.config.get("cat_features", None)

        # Early stopping configuration
        self.early_stopping_rounds = self.config.get("early_stopping_rounds", 50)

        logger.info(
            f"Initialized CatBoostTrainer with iterations={self.model_params['iterations']}, "
            f"task_type={self.task_type}"
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Train CatBoost model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            Training history
        """
        logger.info(f"Training CatBoost on {X_train.shape[0]} samples")

        # Create model params copy and add early stopping if validation set exists
        fit_params = self.model_params.copy()

        if X_val is not None and y_val is not None:
            fit_params["early_stopping_rounds"] = self.early_stopping_rounds

        # Choose classifier or regressor based on task type
        if self.task_type == "regression":
            self.model = CatBoostRegressor(**fit_params)
        else:
            self.model = CatBoostClassifier(**fit_params)

        # Create Pool objects for efficient training
        train_pool = Pool(X_train, y_train, cat_features=self.cat_features)

        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = Pool(X_val, y_val, cat_features=self.cat_features)

        # Train model
        self.model.fit(
            train_pool,
            eval_set=eval_set,
            verbose=False,
            plot=False,
        )

        # Compute training metrics
        y_train_pred = self.model.predict(X_train)

        if self.task_type == "classification":
            y_train_proba = self.model.predict_proba(X_train)
            train_metrics = compute_metrics(
                y_train, y_train_pred, y_train_proba, task="classification"
            )
        else:
            train_metrics = compute_metrics(y_train, y_train_pred, task="regression")

        # Feature importance
        feature_importance = self.model.get_feature_importance().tolist()

        self.training_history = {
            "train_metrics": train_metrics,
            "feature_importance": feature_importance,
            "n_features": X_train.shape[1],
            "best_iteration": self.model.get_best_iteration(),
            "task_type": self.task_type,
        }

        if self.task_type == "classification":
            self.training_history["n_classes"] = len(np.unique(y_train))

        # Validation metrics
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            self.training_history["val_metrics"] = val_metrics

        self.is_trained = True

        metric_key = "accuracy" if self.task_type == "classification" else "r2"
        logger.info(f"Training complete. {metric_key}: {train_metrics.get(metric_key, 0):.4f}")

        best_iter = self.training_history.get("best_iteration")
        if best_iter is not None:
            logger.info(f"Best iteration: {best_iter}")

        return self.training_history

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate CatBoost model.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        y_pred = self.model.predict(X_test)

        if self.task_type == "classification":
            y_proba = self.model.predict_proba(X_test)
            metrics = compute_metrics(y_test, y_pred, y_proba, task="classification")
            logger.info(f"Evaluation complete. Accuracy: {metrics['accuracy']:.4f}")
        else:
            metrics = compute_metrics(y_test, y_pred, task="regression")
            logger.info(f"Evaluation complete. R2: {metrics['r2']:.4f}")

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels or values.

        Args:
            X: Input features

        Returns:
            Predicted labels/values
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        Predict class probabilities (classification only).

        Args:
            X: Input features

        Returns:
            Predicted probabilities or None for regression
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        if self.task_type == "regression":
            return None

        return self.model.predict_proba(X)

    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """
        Get feature importance scores.

        Returns:
            Dictionary with different importance types
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        return {
            "prediction_values_change": self.model.get_feature_importance(
                type="PredictionValuesChange"
            ),
            "loss_function_change": self.model.get_feature_importance(
                type="LossFunctionChange"
            ),
        }

    def save(self, save_dir: Path, formats: List[str]) -> Dict[str, Path]:
        """
        Save CatBoost model.

        Args:
            save_dir: Directory to save models
            formats: List of formats ('pickle', 'joblib', 'catboost', 'onnx')

        Returns:
            Dictionary of saved paths
        """
        if self.model is None:
            raise RuntimeError("No model to save. Train model first.")

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = {}

        for fmt in formats:
            if fmt == "pickle":
                path = save_dir / "catboost_model.pkl"
                with open(path, "wb") as f:
                    pickle.dump(
                        {
                            "model": self.model,
                            "config": self.config,
                            "task_type": self.task_type,
                            "cat_features": self.cat_features,
                        },
                        f,
                    )
                saved_paths["pickle"] = path
                logger.info(f"Saved pickle model to {path}")

            elif fmt == "joblib":
                path = save_dir / "catboost_model.joblib"
                joblib.dump(
                    {
                        "model": self.model,
                        "config": self.config,
                        "task_type": self.task_type,
                        "cat_features": self.cat_features,
                    },
                    path,
                )
                saved_paths["joblib"] = path
                logger.info(f"Saved joblib model to {path}")

            elif fmt == "catboost":
                # Native CatBoost format
                path = save_dir / "catboost_model.cbm"
                self.model.save_model(str(path))
                saved_paths["catboost"] = path
                logger.info(f"Saved CatBoost model to {path}")

            elif fmt == "onnx":
                path = save_dir / "catboost_model.onnx"
                try:
                    self.model.save_model(
                        str(path),
                        format="onnx",
                        export_parameters={
                            "onnx_domain": "ai.catboost",
                            "onnx_model_version": 1,
                        },
                    )
                    saved_paths["onnx"] = path
                    logger.info(f"Saved ONNX model to {path}")
                except Exception as e:
                    logger.error(f"Failed to save ONNX model: {e}")

            else:
                logger.warning(f"Unsupported format: {fmt}")

        return saved_paths

    def load(self, model_path: Path, model_format: str) -> None:
        """
        Load CatBoost model.

        Args:
            model_path: Path to model file
            model_format: Format of the model
        """
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if model_format == "pickle":
            with open(model_path, "rb") as f:
                data = pickle.load(f)
                self.model = data["model"]
                self.config = data.get("config", {})
                self.task_type = data.get("task_type", "classification")
                self.cat_features = data.get("cat_features", None)

        elif model_format == "joblib":
            data = joblib.load(model_path)
            self.model = data["model"]
            self.config = data.get("config", {})
            self.task_type = data.get("task_type", "classification")
            self.cat_features = data.get("cat_features", None)

        elif model_format == "catboost":
            # Load native CatBoost format
            if self.task_type == "regression":
                self.model = CatBoostRegressor()
            else:
                self.model = CatBoostClassifier()
            self.model.load_model(str(model_path))

        elif model_format == "onnx":
            import onnxruntime as ort

            self.model = ort.InferenceSession(str(model_path))

        else:
            raise ValueError(f"Unsupported format: {model_format}")

        self.is_trained = True
        logger.info(f"Loaded {model_format} model from {model_path}")

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """
        Get default hyperparameters.

        Returns:
            Default parameters
        """
        return {
            "iterations": 100,
            "depth": 6,
            "learning_rate": 0.1,
            "l2_leaf_reg": 3.0,
            "border_count": 254,
            "random_seed": 42,
            "thread_count": -1,
            "verbose": False,
            "allow_writing_files": False,
        }

    @classmethod
    def get_param_grid(cls) -> Dict[str, List[Any]]:
        """
        Get parameter grid for hyperparameter tuning.

        Returns:
            Dictionary of parameter ranges
        """
        return {
            "iterations": [50, 100, 200, 300, 500],
            "depth": [4, 6, 8, 10],
            "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
            "l2_leaf_reg": [1.0, 3.0, 5.0, 10.0],
            "border_count": [32, 64, 128, 254],
            "bagging_temperature": [0.0, 0.5, 1.0],
            "random_strength": [0.0, 0.5, 1.0, 2.0],
        }
