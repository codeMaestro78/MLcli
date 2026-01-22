"""
K-Means Clustering Trainer

K-Means clustering algorithm for unsupervised learning.
"""

import numpy as np
import pickle
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, List
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
import logging

from mlcli.trainers.base_trainer import BaseTrainer
from mlcli.utils.registry import register_model

logger = logging.getLogger(__name__)


@register_model(
    name="kmeans",
    description="K-Means clustering algorithm",
    framework="sklearn",
    model_type="clustering",
)
class KMeansTrainer(BaseTrainer):
    """
    Trainer for K-Means clustering.

    K-Means is a popular clustering algorithm that partitions data into
    k clusters, where each observation belongs to the cluster with the
    nearest mean (cluster center).

    Features:
    - Fast and efficient for large datasets
    - Simple to understand and implement
    - Works well with spherical clusters
    - Elbow method for optimal k selection
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize K-Means trainer.

        Args:
            config: Configuration dictionary with model parameters
        """
        super().__init__(config)

        params = self.config.get("params", {})
        default_params = self.get_default_params()
        self.model_params = {**default_params, **params}

        logger.info(f"Initialized KMeansTrainer with n_clusters={self.model_params['n_clusters']}")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray = None,  # Ignored for clustering
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Train K-Means clustering model.

        Args:
            X_train: Training features
            y_train: Ignored (unsupervised learning)
            X_val: Validation features (optional, for evaluation)
            y_val: Ignored

        Returns:
            Training history with clustering metrics
        """
        logger.info(
            f"Training K-Means on {X_train.shape[0]} samples with "
            f"n_clusters={self.model_params['n_clusters']}"
        )

        # Create and fit model
        self.model = KMeans(**self.model_params)
        self.model.fit(X_train)

        # Get cluster labels
        labels = self.model.labels_

        # Compute clustering metrics
        train_metrics = self._compute_clustering_metrics(X_train, labels)

        self.training_history = {
            "train_metrics": train_metrics,
            "n_features": X_train.shape[1],
            "n_samples": X_train.shape[0],
            "n_clusters": self.model_params["n_clusters"],
            "inertia": self.model.inertia_,
            "n_iter": self.model.n_iter_,
            "cluster_centers": self.model.cluster_centers_.tolist(),
            "cluster_sizes": np.bincount(labels).tolist(),
        }

        # Validation metrics
        if X_val is not None:
            val_labels = self.model.predict(X_val)
            val_metrics = self._compute_clustering_metrics(X_val, val_labels)
            self.training_history["val_metrics"] = val_metrics

        self.is_trained = True

        logger.info(
            f"Training complete. Silhouette: {train_metrics['silhouette']:.4f}, "
            f"Inertia: {self.model.inertia_:.4f}"
        )

        return self.training_history

    def _compute_clustering_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Compute clustering evaluation metrics.

        Args:
            X: Feature data
            labels: Cluster labels

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Only compute if we have more than 1 cluster and less than n_samples clusters
        n_labels = len(set(labels))
        if 1 < n_labels < len(X):
            metrics["silhouette"] = float(silhouette_score(X, labels))
            metrics["calinski_harabasz"] = float(calinski_harabasz_score(X, labels))
            metrics["davies_bouldin"] = float(davies_bouldin_score(X, labels))
        else:
            metrics["silhouette"] = 0.0
            metrics["calinski_harabasz"] = 0.0
            metrics["davies_bouldin"] = float("inf")

        return metrics

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray = None) -> Dict[str, float]:
        """
        Evaluate clustering on test data.

        Args:
            X_test: Test features
            y_test: Ignored for clustering

        Returns:
            Clustering metrics
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        labels = self.model.predict(X_test)
        metrics = self._compute_clustering_metrics(X_test, labels)

        logger.info(f"Evaluation complete. Silhouette: {metrics['silhouette']:.4f}")

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.

        Args:
            X: Input features

        Returns:
            Predicted cluster labels
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        K-Means doesn't support probability predictions.
        Returns distance to each cluster center instead.

        Args:
            X: Input features

        Returns:
            Distances to cluster centers (inverse can be used as pseudo-probability)
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        # Return distances to cluster centers
        return self.model.transform(X)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit model and predict cluster labels in one step.

        Args:
            X: Input features

        Returns:
            Cluster labels
        """
        self.train(X)
        return self.model.labels_

    def get_cluster_centers(self) -> np.ndarray:
        """
        Get cluster center coordinates.

        Returns:
            Array of cluster centers
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        return self.model.cluster_centers_

    def find_optimal_k(self, X: np.ndarray, k_range: range = range(2, 11)) -> Dict[str, Any]:
        """
        Find optimal number of clusters using elbow method and silhouette analysis.

        Args:
            X: Input features
            k_range: Range of k values to try

        Returns:
            Dictionary with metrics for each k
        """
        results = {
            "k_values": [],
            "inertias": [],
            "silhouettes": [],
            "calinski_harabasz": [],
            "davies_bouldin": [],
        }

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.model_params.get("random_state", 42))
            labels = kmeans.fit_predict(X)

            results["k_values"].append(k)
            results["inertias"].append(kmeans.inertia_)

            if k > 1:
                results["silhouettes"].append(silhouette_score(X, labels))
                results["calinski_harabasz"].append(calinski_harabasz_score(X, labels))
                results["davies_bouldin"].append(davies_bouldin_score(X, labels))
            else:
                results["silhouettes"].append(0)
                results["calinski_harabasz"].append(0)
                results["davies_bouldin"].append(float("inf"))

        # Find optimal k based on silhouette score
        optimal_k = k_range[np.argmax(results["silhouettes"])]
        results["optimal_k"] = optimal_k

        logger.info(f"Optimal k based on silhouette score: {optimal_k}")

        return results

    def save(self, save_dir: Path, formats: List[str]) -> Dict[str, Path]:
        """
        Save K-Means model.

        Args:
            save_dir: Directory to save models
            formats: List of formats ('pickle', 'joblib')

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
                path = save_dir / "kmeans_model.pkl"
                with open(path, "wb") as f:
                    pickle.dump(
                        {
                            "model": self.model,
                            "config": self.config,
                            "training_history": self.training_history,
                        },
                        f,
                    )
                saved_paths["pickle"] = path
                logger.info(f"Saved pickle model to {path}")

            elif fmt == "joblib":
                path = save_dir / "kmeans_model.joblib"
                joblib.dump(
                    {
                        "model": self.model,
                        "config": self.config,
                        "training_history": self.training_history,
                    },
                    path,
                )
                saved_paths["joblib"] = path
                logger.info(f"Saved joblib model to {path}")

            else:
                logger.warning(f"Unsupported format for K-Means: {fmt}")

        return saved_paths

    def load(self, model_path: Path, model_format: str) -> None:
        """
        Load K-Means model.

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
                self.training_history = data.get("training_history", {})

        elif model_format == "joblib":
            data = joblib.load(model_path)
            self.model = data["model"]
            self.config = data.get("config", {})
            self.training_history = data.get("training_history", {})

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
            "n_clusters": 8,
            "init": "k-means++",
            "n_init": 10,
            "max_iter": 300,
            "tol": 1e-4,
            "random_state": 42,
            "algorithm": "lloyd",
        }

    @classmethod
    def get_param_grid(cls) -> Dict[str, List[Any]]:
        """
        Get parameter grid for hyperparameter tuning.

        Returns:
            Dictionary of parameter ranges
        """
        return {
            "n_clusters": [2, 3, 4, 5, 6, 7, 8, 10, 12, 15],
            "init": ["k-means++", "random"],
            "n_init": [10, 20, 30],
            "max_iter": [100, 300, 500],
            "algorithm": ["lloyd", "elkan"],
        }
