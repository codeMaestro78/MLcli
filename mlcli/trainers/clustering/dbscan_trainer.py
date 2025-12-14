"""
DBSCAN Clustering Trainer

Density-Based Spatial Clustering of Applications with Noise (DBSCAN).
"""

import numpy as np
import pickle
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, List
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.neighbors import NearestNeighbors
import logging

from mlcli.trainers.base_trainer import BaseTrainer
from mlcli.utils.registry import register_model

logger = logging.getLogger(__name__)


@register_model(
    name="dbscan",
    description="DBSCAN density-based clustering algorithm",
    framework="sklearn",
    model_type="clustering",
)
class DBSCANTrainer(BaseTrainer):
    """
    Trainer for DBSCAN clustering.

    DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
    is a density-based clustering algorithm that:
    - Discovers clusters of arbitrary shape
    - Handles noise/outliers naturally
    - Does not require specifying number of clusters
    - Works well with varying density clusters

    Key parameters:
    - eps: Maximum distance between two samples in the same neighborhood
    - min_samples: Minimum samples in a neighborhood for a core point
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize DBSCAN trainer.

        Args:
            config: Configuration dictionary with model parameters
        """
        super().__init__(config)

        params = self.config.get("params", {})
        default_params = self.get_default_params()
        self.model_params = {**default_params, **params}

        logger.info(
            f"Initialized DBSCANTrainer with eps={self.model_params['eps']}, "
            f"min_samples={self.model_params['min_samples']}"
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray = None,  # Ignored for clustering
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Train DBSCAN clustering model.

        Args:
            X_train: Training features
            y_train: Ignored (unsupervised learning)
            X_val: Not used for DBSCAN (no predict method for new data)
            y_val: Ignored

        Returns:
            Training history with clustering metrics
        """
        logger.info(
            f"Training DBSCAN on {X_train.shape[0]} samples with "
            f"eps={self.model_params['eps']}, min_samples={self.model_params['min_samples']}"
        )

        # Create and fit model
        self.model = DBSCAN(**self.model_params)
        labels = self.model.fit_predict(X_train)

        # Store training data for later predictions
        self._X_train = X_train
        self._labels = labels

        # Count clusters and noise points
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        # Compute clustering metrics (excluding noise points)
        train_metrics = self._compute_clustering_metrics(X_train, labels)

        self.training_history = {
            "train_metrics": train_metrics,
            "n_features": X_train.shape[1],
            "n_samples": X_train.shape[0],
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_ratio": n_noise / X_train.shape[0],
            "cluster_sizes": self._get_cluster_sizes(labels),
            "core_sample_indices": (
                self.model.core_sample_indices_.tolist()
                if hasattr(self.model, "core_sample_indices_")
                else []
            ),
        }

        self.is_trained = True

        logger.info(f"Training complete. Found {n_clusters} clusters and {n_noise} noise points")
        if train_metrics.get("silhouette") is not None:
            logger.info(f"Silhouette score: {train_metrics['silhouette']:.4f}")

        return self.training_history

    def _get_cluster_sizes(self, labels: np.ndarray) -> Dict[str, int]:
        """
        Get size of each cluster.

        Args:
            labels: Cluster labels

        Returns:
            Dictionary mapping cluster id to size
        """
        unique, counts = np.unique(labels, return_counts=True)
        return {f"cluster_{int(k)}" if k != -1 else "noise": int(v) for k, v in zip(unique, counts)}

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

        # Exclude noise points for metric calculation
        mask = labels != -1
        n_labels = len(set(labels[mask])) if mask.any() else 0

        if n_labels > 1 and mask.sum() > n_labels:
            X_filtered = X[mask]
            labels_filtered = labels[mask]

            metrics["silhouette"] = float(silhouette_score(X_filtered, labels_filtered))
            metrics["calinski_harabasz"] = float(
                calinski_harabasz_score(X_filtered, labels_filtered)
            )
            metrics["davies_bouldin"] = float(davies_bouldin_score(X_filtered, labels_filtered))
        else:
            metrics["silhouette"] = None
            metrics["calinski_harabasz"] = None
            metrics["davies_bouldin"] = None

        return metrics

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray = None) -> Dict[str, float]:
        """
        Evaluate clustering on test data.

        Note: DBSCAN doesn't have a native predict method.
        This uses approximate nearest neighbor assignment.

        Args:
            X_test: Test features
            y_test: Ignored for clustering

        Returns:
            Clustering metrics
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        labels = self.predict(X_test)
        metrics = self._compute_clustering_metrics(X_test, labels)

        if metrics.get("silhouette") is not None:
            logger.info(f"Evaluation complete. Silhouette: {metrics['silhouette']:.4f}")
        else:
            logger.info("Evaluation complete. Unable to compute metrics (insufficient clusters)")

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.

        Note: DBSCAN doesn't have a native predict method.
        This assigns new points to the nearest core point's cluster,
        or labels them as noise if too far from any core point.

        Args:
            X: Input features

        Returns:
            Predicted cluster labels (-1 for noise)
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        # Get core samples
        core_samples = self._X_train[self.model.core_sample_indices_]
        core_labels = self._labels[self.model.core_sample_indices_]

        if len(core_samples) == 0:
            # No core samples, all points are noise
            return np.full(X.shape[0], -1)

        # Find nearest core sample for each new point
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(core_samples)
        distances, indices = nn.kneighbors(X)

        # Assign to cluster if within eps, otherwise noise
        labels = np.where(
            distances.flatten() <= self.model_params["eps"],
            core_labels[indices.flatten()],
            -1,
        )

        return labels

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        DBSCAN doesn't support probability predictions.

        Args:
            X: Input features

        Returns:
            None (DBSCAN doesn't provide probabilities)
        """
        return None

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit model and predict cluster labels in one step.

        Args:
            X: Input features

        Returns:
            Cluster labels
        """
        self.train(X)
        return self._labels

    def find_optimal_eps(self, X: np.ndarray, k: int = None, plot: bool = False) -> float:
        """
        Find optimal eps using k-distance graph method.

        Args:
            X: Input features
            k: Number of nearest neighbors (default: min_samples)
            plot: Whether to display the k-distance graph

        Returns:
            Suggested eps value
        """
        if k is None:
            k = self.model_params.get("min_samples", 5)

        # Compute k-nearest neighbors distances
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(X)
        distances, _ = nn.kneighbors(X)

        # Sort distances to k-th neighbor
        k_distances = np.sort(distances[:, k - 1])

        # Find elbow point using gradient
        gradients = np.gradient(k_distances)
        elbow_idx = np.argmax(gradients)
        optimal_eps = k_distances[elbow_idx]

        logger.info(f"Suggested eps value: {optimal_eps:.4f}")

        return float(optimal_eps)

    def save(self, save_dir: Path, formats: List[str]) -> Dict[str, Path]:
        """
        Save DBSCAN model.

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
                path = save_dir / "dbscan_model.pkl"
                with open(path, "wb") as f:
                    pickle.dump(
                        {
                            "model": self.model,
                            "config": self.config,
                            "training_history": self.training_history,
                            "X_train": self._X_train,
                            "labels": self._labels,
                        },
                        f,
                    )
                saved_paths["pickle"] = path
                logger.info(f"Saved pickle model to {path}")

            elif fmt == "joblib":
                path = save_dir / "dbscan_model.joblib"
                joblib.dump(
                    {
                        "model": self.model,
                        "config": self.config,
                        "training_history": self.training_history,
                        "X_train": self._X_train,
                        "labels": self._labels,
                    },
                    path,
                )
                saved_paths["joblib"] = path
                logger.info(f"Saved joblib model to {path}")

            else:
                logger.warning(f"Unsupported format for DBSCAN: {fmt}")

        return saved_paths

    def load(self, model_path: Path, model_format: str) -> None:
        """
        Load DBSCAN model.

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
                self._X_train = data.get("X_train")
                self._labels = data.get("labels")

        elif model_format == "joblib":
            data = joblib.load(model_path)
            self.model = data["model"]
            self.config = data.get("config", {})
            self.training_history = data.get("training_history", {})
            self._X_train = data.get("X_train")
            self._labels = data.get("labels")

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
            "eps": 0.5,
            "min_samples": 5,
            "metric": "euclidean",
            "algorithm": "auto",
            "leaf_size": 30,
            "n_jobs": -1,
        }

    @classmethod
    def get_param_grid(cls) -> Dict[str, List[Any]]:
        """
        Get parameter grid for hyperparameter tuning.

        Returns:
            Dictionary of parameter ranges
        """
        return {
            "eps": [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
            "min_samples": [3, 5, 10, 15, 20],
            "metric": ["euclidean", "manhattan", "cosine"],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        }
