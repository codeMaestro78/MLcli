"""
TensorFlow Dense Neural Network Trainer

Keras-based trainer for fully-connected deep neural networks.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

from tensorflow import keras
from tensorflow.keras import layers, models, callbacks

from mlcli.trainers.base_trainer import BaseTrainer
from mlcli.utils.registry import register_model
from mlcli.utils.metrics import compute_metrics

logger = logging.getLogger(__name__)


@register_model(
    name="tf_dnn",
    description="Tensorflow Dense Feedforward Neural Network",
    framework="tensorflow",
    model_type="classification",
)
class TFDNNTrainer(BaseTrainer):
    """
    Trainer for TensorFlow/Keras Dense Neural Networks.

    Supports dynamic layer construction, dropout regularization,
    batch normalization, and various optimizers.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize TensorFlow DNN trainer.

        Args:
            config: Configuration dictionary with model parameters
        """
        super().__init__(config)

        params = self.config.get("params", {})
        default_params = self.get_default_params()
        self.model_params = {**default_params, **params}

        # Architecture configuration
        self.layers_config = self.model_params.get("layers", [128, 64, 32])
        self.activation = self.model_params.get("activation", "relu")
        self.dropout = self.model_params.get("dropout", 0.2)
        self.use_batch_norm = self.model_params.get("batch_normalization", False)

        # Training configuration
        self.optimizer = self.model_params.get("optimizer", "adam")
        self.learning_rate = self.model_params.get("learning_rate", 0.001)
        self.loss = self.model_params.get("loss", "sparse_categorical_crossentropy")
        self.epochs = self.model_params.get("epochs", 20)
        self.batch_size = self.model_params.get("batch_size", 32)
        self.validation_split = self.model_params.get("validation_split", 0.0)

        # Callbacks
        self.early_stopping_patience = self.model_params.get("early_stopping_patience", 5)
        self.reduce_lr_patience = self.model_params.get("reduce_lr_patience", 3)

        logger.info(f"Initialized TFDNNTrainer with architecture: {self.layers_config}")

    def _build_model(self, input_dim: int, n_classes: int) -> keras.Model:
        """
        Build dense neural network architecture.

        Args:
            input_dim: Number of input features
            n_classes: Number of output classes

        Returns:
            Compiled Keras model
        """

        model = models.Sequential(name="DenseNN")

        # Input layer
        model.add(layers.Input(shape=(input_dim,)))

        # Hidden layers
        for i, units in enumerate(self.layers_config):
            model.add(layers.Dense(units, activation=None, name=f"dense_{i + 1}"))

            # Batch normalization
            if self.use_batch_norm:
                model.add(layers.BatchNormalization(name=f"batch_norm_{i + 1}"))

            # Activation
            model.add(layers.Activation(self.activation, name=f"activation_{i + 1}"))

            # Dropout
            if self.dropout > 0:
                model.add(layers.Dropout(self.dropout, name=f"dropout_{i + 1}"))

        # Output layer
        if n_classes == 2:
            # Binary classification
            model.add(layers.Dense(1, activation="sigmoid", name="output"))
            output_loss = "binary_crossentropy"
        else:
            # Multi - class classification
            model.add(layers.Dense(n_classes, activation="softmax", name="output"))

        # Compile model
        optimizer_instance = self._get_optimizer()

        model.compile(
            optimizer=optimizer_instance,
            loss=output_loss,
            metrics=["accuracy", "precision", "recall"],
        )

        logger.info(f"Built DNN model with {model.count_params()} parameters")
        return model

    def _get_optimizer(self) -> keras.optimizers.Optimizer:
        """
        Get optimizer instance.

        Returns:
            Keras optimizer
        """
        optimizer_name = self.optimizer.lower()

        if optimizer_name == "adam":
            return keras.optimizers.Adam(learning_rate=self.learning_rate)
        elif optimizer_name == "sgd":
            return keras.optimizers.SGD(learning_rate=self.learning_rate)
        elif optimizer_name == "rmsprop":
            return keras.optimizers.RMSprop(learning_rate=self.learning_rate)
        elif optimizer_name == "adamw":
            return keras.optimizers.AdamW(learning_rate=self.learning_rate)
        else:
            logger.warning(f"Unknown optimizer {optimizer_name}, using Adam")
            return keras.optimizers.Adam(learning_rate=self.learning_rate)

    def _get_callbacks(self, X_val: Optional[np.ndarray] = None) -> List[callbacks.Callback]:
        """
        Get training callbacks.

        Args:
            X_val: Validation data (to determine if validation callbacks needed)

        Returns:
            List of Keras callbacks
        """
        callbacks_list = []

        # Early stopping (only if validation data available)
        if X_val is not None or self.validation_split > 0:
            callbacks_list.append(
                callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.early_stopping_patience,
                    restore_best_weights=True,
                    verbose=1,
                )
            )
            # Reduce learning rare on plateau
            callbacks_list.append(
                callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=self.reduce_lr_patience,
                    min_lr=1e-7,
                    verbose=1,
                )
            )
        return callbacks_list

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Train DNN model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            Training history with metrics
        """
        logger.info(f"Training DNN on {X_train.shape[0]} samples")

        # Get dimensions
        input_dim = X_train.shape[1]
        n_classes = len(np.unique(y_train))

        # Build model
        self.model = self._build_model(input_dim, n_classes)

        # Print model summary
        logger.info("Model architecture:")
        self.model.summary(print_fn=logger.info)

        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)

        # Get callbacks
        callback_list = self._get_callbacks(X_val)

        # Train model
        history = self.model.fit(
            X_train,
            y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=validation_data,
            validation_split=self.validation_split if validation_data is None else 0.0,
            callbacks=callback_list,
            verbose=1,
        )

        # Store training history
        self.training_history = {
            "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
            "epochs_trained": len(history.history["loss"]),
            "n_features": input_dim,
            "n_classes": n_classes,
            "total_params": self.model.count_params(),
        }

        # Compute final training metrics
        y_train_pred = self.predict(X_train)
        y_train_proba = self.predict_proba(X_train)

        train_metrics = compute_metrics(y_train, y_train_pred, y_train_proba, task="classification")

        self.training_history["train_metrics"] = train_metrics

        # Validation metrics
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            self.training_history["val_metrics"] = val_metrics

        self.is_trained = True
        logger.info(f"Training complete. Final accuracy: {train_metrics['accuracy']:.4f}")

        return self.training_history

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate DNN model.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        # Keras evaluation
        loss, accuracy, precision, recall = self.model.evaluate(X_test, y_test, verbose=0)

        # Detailed metrics
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        metrics = compute_metrics(y_test, y_pred, y_proba, task="classification")

        metrics["loss"] = float(loss)

        logger.info(f"Evaluation complete. Accuracy: {metrics['accuracy']:.4f}")

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Input features

        Returns:
            Predicted labels
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        y_proba = self.model.predict(X, verbose=0)

        # Convert probabilities to class labels
        if y_proba.shape[1] == 1:
            # Binary classification
            y_pred = (y_proba > 0.5).astype(int).flatten()
        else:
            # Multi-class classification
            y_pred = np.argmax(y_proba, axis=1)

        return y_pred

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Input features

        Returns:
            Predicted probabilities
        """
        if self.model is None:
            raise RuntimeError("Model not trained, Call train() first.")

        y_proba = self.model.predict(X, verbose=0)

        # For binary classification, expand to 2 columns
        if y_proba.shape[1] == 1:
            y_proba = np.hstack([1 - y_proba, y_proba])

        return y_proba

    def save(self, save_dir: Path, formats: List[str]) -> Dict[str, Path]:
        """
        Save DNN model.

        Args:
            save_dir: Directory to save models
            formats: List of formats ('h5', 'savedmodel')

        Returns:
            Dictionary of saved paths
        """
        if self.model is None:
            raise RuntimeError("No model to save. Train model first.")

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = {}

        for fmt in formats:
            if fmt == "h5":
                path = save_dir / "dnn_model.h5"
                self.model.save(str(path))
                saved_paths["h5"] = path
                logger.info(f"Saved H5 model to {path}")

            elif fmt == "savedmodel":
                path = save_dir / "dnn_model.keras"
                self.model.save(str(path))
                saved_paths["savedmodel"] = path
                logger.info(f"Saved Keras model to {path}")

            elif fmt in ["pickle", "joblib"]:
                logger.warning(
                    f"Format {fmt} not recommended for TensorFlow models. "
                    f"Use 'h5' or 'savedmodel' instead."
                )

            else:
                logger.warning(f"Unsupported format for TensorFlow model: {fmt}")

        return saved_paths

    def load(self, model_path: Path, model_format: str) -> None:
        """
        Load DNN model.

        Args:
            model_path: Path to model file/directory
            model_format: Format ('h5', 'savedmodel')
        """
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")

        if model_format == "h5":
            self.model = keras.models.load_model(str(model_path))

        elif model_format == "savedmodel":
            self.model = keras.models.load_model(str(model_path))

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
            "layers": [128, 64, 32],
            "activation": "relu",
            "dropout": 0.2,
            "batch_normalization": False,
            "optimizer": "adam",
            "learning_rate": 0.001,
            "loss": "sparse_categorical_crossentropy",
            "epochs": 20,
            "batch_size": 32,
            "validation_split": 0.0,
            "early_stopping_patience": 5,
            "reduce_lr_patience": 3,
        }
