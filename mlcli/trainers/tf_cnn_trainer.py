"""
TensorFlow Convolutional Neural Network Trainer

Keras-based trainer for image classification using CNNs.
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
    name="tf_cnn",
    description="TensorFlow Convolutional Neural Network for image classification",
    framework="tensorflow",
    model_type="classification",
)
class TFCNNTrainer(BaseTrainer):
    """
    Trainer for TensorFlow/Keras CNNs.

    Supports custom Conv2D architectures, pooling strategies,
    data augmentation, and transfer learning preparation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize TensorFlow CNN trainer.

        Args:
            config: Configuration dictionary with model parameters
        """
        super().__init__(config)

        params = self.config.get("params", {})
        default_params = self.get_default_params()
        self.model_params = {**default_params, **params}

        # Architecture configuration
        self.conv_layers = self.model_params.get("conv_layers", [32, 64, 128])
        self.kernel_size = self.model_params.get("kernel_size", 3)
        self.pool_size = self.model_params.get("pool_size", 2)
        self.dense_layers = self.model_params.get("dense_layers", [128])
        self.dropout = self.model_params.get("dropout", 0.3)
        self.use_batch_norm = self.model_params.get("batch_normalization", True)

        # Input configuration
        self.input_shape = tuple(self.model_params.get("input_shape", [224, 224, 3]))

        # Training configuration
        self.optimizer = self.model_params.get("optimizer", "adam")
        self.learning_rate = self.model_params.get("learning_rate", 0.001)
        self.epochs = self.model_params.get("epochs", 30)
        self.batch_size = self.model_params.get("batch_size", 32)

        # Data augmentation
        self.use_augmentation = self.model_params.get("data_augmentation", False)

        logger.info(f"Initialized TFCNNTrainer with input_shape={self.input_shape}")

    def _build_model(self, n_classes: int) -> keras.Model:
        """
        Build CNN architecture.

        Args:
            n_classes: Number of output classes

        Returns:
            Compiled Keras model
        """
        model = models.Sequential(name="CNN")

        # Input layer
        model.add(layers.Input(shape=self.input_shape))

        # Data augmentation layers (if enabled)
        if self.use_augmentation:
            model.add(layers.RandomFlip("horizontal"))
            model.add(layers.RandomRotation(0.1))
            model.add(layers.RandomZoom(0.1))

        # Convolutional blocks
        for i, filters in enumerate(self.conv_layers):
            # Convolutional layer
            model.add(
                layers.Conv2D(
                    filters,
                    kernel_size=self.kernel_size,
                    padding="same",
                    activation=None,
                    name=f"conv_{i+1}",
                )
            )

            # Batch normalization
            if self.use_batch_norm:
                model.add(layers.BatchNormalization(name=f"batch_norm_{i+1}"))

            # Activation
            model.add(layers.Activation("relu", name=f"relu_{i+1}"))

            # Max pooling
            model.add(layers.MaxPooling2D(pool_size=self.pool_size, name=f"pool_{i+1}"))

            # Dropout
            if self.dropout > 0:
                model.add(layers.Dropout(self.dropout, name=f"dropout_{i+1}"))

        # Flatten
        model.add(layers.Flatten(name="flatten"))

        # Dense layers
        for i, units in enumerate(self.dense_layers):
            model.add(layers.Dense(units, activation="relu", name=f"dense_{i+1}"))

            if self.dropout > 0:
                model.add(layers.Dropout(self.dropout, name=f"dense_dropout_{i+1}"))

        # Output layer
        if n_classes == 2:
            model.add(layers.Dense(1, activation="sigmoid", name="output"))
            output_loss = "binary_crossentropy"
        else:
            model.add(layers.Dense(n_classes, activation="softmax", name="output"))
            output_loss = "sparse_categorical_crossentropy"

        # Compile model
        optimizer_instance = self._get_optimizer()

        model.compile(optimizer=optimizer_instance, loss=output_loss, metrics=["accuracy"])

        logger.info(f"Built CNN model with {model.count_params()} parameters")

        return model

    def _get_optimizer(self) -> keras.optimizers.Optimizer:
        """Get optimizer instance."""
        optimizer_name = self.optimizer.lower()

        if optimizer_name == "adam":
            return keras.optimizers.Adam(learning_rate=self.learning_rate)
        elif optimizer_name == "sgd":
            return keras.optimizers.SGD(
                learning_rate=self.learning_rate, momentum=0.9, nesterov=True
            )
        elif optimizer_name == "rmsprop":
            return keras.optimizers.RMSprop(learning_rate=self.learning_rate)
        else:
            return keras.optimizers.Adam(learning_rate=self.learning_rate)

    def _get_callbacks(self, X_val: Optional[np.ndarray] = None) -> List[callbacks.Callback]:
        """Get training callbacks."""
        callback_list = []

        if X_val is not None:
            # Early stopping
            callback_list.append(
                callbacks.EarlyStopping(
                    monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
                )
            )

            # Reduce learning rate
            callback_list.append(
                callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1
                )
            )

        return callback_list

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Train CNN model.

        Args:
            X_train: Training images (N, H, W, C)
            y_train: Training labels
            X_val: Validation images (optional)
            y_val: Validation labels (optional)

        Returns:
            Training history
        """
        logger.info(f"Training CNN on {X_train.shape[0]} images")

        # Validate input shape
        if len(X_train.shape) != 4:
            raise ValueError(f"Expected 4D input (N, H, W, C), got shape {X_train.shape}")

        # Update input shape from data
        self.input_shape = X_train.shape[1:]

        # Get number of classes
        n_classes = len(np.unique(y_train))

        # Build model
        self.model = self._build_model(n_classes)

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
            callbacks=callback_list,
            verbose=1,
        )

        # Store training history
        self.training_history = {
            "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
            "epochs_trained": len(history.history["loss"]),
            "input_shape": self.input_shape,
            "n_classes": n_classes,
            "total_params": self.model.count_params(),
        }

        # Compute final metrics
        y_train_pred = self.predict(X_train)
        y_train_proba = self.predict_proba(X_train)

        train_metrics = compute_metrics(y_train, y_train_pred, y_train_proba, task="classification")

        self.training_history["train_metrics"] = train_metrics

        # Validation metrics
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            self.training_history["val_metrics"] = val_metrics

        self.is_trained = True
        logger.info(f"Training complete. Accuracy: {train_metrics['accuracy']:.4f}")

        return self.training_history

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate CNN model.

        Args:
            X_test: Test images
            y_test: Test labels

        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        # Keras evaluation
        results = self.model.evaluate(X_test, y_test, verbose=0)
        loss = results[0]

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
            X: Input images

        Returns:
            Predicted labels
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        y_proba = self.model.predict(X, verbose=0)

        if y_proba.shape[1] == 1:
            y_pred = (y_proba > 0.5).astype(int).flatten()
        else:
            y_pred = np.argmax(y_proba, axis=1)

        return y_pred

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Input images

        Returns:
            Predicted probabilities
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        y_proba = self.model.predict(X, verbose=0)

        if y_proba.shape[1] == 1:
            y_proba = np.hstack([1 - y_proba, y_proba])

        return y_proba

    def save(self, save_dir: Path, formats: List[str]) -> Dict[str, Path]:
        """
        Save CNN model.

        Args:
            save_dir: Directory to save models
            formats: List of formats

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
                path = save_dir / "cnn_model.h5"
                self.model.save(str(path), save_format="h5")
                saved_paths["h5"] = path
                logger.info(f"Saved H5 model to {path}")

            elif fmt == "savedmodel":
                path = save_dir / "cnn_savedmodel"
                self.model.save(str(path), save_format="tf")
                saved_paths["savedmodel"] = path
                logger.info(f"Saved SavedModel to {path}")

            else:
                logger.warning(f"Unsupported format for CNN model: {fmt}")

        return saved_paths

    def load(self, model_path: Path, model_format: str) -> None:
        """
        Load CNN model.

        Args:
            model_path: Path to model
            model_format: Format of the model
        """
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")

        if model_format in ["h5", "savedmodel"]:
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
            "conv_layers": [32, 64, 128],
            "kernel_size": 3,
            "pool_size": 2,
            "dense_layers": [128],
            "dropout": 0.3,
            "batch_normalization": True,
            "input_shape": [224, 224, 3],
            "optimizer": "adam",
            "learning_rate": 0.001,
            "epochs": 30,
            "batch_size": 32,
            "data_augmentation": False,
        }
