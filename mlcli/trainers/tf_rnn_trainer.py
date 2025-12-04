"""
TensorFlow Recurrent Neural Network Trainer

Keras-based trainer for sequence classification using RNN/LSTM/GRU.
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
    name="tf_rnn",
    description="TensorFlow RNN/LSTM/GRU for sequence classification",
    framework="tensorflow",
    model_type="classification",
)
class TFRNNTrainer(BaseTrainer):
    """
    Trainer for TensorFlow/Keras RNN models.

    Supports LSTM, GRU, and simple RNN cells with bidirectional
    layers, attention mechanisms, and sequence masking.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize TensorFlow RNN trainer.

        Args:
            config: Configuration dictionary with model parameters
        """
        super().__init__(config)

        params = self.config.get("params", {})
        default_params = self.get_default_params()
        self.model_params = {**default_params, **params}

        # Architecture configuration
        self.rnn_type = self.model_params.get("rnn_type", "lstm")  # lstm, gru, simple_rnn
        self.rnn_units = self.model_params.get("rnn_units", [128, 64])
        self.bidirectional = self.model_params.get("bidirectional", False)
        self.dropout = self.model_params.get("dropout", 0.2)
        self.recurrent_dropout = self.model_params.get("recurrent_dropout", 0.2)

        # Dense layers after RNN
        self.dense_layers = self.model_params.get("dense_layers", [64])

        # Input configuration
        self.max_sequence_length = self.model_params.get("max_sequence_length", 100)
        self.embedding_dim = self.model_params.get("embedding_dim", None)
        self.vocab_size = self.model_params.get("vocab_size", None)

        # Training configuration
        self.optimizer = self.model_params.get("optimizer", "adam")
        self.learning_rate = self.model_params.get("learning_rate", 0.001)
        self.epochs = self.model_params.get("epochs", 25)
        self.batch_size = self.model_params.get("batch_size", 32)

        logger.info(f"Initialized TFRNNTrainer with {self.rnn_type.upper()} architecture")

    def _get_rnn_layer(self, units: int, return_sequences: bool = True):
        """
        Get RNN layer based on configuration.

        Args:
            units: Number of units in the layer
            return_sequences: Whether to return sequences or final output

        Returns:
            RNN layer
        """
        rnn_type = self.rnn_type.lower()

        if rnn_type == "lstm":
            layer = layers.LSTM(
                units,
                return_sequences=return_sequences,
                dropout=self.dropout,
                recurrent_dropout=self.recurrent_dropout,
            )
        elif rnn_type == "gru":
            layer = layers.GRU(
                units,
                return_sequences=return_sequences,
                dropout=self.dropout,
                recurrent_dropout=self.recurrent_dropout,
            )
        elif rnn_type == "simple_rnn":
            layer = layers.SimpleRNN(
                units,
                return_sequences=return_sequences,
                dropout=self.dropout,
                recurrent_dropout=self.recurrent_dropout,
            )
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}")

        # Wrap in Bidirectional if configured
        if self.bidirectional:
            layer = layers.Bidirectional(layer)

        return layer

    def _build_model(self, input_shape: tuple, n_classes: int) -> keras.Model:
        """
        Build RNN architecture.

        Args:
            input_shape: Shape of input sequences
            n_classes: Number of output classes

        Returns:
            Compiled Keras model
        """
        model = models.Sequential(name="RNN")

        # Input layer
        model.add(layers.Input(shape=input_shape))

        # Embedding layer (if vocabulary-based input)
        if self.vocab_size and self.embedding_dim:
            model.add(
                layers.Embedding(
                    input_dim=self.vocab_size,
                    output_dim=self.embedding_dim,
                    mask_zero=True,
                    name="embedding",
                )
            )

        # RNN layers
        for i, units in enumerate(self.rnn_units):
            # Return sequences for all but the last RNN layer
            return_sequences = i < len(self.rnn_units) - 1

            rnn_layer = self._get_rnn_layer(units, return_sequences)
            model.add(rnn_layer)

            # Add dropout after RNN layer
            if self.dropout > 0 and not return_sequences:
                model.add(layers.Dropout(self.dropout))

        # Dense layers
        for i, units in enumerate(self.dense_layers):
            model.add(layers.Dense(units, activation="relu", name=f"dense_{i+1}"))

            if self.dropout > 0:
                model.add(layers.Dropout(self.dropout, name=f"dropout_{i+1}"))

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

        logger.info(f"Built RNN model with {model.count_params()} parameters")

        return model

    def _get_optimizer(self) -> keras.optimizers.Optimizer:
        """Get optimizer instance."""
        optimizer_name = self.optimizer.lower()

        if optimizer_name == "adam":
            return keras.optimizers.Adam(learning_rate=self.learning_rate)
        elif optimizer_name == "rmsprop":
            return keras.optimizers.RMSprop(learning_rate=self.learning_rate)
        elif optimizer_name == "sgd":
            return keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=0.9)
        else:
            return keras.optimizers.Adam(learning_rate=self.learning_rate)

    def _get_callbacks(self, X_val: Optional[np.ndarray] = None) -> List[callbacks.Callback]:
        """Get training callbacks."""
        callback_list = []

        if X_val is not None:
            callback_list.append(
                callbacks.EarlyStopping(
                    monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
                )
            )

            callback_list.append(
                callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1
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
        Train RNN model.

        Args:
            X_train: Training sequences (N, seq_length) or (N, seq_length, features)
            y_train: Training labels
            X_val: Validation sequences (optional)
            y_val: Validation labels (optional)

        Returns:
            Training history
        """
        logger.info(f"Training RNN on {X_train.shape[0]} sequences")

        # Validate input shape
        if len(X_train.shape) < 2:
            raise ValueError(f"Expected 2D or 3D input, got shape {X_train.shape}")

        # Determine input shape (excluding batch dimension)
        input_shape = X_train.shape[1:]

        # Get number of classes
        n_classes = len(np.unique(y_train))

        # Build model
        self.model = self._build_model(input_shape, n_classes)

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
            "input_shape": input_shape,
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
        Evaluate RNN model.

        Args:
            X_test: Test sequences
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
            X: Input sequences

        Returns:
            Predicted labels
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        y_proba = self.model.predict(X, verbose=0)

        if len(y_proba.shape) == 1 or y_proba.shape[1] == 1:
            y_pred = (y_proba > 0.5).astype(int).flatten()
        else:
            y_pred = np.argmax(y_proba, axis=1)

        return y_pred

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Input sequences

        Returns:
            Predicted probabilities
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        y_proba = self.model.predict(X, verbose=0)

        if len(y_proba.shape) == 1 or y_proba.shape[1] == 1:
            y_proba = y_proba.reshape(-1, 1)
            y_proba = np.hstack([1 - y_proba, y_proba])

        return y_proba

    def save(self, save_dir: Path, formats: List[str]) -> Dict[str, Path]:
        """
        Save RNN model.

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
                path = save_dir / "rnn_model.h5"
                self.model.save(str(path), save_format="h5")
                saved_paths["h5"] = path
                logger.info(f"Saved H5 model to {path}")

            elif fmt == "savedmodel":
                path = save_dir / "rnn_savedmodel"
                self.model.save(str(path), save_format="tf")
                saved_paths["savedmodel"] = path
                logger.info(f"Saved SavedModel to {path}")

            else:
                logger.warning(f"Unsupported format for RNN model: {fmt}")

        return saved_paths

    def load(self, model_path: Path, model_format: str) -> None:
        """
        Load RNN model.

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
            "rnn_type": "lstm",
            "rnn_units": [128, 64],
            "bidirectional": False,
            "dropout": 0.2,
            "recurrent_dropout": 0.2,
            "dense_layers": [64],
            "max_sequence_length": 100,
            "embedding_dim": None,
            "vocab_size": None,
            "optimizer": "adam",
            "learning_rate": 0.001,
            "epochs": 25,
            "batch_size": 32,
        }
