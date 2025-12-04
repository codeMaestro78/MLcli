"""Model trainers module with lazy loading to avoid slow imports."""

from mlcli.trainers.base_trainer import BaseTrainer

# Lazy imports - TensorFlow trainers are only loaded when accessed
_LAZY_IMPORTS = {
    "LogisticRegressionTrainer": "mlcli.trainers.logistic_trainer",
    "SVMTrainer": "mlcli.trainers.svm_trainer",
    "RFTrainer": "mlcli.trainers.rf_trainer",
    "XGBTrainer": "mlcli.trainers.xgb_trainer",
    "TFDNNTrainer": "mlcli.trainers.tf_dnn_trainer",
    "TFCNNTrainer": "mlcli.trainers.tf_cnn_trainer",
    "TFRNNTrainer": "mlcli.trainers.tf_rnn_trainer",
}

# Pre-register models without importing heavy dependencies
_MODEL_METADATA = {
    "logistic_regression": {
        "class": "LogisticRegressionTrainer",
        "module": "mlcli.trainers.logistic_trainer",
        "description": "Logistic Regression classifier with L2 regularization",
        "framework": "sklearn",
        "model_type": "classification",
    },
    "svm": {
        "class": "SVMTrainer",
        "module": "mlcli.trainers.svm_trainer",
        "description": "Support Vector Machine with RBF/Linear/Poly kernels",
        "framework": "sklearn",
        "model_type": "classification",
    },
    "random_forest": {
        "class": "RFTrainer",
        "module": "mlcli.trainers.rf_trainer",
        "description": "Random Forest ensemble classifier",
        "framework": "sklearn",
        "model_type": "classification",
    },
    "xgboost": {
        "class": "XGBTrainer",
        "module": "mlcli.trainers.xgb_trainer",
        "description": "XGBoost gradient boosting classifier",
        "framework": "xgboost",
        "model_type": "classification",
    },
    "tf_dnn": {
        "class": "TFDNNTrainer",
        "module": "mlcli.trainers.tf_dnn_trainer",
        "description": "Tensorflow Dense Feedforward Neural Network",
        "framework": "tensorflow",
        "model_type": "classification",
    },
    "tf_cnn": {
        "class": "TFCNNTrainer",
        "module": "mlcli.trainers.tf_cnn_trainer",
        "description": "TensorFlow Convolutional Neural Network for image classification",
        "framework": "tensorflow",
        "model_type": "classification",
    },
    "tf_rnn": {
        "class": "TFRNNTrainer",
        "module": "mlcli.trainers.tf_rnn_trainer",
        "description": "TensorFlow RNN/LSTM/GRU for sequence classification",
        "framework": "tensorflow",
        "model_type": "classification",
    },
}


def __getattr__(name: str):
    """Lazy import trainers only when accessed."""
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def register_all_models():
    """Register all models in the registry without importing heavy modules."""
    from mlcli import registry

    for model_name, meta in _MODEL_METADATA.items():
        if not registry.is_registered(model_name):
            # Register with lazy loader
            registry.register_lazy(
                name=model_name,
                module_path=meta["module"],
                class_name=meta["class"],
                description=meta["description"],
                framework=meta["framework"],
                model_type=meta["model_type"],
            )


def get_trainer_class(model_type: str):
    """Get trainer class by model type, importing only when needed."""
    if model_type not in _MODEL_METADATA:
        raise ValueError(f"Unknown model type: {model_type}")

    import importlib

    meta = _MODEL_METADATA[model_type]
    module = importlib.import_module(meta["module"])
    return getattr(module, meta["class"])


__all__ = [
    "BaseTrainer",
    "LogisticRegressionTrainer",
    "SVMTrainer",
    "RFTrainer",
    "XGBTrainer",
    "TFDNNTrainer",
    "TFCNNTrainer",
    "TFRNNTrainer",
    "register_all_models",
    "get_trainer_class",
]
