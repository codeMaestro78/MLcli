"""Model trainers module with lazy loading to avoid slow imports."""

from mlcli.trainers.base_trainer import BaseTrainer

# Lazy imports - Trainers are only loaded when accessed
_LAZY_IMPORTS = {
    # Classification/Regression - sklearn
    "LogisticRegressionTrainer": "mlcli.trainers.logistic_trainer",
    "SVMTrainer": "mlcli.trainers.svm_trainer",
    "RFTrainer": "mlcli.trainers.rf_trainer",
    # Regression - sklearn
    "RFRegressorTrainer": "mlcli.trainers.rf_regressor_trainer",
    # Gradient Boosting - Classification
    "XGBTrainer": "mlcli.trainers.xgb_trainer",
    "LightGBMTrainer": "mlcli.trainers.lightgbm_trainer",
    "CatBoostTrainer": "mlcli.trainers.catboost_trainer",
    # Gradient Boosting - Regression
    "XGBRegressorTrainer": "mlcli.trainers.xgb_regressor_trainer",
    "LightGBMRegressorTrainer": "mlcli.trainers.lightgbm_regressor_trainer",
    # Deep Learning - TensorFlow
    "TFDNNTrainer": "mlcli.trainers.tf_dnn_trainer",
    "TFCNNTrainer": "mlcli.trainers.tf_cnn_trainer",
    "TFRNNTrainer": "mlcli.trainers.tf_rnn_trainer",
    # Deep Learning - PyTorch
    "PyTorchMLPTrainer": "mlcli.trainers.pytorch_mlp_trainer",
    "PyTorchCNNTrainer": "mlcli.trainers.pytorch_cnn_trainer",
    "PyTorchRNNTrainer": "mlcli.trainers.pytorch_rnn_trainer",
    "PyTorchTransformerTrainer": "mlcli.trainers.pytorch_transformer_trainer",
    # Clustering
    "KMeansTrainer": "mlcli.trainers.clustering.kmeans_trainer",
    "DBSCANTrainer": "mlcli.trainers.clustering.dbscan_trainer",
    # Anomaly Detection
    "IsolationForestTrainer": "mlcli.trainers.anomaly.isolation_forest_trainer",
    "OneClassSVMTrainer": "mlcli.trainers.anomaly.one_class_svm_trainer",
}

# Pre-register models without importing heavy dependencies
_MODEL_METADATA = {
    # Classification - sklearn
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
    # Gradient Boosting
    "xgboost": {
        "class": "XGBTrainer",
        "module": "mlcli.trainers.xgb_trainer",
        "description": "XGBoost gradient boosting classifier",
        "framework": "xgboost",
        "model_type": "classification",
    },
    "lightgbm": {
        "class": "LightGBMTrainer",
        "module": "mlcli.trainers.lightgbm_trainer",
        "description": "LightGBM gradient boosting classifier/regressor",
        "framework": "lightgbm",
        "model_type": "classification",
    },
    "catboost": {
        "class": "CatBoostTrainer",
        "module": "mlcli.trainers.catboost_trainer",
        "description": "CatBoost gradient boosting with automatic categorical handling",
        "framework": "catboost",
        "model_type": "classification",
    },
    # Regression - sklearn
    "rf_regressor": {
        "class": "RFRegressorTrainer",
        "module": "mlcli.trainers.rf_regressor_trainer",
        "description": "Random Forest ensemble regressor",
        "framework": "sklearn",
        "model_type": "regression",
    },
    # Regression - Gradient Boosting
    "xgb_regressor": {
        "class": "XGBRegressorTrainer",
        "module": "mlcli.trainers.xgb_regressor_trainer",
        "description": "XGBoost gradient boosting regressor",
        "framework": "xgboost",
        "model_type": "regression",
    },
    "lgbm_regressor": {
        "class": "LightGBMRegressorTrainer",
        "module": "mlcli.trainers.lightgbm_regressor_trainer",
        "description": "LightGBM gradient boosting regressor",
        "framework": "lightgbm",
        "model_type": "regression",
    },
    # Deep Learning - TensorFlow
    "tf_rnn": {
        "class": "TFRNNTrainer",
        "module": "mlcli.trainers.tf_rnn_trainer",
        "description": "TensorFlow RNN/LSTM/GRU for sequence classification",
        "framework": "tensorflow",
        "model_type": "classification",
    },
    # Deep Learning - PyTorch
    "pytorch_mlp": {
        "class": "PyTorchMLPTrainer",
        "module": "mlcli.trainers.pytorch_mlp_trainer",
        "description": "PyTorch Multi-Layer Perceptron with configurable architecture",
        "framework": "pytorch",
        "model_type": "classification",
    },
    "pytorch_cnn": {
        "class": "PyTorchCNNTrainer",
        "module": "mlcli.trainers.pytorch_cnn_trainer",
        "description": "PyTorch Convolutional Neural Network for image classification",
        "framework": "pytorch",
        "model_type": "classification",
    },
    "pytorch_rnn": {
        "class": "PyTorchRNNTrainer",
        "module": "mlcli.trainers.pytorch_rnn_trainer",
        "description": "PyTorch RNN/LSTM/GRU for sequence classification",
        "framework": "pytorch",
        "model_type": "classification",
    },
    "pytorch_transformer": {
        "class": "PyTorchTransformerTrainer",
        "module": "mlcli.trainers.pytorch_transformer_trainer",
        "description": "PyTorch Transformer encoder for sequence classification",
        "framework": "pytorch",
        "model_type": "classification",
    },
    "tf_cnn": {
        "class": "TFCNNTrainer",
        "module": "mlcli.trainers.tf_cnn_trainer",
        "description": "TensorFlow Convolutional Neural Network for image classification",
        "framework": "tensorflow",
        "model_type": "classification",
    },
    "tf_dnn": {
        "class": "TFDNNTrainer",
        "module": "mlcli.trainers.tf_dnn_trainer",
        "description": "TensorFlow Deep Neural Network for classification",
        "framework": "tensorflow",
        "model_type": "classification",
    },
    # Clustering
    "kmeans": {
        "class": "KMeansTrainer",
        "module": "mlcli.trainers.clustering.kmeans_trainer",
        "description": "K-Means clustering algorithm",
        "framework": "sklearn",
        "model_type": "clustering",
    },
    "dbscan": {
        "class": "DBSCANTrainer",
        "module": "mlcli.trainers.clustering.dbscan_trainer",
        "description": "DBSCAN density-based clustering algorithm",
        "framework": "sklearn",
        "model_type": "clustering",
    },
    # Anomaly Detection
    "isolation_forest": {
        "class": "IsolationForestTrainer",
        "module": "mlcli.trainers.anomaly.isolation_forest_trainer",
        "description": "Isolation Forest anomaly detection algorithm",
        "framework": "sklearn",
        "model_type": "anomaly_detection",
    },
    "one_class_svm": {
        "class": "OneClassSVMTrainer",
        "module": "mlcli.trainers.anomaly.one_class_svm_trainer",
        "description": "One-Class SVM anomaly detection algorithm",
        "framework": "sklearn",
        "model_type": "anomaly_detection",
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
    # Classification - sklearn
    "LogisticRegressionTrainer",
    "SVMTrainer",
    "RFTrainer",
    # Regression - sklearn
    "RFRegressorTrainer",
    # Gradient Boosting - Classification
    "XGBTrainer",
    "LightGBMTrainer",
    "CatBoostTrainer",
    # Gradient Boosting - Regression
    "XGBRegressorTrainer",
    "LightGBMRegressorTrainer",
    # Deep Learning - TensorFlow
    "TFDNNTrainer",
    "TFCNNTrainer",
    "TFRNNTrainer",
    # Clustering
    "KMeansTrainer",
    "DBSCANTrainer",
    # Anomaly Detection
    "IsolationForestTrainer",
    "OneClassSVMTrainer",
    # Utility functions
    "register_all_models",
    "get_trainer_class",
]
