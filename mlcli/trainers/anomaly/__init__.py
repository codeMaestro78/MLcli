"""
Anomaly detection trainers module.

Provides unsupervised anomaly/outlier detection algorithms.
"""

from mlcli.trainers.anomaly.isolation_forest_trainer import IsolationForestTrainer
from mlcli.trainers.anomaly.one_class_svm_trainer import OneClassSVMTrainer

__all__ = [
    "IsolationForestTrainer",
    "OneClassSVMTrainer",
]
