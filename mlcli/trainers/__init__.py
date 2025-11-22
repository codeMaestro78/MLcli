"""Model trainers module."""

from mlcli.trainers.base_trainer import  BaseTrainer

# Import all trainers to trigger auto-registration
from mlcli.trainers.logistic_trainer import LogisticRegressionTrainer
from mlcli.trainers.svm_trainer import SVMTrainer
from mlcli.trainers.rf_trainer import RFTrainer
from mlcli.trainers.xgb_trainer import XGBTrainer
from mlcli.trainers.tf_dnn_trainer import TFDNNTrainer
from mlcli.trainers.tf_cnn_trainer import TFCNNTrainer
from mlcli.trainers.tf_rnn_trainer import TFRNNTrainer

__all__ = [
    "BaseTrainer",
    "LogisticRegressionTrainer",
    "SVMTrainer",
    "RFTrainer",
    "XGBTrainer",
    "TFDNNTrainer",
    "TFCNNTrainer",
    "TFRNNTrainer",
]
