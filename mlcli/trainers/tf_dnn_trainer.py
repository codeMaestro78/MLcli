"""
TensorFlow Dense Neural Network Trainer

Keras-based trainer for fully-connected deep neural networks.
"""

import numpy as np
from pathlib import Path
from typing import Dict,Any,Optional,List
import logging

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,models,callbacks

from mlcli.trainers.base_trainer import BaseTrainer
from mlcli.utils.registry import register_model
from mlcli.utils.metrics import compute_metrics

logger = logging.getLogger(__name__)

@register_model(name="tf_dnn",description="Tensorflow Dense Feedforward Neural Network",
                framework="tensorflow",
                model_type="classification")

class TFDNNTrainer(BaseTrainer):
    """
    Trainer for TensorFlow/Keras Dense Neural Networks.

    Supports dynamic layer construction, dropout regularization,
    batch normalization, and various optimizers.
    """

    def __init__(self,config:Optional[Dict[str,Any]]=None):
        """
        Initialize TensorFlow DNN trainer.

        Args:
            config: Configuration dictionary with model parameters
        """
        super().__init__(config)

        params = self.config.get('params',{})
        default_params=self.get_default_params()
        self.model_params ={**default_params,**params}

        # Architecture configuration
        self.layers_config= self.model_params.get('layes',[128,64,32])
        self.activation = self.model_params.get('activation','relu')
        self.dropout = self.model_params.get('dropout',0.2)
        self.use_batch_norm = self.model_params.get('batch_normalization',False)

        # Training configuration
        self.optimizer = self.model_params.get('optimizer','adam')
        self.learning_rate= self.model_params.get('learning_rate',0.001)
        self.loss=self.model_params.get('loss','sparse_categorical_crossentropy')
        self.epochs = self.model_params.get('epochs',20)

