# MLCLI User Guide

This user guide provides comprehensive instructions for using MLCLI to train, evaluate, tune, preprocess data, and explain machine learning models.

## Overview

MLCLI is a command-line interface for training and managing machine learning and deep learning models. It supports multiple frameworks including scikit-learn, TensorFlow, XGBoost, and more.

## Sections

- **[Training Models](training.md)** - Learn how to train ML/DL models with configuration files
- **[Model Evaluation](evaluation.md)** - Evaluate trained models on test data
- **[Hyperparameter Tuning](tuning.md)** - Optimize model parameters using various tuning methods
- **[Data Preprocessing](preprocessing.md)** - Prepare and transform your data
- **[Model Explainability](explainability.md)** - Understand model predictions with SHAP and LIME

## Quick Start

```bash
# Install MLCLI
pip install mlcli-toolkit

# Train a model
mlcli train --config configs/rf_config.json

# Evaluate the model
mlcli eval --model models/rf_model.pkl --data test.csv --type random_forest

# Launch interactive UI
mlcli ui
```

## Configuration Basics

All MLCLI operations use JSON configuration files. A basic config includes:

```json
{
  "model": {
    "type": "model_name",
    "params": { ... }
  },
  "dataset": {
    "path": "data/train.csv",
    "target_column": "target"
  },
  "training": {
    "test_size": 0.2,
    "random_state": 42
  },
  "output": {
    "model_dir": "artifacts"
  }
}
```

## Getting Help

```bash
# General help
mlcli --help

# Command-specific help
mlcli train --help
mlcli tune --help
```

For more detailed information, see the [API Reference](../api/index.md) or [Tutorials](../tutorials/index.md).