# MLCLI Examples

This directory contains example configurations and scripts to help you get started with MLCLI.

## Directory Structure

```
examples/
├── README.md                    # This file
├── configs/                     # Example configuration files
│   ├── random_forest.json
│   ├── xgboost.json
│   ├── logistic_regression.json
│   ├── svm.json
│   ├── tensorflow_dnn.json
│   └── tuning/
│       ├── tune_rf.json
│       └── tune_xgb.json
├── data/                        # Sample datasets
│   └── README.md
└── notebooks/                   # Jupyter notebooks
    └── getting_started.ipynb
```

## Quick Start Examples

### 1. Train a Random Forest Model

```bash
mlcli train --config examples/configs/random_forest.json
```

### 2. Hyperparameter Tuning

```bash
mlcli tune --config examples/configs/tuning/tune_rf.json --method random --n-trials 20
```

### 3. Model Explanation

```bash
mlcli explain --model models/rf_model.pkl --data data/test.csv --method shap
```

### 4. Preprocessing Pipeline

```bash
mlcli preprocess --data data/raw.csv --output data/processed.csv --methods standard_scaler,select_k_best
```

## Sample Configurations

### Random Forest

```json
{
  "dataset": {
    "path": "data/your_data.csv",
    "type": "csv",
    "target_column": "target"
  },
  "model": {
    "type": "random_forest",
    "params": {
      "n_estimators": 100,
      "max_depth": 10,
      "min_samples_split": 2,
      "random_state": 42
    }
  },
  "training": {
    "test_size": 0.2,
    "random_state": 42
  },
  "output": {
    "model_dir": "models",
    "save_format": ["pickle", "onnx"]
  }
}
```

### XGBoost

```json
{
  "dataset": {
    "path": "data/your_data.csv",
    "type": "csv",
    "target_column": "target"
  },
  "model": {
    "type": "xgboost",
    "params": {
      "n_estimators": 100,
      "max_depth": 6,
      "learning_rate": 0.1,
      "random_state": 42
    }
  },
  "training": {
    "test_size": 0.2,
    "random_state": 42
  },
  "output": {
    "model_dir": "models",
    "save_format": ["pickle"]
  }
}
```

### TensorFlow DNN

```json
{
  "dataset": {
    "path": "data/your_data.csv",
    "type": "csv",
    "target_column": "target"
  },
  "model": {
    "type": "tf_dnn",
    "params": {
      "hidden_layers": [128, 64, 32],
      "activation": "relu",
      "dropout_rate": 0.3,
      "learning_rate": 0.001
    }
  },
  "training": {
    "epochs": 100,
    "batch_size": 32,
    "validation_split": 0.2,
    "early_stopping": true,
    "patience": 10
  },
  "output": {
    "model_dir": "models",
    "save_format": ["keras", "h5"]
  }
}
```

## Using Your Own Data

1. Place your CSV file in the `data/` directory
2. Update the `dataset.path` in any config file
3. Set the correct `target_column` name
4. Run training!

## Need Help?

- Check the [documentation](../docs/index.md)
- Open an [issue](https://github.com/codeMaestro78/mlcli/issues)
- Start a [discussion](https://github.com/codeMaestro78/mlcli/discussions)
