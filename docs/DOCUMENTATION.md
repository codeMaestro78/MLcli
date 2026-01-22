# MLCLI Complete Documentation

> **Version:** 0.3.0
> **Last Updated:** December 2025

This document contains comprehensive documentation for MLCLI - Machine Learning Command Line Interface.

---

## Table of Contents

1. [Installation](#installation)
2. [Project Structure](#project-structure)
3. [CLI Commands](#cli-commands)
4. [Configuration Files](#configuration-files)
5. [Hyperparameter Tuning](#hyperparameter-tuning)
6. [Model Explainability](#model-explainability)
7. [Data Preprocessing](#data-preprocessing)
8. [Interactive TUI](#interactive-tui)
9. [Experiment Tracking](#experiment-tracking)
10. [Extending MLCLI](#extending-mlcli)
11. [Troubleshooting](#troubleshooting)

---

## Installation

### From PyPI (Recommended)

```bash
pip install mlcli-toolkit
```

### From Source

```bash
git clone https://github.com/codeMaestro78/MLcli.git
cd mlcli
```

**Create Virtual Environment:**

```powershell
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux/macOS
python -m venv .venv
source .venv/bin/activate
```

**Install Dependencies:**

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

**Verify Installation:**

```bash
mlcli --help
```

---

## Project Structure

```
mlcli/
├── mlcli/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── loader.py
│   ├── trainers/
│   │   ├── __init__.py
│   │   ├── base_trainer.py
│   │   ├── logistic_trainer.py
│   │   ├── svm_trainer.py
│   │   ├── rf_trainer.py
│   │   ├── xgb_trainer.py
│   │   ├── tf_dnn_trainer.py
│   │   ├── tf_cnn_trainer.py
│   │   └── tf_rnn_trainer.py
│   ├── tuner/
│   │   ├── __init__.py
│   │   ├── base_tuner.py
│   │   ├── grid_tuner.py
│   │   ├── random_tuner.py
│   │   └── optuna_tuner.py
│   ├── explainer/
│   │   ├── __init__.py
│   │   ├── base_explainer.py
│   │   ├── shap_explainer.py
│   │   ├── lime_explainer.py
│   │   └── explainer_factory.py
│   ├── preprocessor/
│   │   ├── __init__.py
│   │   ├── base_preprocessor.py
│   │   ├── scalers.py
│   │   ├── normalizers.py
│   │   ├── encoders.py
│   │   ├── feature_selectors.py
│   │   ├── preprocessor_factory.py
│   │   └── pipeline.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── io.py
│   │   ├── metrics.py
│   │   ├── logger.py
│   │   └── registry.py
│   ├── runner/
│   │   ├── __init__.py
│   │   └── experiment_tracker.py
│   └── ui/
│       ├── __init__.py
│       └── app.py
├── configs/
├── data/
├── artifacts/
├── logs/
├── runs/
├── docs/
├── README.md
├── pyproject.toml
└── requirements.txt
```

---

## CLI Commands

### List Available Models

```bash
mlcli list-models
```

**Output:**

```
Available Model Trainers:
================================================================================
  logistic_regression    Logistic Regression Classifier           [sklearn]
  svm                    Support Vector Machine Classifier        [sklearn]
  random_forest          Random Forest Classifier                 [sklearn]
  xgboost                XGBoost Gradient Boosting Classifier     [xgboost]
  tf_dnn                 TensorFlow Dense Neural Network          [tensorflow]
  tf_cnn                 TensorFlow CNN for Image Classification  [tensorflow]
  tf_rnn                 TensorFlow RNN for Sequence Data         [tensorflow]
================================================================================
```

### Train Models

```bash
# Train with configuration file
mlcli train --config <path-to-config.json>

# Examples
mlcli train --config configs/logistic_config.json
mlcli train --config configs/rf_config.json
mlcli train --config configs/xgb_config.json
mlcli train --config configs/tf_dnn_config.json

# Train with parameter overrides
mlcli train --config configs/tf_dnn_config.json --epochs 50 --batch-size 64
```

### Evaluate Models

```bash
mlcli eval --model-path <path-to-model> --data-path <path-to-test-data> --model-type <model-type>

# Examples
mlcli eval --model-path artifacts/model.pkl --data-path data/test.csv --model-type logistic_regression
mlcli eval --model-path artifacts/model.joblib --data-path data/test.csv --model-type random_forest
mlcli eval --model-path artifacts/model.h5 --data-path data/test.csv --model-type tf_dnn
```

### Experiment Tracking

```bash
# List all experiment runs
mlcli list-runs

# Show details of a specific run
mlcli show-run <run-id>

# Export all runs to CSV
mlcli export-runs --output experiments.csv
```

### Launch Interactive UI

```bash
mlcli ui
```

---

## Configuration Files

### Basic Structure

```json
{
  "model": {
    "type": "<model-type>",
    "params": { ... }
  },
  "dataset": {
    "path": "<path-to-data>",
    "type": "csv",
    "target_column": "<target-column-name>"
  },
  "training": {
    "test_size": 0.2,
    "random_state": 42
  },
  "output": {
    "model_dir": "artifacts",
    "save_formats": ["pickle", "joblib"]
  }
}
```

### Example: Logistic Regression

```json
{
  "model": {
    "type": "logistic_regression",
    "params": {
      "penalty": "l2",
      "C": 1.0,
      "solver": "lbfgs",
      "max_iter": 1000
    }
  },
  "dataset": {
    "path": "data/train.csv",
    "type": "csv",
    "target_column": "target"
  },
  "training": {
    "test_size": 0.2,
    "random_state": 42
  },
  "output": {
    "model_dir": "artifacts",
    "save_formats": ["pickle", "joblib"]
  }
}
```

### Example: Random Forest

```json
{
  "model": {
    "type": "random_forest",
    "params": {
      "n_estimators": 100,
      "max_depth": null,
      "min_samples_split": 2,
      "min_samples_leaf": 1,
      "random_state": 42
    }
  },
  "dataset": {
    "path": "data/train.csv",
    "type": "csv",
    "target_column": "target"
  },
  "training": {
    "test_size": 0.2,
    "random_state": 42
  },
  "output": {
    "model_dir": "artifacts",
    "save_formats": ["pickle", "joblib"]
  }
}
```

### Example: XGBoost

```json
{
  "model": {
    "type": "xgboost",
    "params": {
      "n_estimators": 100,
      "max_depth": 6,
      "learning_rate": 0.1,
      "subsample": 0.8,
      "colsample_bytree": 0.8,
      "early_stopping_rounds": 10,
      "random_state": 42
    }
  },
  "dataset": {
    "path": "data/train.csv",
    "type": "csv",
    "target_column": "target"
  },
  "training": {
    "test_size": 0.2,
    "random_state": 42
  },
  "output": {
    "model_dir": "artifacts",
    "save_formats": ["pickle", "joblib"]
  }
}
```

### Example: TensorFlow DNN

```json
{
  "model": {
    "type": "tf_dnn",
    "params": {
      "layers": [128, 64, 32],
      "activation": "relu",
      "dropout": 0.3,
      "optimizer": "adam",
      "learning_rate": 0.001,
      "epochs": 20,
      "batch_size": 32,
      "early_stopping": true,
      "patience": 5
    }
  },
  "dataset": {
    "path": "data/train.csv",
    "type": "csv",
    "target_column": "target"
  },
  "training": {
    "test_size": 0.2,
    "random_state": 42
  },
  "output": {
    "model_dir": "artifacts",
    "save_formats": ["h5", "savedmodel"]
  }
}
```

---

## Hyperparameter Tuning

### Available Methods

| Method     | Name                           | Best For                                    |
| ---------- | ------------------------------ | ------------------------------------------- |
| `grid`     | Grid Search                    | Small parameter spaces with discrete values |
| `random`   | Random Search                  | Large parameter spaces, continuous params   |
| `bayesian` | Bayesian Optimization (Optuna) | Expensive evaluations, complex param spaces |

### Commands

```bash
# List available tuning methods
mlcli list-tuners

# Tune with Grid Search
mlcli tune --config configs/tune_rf_config.json --method grid --cv 5

# Tune with Random Search
mlcli tune --config configs/tune_rf_config.json --method random --n-trials 100 --cv 5

# Tune with Bayesian Optimization
mlcli tune --config configs/tune_xgb_config.json --method bayesian --n-trials 200 --scoring accuracy

# Tune and train best model
mlcli tune --config configs/tune_rf_config.json --method random --n-trials 50 --train-best
```

### Tuning Options

| Option             | Description                                                            |
| ------------------ | ---------------------------------------------------------------------- |
| `--config`, `-c`   | Path to tuning configuration file                                      |
| `--method`, `-m`   | Tuning method: `grid`, `random`, or `bayesian`                         |
| `--n-trials`, `-n` | Number of trials (for random/bayesian)                                 |
| `--cv`             | Number of cross-validation folds                                       |
| `--scoring`, `-s`  | Metric to optimize: `accuracy`, `f1`, `roc_auc`, `precision`, `recall` |
| `--output`, `-o`   | Path to save tuning results (JSON)                                     |
| `--train-best`     | Train a model with best params after tuning                            |

### Grid Search Configuration

```json
{
  "model": {
    "type": "random_forest",
    "params": {}
  },
  "dataset": {
    "path": "data/train.csv",
    "type": "csv",
    "target_column": "target"
  },
  "training": {
    "test_size": 0.2,
    "random_state": 42
  },
  "tuning": {
    "param_space": {
      "n_estimators": [50, 100, 200, 300],
      "max_depth": [5, 10, 15, 20, null],
      "min_samples_split": [2, 5, 10],
      "min_samples_leaf": [1, 2, 4],
      "max_features": ["sqrt", "log2"]
    }
  },
  "output": {
    "model_dir": "artifacts",
    "save_formats": ["pickle", "joblib"]
  }
}
```

### Random/Bayesian Search Configuration

```json
{
  "model": {
    "type": "xgboost",
    "params": {}
  },
  "dataset": {
    "path": "data/train.csv",
    "type": "csv",
    "target_column": "target"
  },
  "training": {
    "test_size": 0.2,
    "random_state": 42
  },
  "tuning": {
    "param_space": {
      "n_estimators": { "type": "int", "low": 50, "high": 500 },
      "max_depth": { "type": "int", "low": 3, "high": 15 },
      "learning_rate": { "type": "loguniform", "low": 0.01, "high": 0.3 },
      "subsample": { "type": "uniform", "low": 0.6, "high": 1.0 },
      "colsample_bytree": { "type": "uniform", "low": 0.6, "high": 1.0 },
      "min_child_weight": { "type": "int", "low": 1, "high": 10 }
    }
  },
  "output": {
    "model_dir": "artifacts",
    "save_formats": ["pickle", "joblib"]
  }
}
```

### Parameter Distribution Types

| Type          | Description      | Example                                             |
| ------------- | ---------------- | --------------------------------------------------- |
| `list/tuple`  | Discrete choices | `[50, 100, 200]`                                    |
| `int`         | Integer range    | `{"type": "int", "low": 1, "high": 100}`            |
| `uniform`     | Uniform float    | `{"type": "uniform", "low": 0.0, "high": 1.0}`      |
| `loguniform`  | Log-uniform      | `{"type": "loguniform", "low": 0.001, "high": 1.0}` |
| `categorical` | Choice           | `{"type": "categorical", "choices": ["a", "b"]}`    |

---

## Model Explainability

### Available Methods

| Method | Full Name                                       | Best For                               |
| ------ | ----------------------------------------------- | -------------------------------------- |
| `shap` | SHapley Additive exPlanations                   | Tree-based models, global explanations |
| `lime` | Local Interpretable Model-agnostic Explanations | Any model, local explanations          |

### Commands

```bash
# List available explainers
mlcli list-explainers

# Explain model with SHAP
mlcli explain --model models/rf_model.pkl --data data/train.csv --type random_forest --method shap

# Explain model with LIME
mlcli explain --model models/xgb_model.pkl --data data/train.csv --type xgboost --method lime

# Explain with plot output
mlcli explain -m models/rf_model.pkl -d data/train.csv -t random_forest -e shap --plot-output feature_importance.png

# Explain single instance
mlcli explain-instance --model models/rf_model.pkl --data data/test.csv --type random_forest --instance 0
mlcli explain-instance -m models/xgb_model.pkl -d data/test.csv -t xgboost -i 5 -e lime
```

### Explainability Options

| Option                | Description                                              |
| --------------------- | -------------------------------------------------------- |
| `--model`, `-m`       | Path to saved model file                                 |
| `--data`, `-d`        | Path to data file                                        |
| `--type`, `-t`        | Model type (random_forest, xgboost, logistic_regression) |
| `--method`, `-e`      | Explanation method: `shap` or `lime`                     |
| `--num-samples`, `-n` | Number of samples to explain (default: 100)              |
| `--output`, `-o`      | Path to save explanation results (JSON)                  |
| `--plot/--no-plot`    | Generate explanation plot                                |
| `--plot-output`, `-p` | Path to save plot (PNG)                                  |

### SHAP vs LIME Comparison

| Feature         | SHAP                         | LIME                      |
| --------------- | ---------------------------- | ------------------------- |
| **Type**        | Global + Local               | Local                     |
| **Theory**      | Game Theory (Shapley Values) | Local Surrogate Models    |
| **Best For**    | Tree models (RF, XGBoost)    | Any black-box model       |
| **Speed**       | Fast for trees               | Slower (samples required) |
| **Consistency** | Mathematically consistent    | Varies by sampling        |

---

## Data Preprocessing

### Available Preprocessors

| Category              | Method               | Description                                |
| --------------------- | -------------------- | ------------------------------------------ |
| **Scaling**           | `standard_scaler`    | Standardize to zero mean, unit variance    |
|                       | `minmax_scaler`      | Scale to range (default 0-1)               |
|                       | `robust_scaler`      | Scale using median/IQR (outlier-resistant) |
| **Normalization**     | `normalizer`         | Normalize samples to unit norm             |
|                       | `l1_normalizer`      | L1 norm normalization                      |
|                       | `l2_normalizer`      | L2 norm normalization                      |
| **Encoding**          | `label_encoder`      | Encode labels to 0 to n_classes-1          |
|                       | `onehot_encoder`     | One-hot encode categorical features        |
|                       | `ordinal_encoder`    | Ordinal encode categorical features        |
| **Feature Selection** | `select_k_best`      | Select top K features                      |
|                       | `rfe`                | Recursive Feature Elimination              |
|                       | `variance_threshold` | Remove low-variance features               |

### Commands

```bash
# List available preprocessors
mlcli list-preprocessors

# StandardScaler
mlcli preprocess --data data/train.csv --output data/train_scaled.csv --method standard_scaler

# MinMaxScaler
mlcli preprocess -d data/train.csv -o data/train_minmax.csv -m minmax_scaler --range-min 0 --range-max 1

# RobustScaler (outlier-resistant)
mlcli preprocess -d data/train.csv -o data/train_robust.csv -m robust_scaler

# Normalize Data (L2 norm)
mlcli preprocess -d data/train.csv -o data/train_norm.csv -m normalizer --norm l2

# Feature Selection with SelectKBest
mlcli preprocess -d data/train.csv -o data/train_selected.csv -m select_k_best --target label --k 10

# Feature Selection with RFE
mlcli preprocess -d data/train.csv -o data/train_rfe.csv -m rfe --target label --k 15

# Remove Low-Variance Features
mlcli preprocess -d data/train.csv -o data/train_var.csv -m variance_threshold --threshold 0.1

# Save Fitted Preprocessor
mlcli preprocess -d data/train.csv -o data/train_scaled.csv -m standard_scaler --save-preprocessor models/scaler.pkl

# Preprocessing Pipeline (Multiple Steps)
mlcli preprocess-pipeline --data data/train.csv --output data/processed.csv --steps "standard_scaler,select_k_best" --target label
```

### Preprocessing Options

| Option                       | Description                           |
| ---------------------------- | ------------------------------------- |
| `--data`, `-d`               | Path to input CSV data                |
| `--output`, `-o`             | Path to save preprocessed data        |
| `--method`, `-m`             | Preprocessing method                  |
| `--target`, `-t`             | Target column (for feature selection) |
| `--columns`, `-c`            | Specific columns to preprocess        |
| `--k`                        | Number of features (SelectKBest/RFE)  |
| `--threshold`                | Variance threshold                    |
| `--norm`                     | Norm type (l1, l2, max)               |
| `--range-min`, `--range-max` | MinMaxScaler range                    |
| `--save-preprocessor`, `-s`  | Save fitted preprocessor              |

---

## Interactive TUI

Launch the interactive terminal UI:

```bash
mlcli ui
```

### TUI Features

- **Train Model** - Select config, model type, and override parameters
- **Evaluate Model** - Load and evaluate saved models
- **View Experiments** - Browse, filter, and export experiment history
- **List Models** - View all registered trainers with metadata

### Keyboard Shortcuts

| Key     | Action              |
| ------- | ------------------- |
| `h`     | Go to Home screen   |
| `q`     | Quit application    |
| `Enter` | Select/Confirm      |
| `↑↓`    | Navigate lists      |
| `Tab`   | Move between fields |

---

## Experiment Tracking

MLCLI includes a built-in experiment tracker that logs all training runs.

### Logged Information

- Run ID (UUID)
- Model type
- Configuration parameters
- Training metrics (accuracy, precision, recall, F1, etc.)
- Training duration
- Timestamp

### View Experiments

```bash
# List all runs
mlcli list-runs

# Show specific run details
mlcli show-run <run-id>

# Export to CSV
mlcli export-runs --output experiments.csv
```

### Runs Directory

All experiment data is stored in `runs/` directory as JSON files.

---

## Extending MLCLI

### Adding a New Trainer

1. Create a new file in `mlcli/trainers/`:

```python
from mlcli.trainers.base_trainer import BaseTrainer
from mlcli.utils.registry import register_model

@register_model(
    name="my_custom_model",
    description="My Custom Model Trainer",
    framework="custom",
    model_type="classification"
)
class MyCustomTrainer(BaseTrainer):
    def train(self, X_train, y_train, X_val=None, y_val=None):
        # Implementation
        pass

    def evaluate(self, X_test, y_test):
        # Implementation
        pass

    def predict(self, X):
        # Implementation
        pass

    @classmethod
    def get_default_params(cls):
        return {"param1": "value1"}
```

2. Import in `mlcli/trainers/__init__.py`:

```python
from mlcli.trainers.my_custom_trainer import MyCustomTrainer
```

The model will be automatically registered and available via CLI!

---

## Troubleshooting

### "mlcli: command not found"

Make sure the virtual environment is activated and mlcli is installed:

```bash
.\.venv\Scripts\Activate.ps1
pip install -e .
```

### "ModuleNotFoundError: No module named 'mlcli'"

Install in development mode:

```bash
pip install -e .
```

### "FileNotFoundError: data/train.csv"

Ensure your data file exists at the specified path in the config file.

### TensorFlow DNN Poor Performance

Neural networks need standardized features. Use preprocessing:

```bash
mlcli preprocess -d data/train.csv -o data/train_scaled.csv -m standard_scaler
```

### ONNX Export Errors

Install skl2onnx:

```bash
pip install skl2onnx
```

### Optuna Not Found

Install optuna for Bayesian optimization:

```bash
pip install optuna
```

### SHAP/LIME Not Found

Install SHAP and LIME:

```bash
pip install shap lime matplotlib
```

---

## Quick Reference

| Task                 | Command                                                                  |
| -------------------- | ------------------------------------------------------------------------ |
| Install              | `pip install mlcli-toolkit`                                              |
| Show help            | `mlcli --help`                                                           |
| List models          | `mlcli list-models`                                                      |
| List tuners          | `mlcli list-tuners`                                                      |
| List explainers      | `mlcli list-explainers`                                                  |
| List preprocessors   | `mlcli list-preprocessors`                                               |
| Train model          | `mlcli train --config <config.json>`                                     |
| Tune hyperparameters | `mlcli tune -c <config> -m random -n 100`                                |
| Explain model (SHAP) | `mlcli explain -m <model.pkl> -d <data.csv> -t <type> -e shap`           |
| Explain instance     | `mlcli explain-instance -m <model.pkl> -d <data.csv> -t <type> -i <idx>` |
| Preprocess data      | `mlcli preprocess -d <data.csv> -o <output.csv> -m standard_scaler`      |
| Evaluate model       | `mlcli eval --model-path <path> --data-path <path> --model-type <type>`  |
| List runs            | `mlcli list-runs`                                                        |
| Export runs          | `mlcli export-runs --output <file.csv>`                                  |
| Launch UI            | `mlcli ui`                                                               |

---

## License

This project is licensed under the MIT License.
