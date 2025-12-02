<div align="center">

```
███╗   ███╗██╗      ██████╗██╗     ██╗
████╗ ████║██║     ██╔════╝██║     ██║
██╔████╔██║██║     ██║     ██║     ██║
██║╚██╔╝██║██║     ██║     ██║     ██║
██║ ╚═╝ ██║███████╗╚██████╗███████╗██║
╚═╝     ╚═╝╚══════╝ ╚═════╝╚══════╝╚═╝
```

# 🤖 MLCLI - Machine Learning Command Line Interface

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-green?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-red?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.ai)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

**A powerful, modular CLI tool for training, evaluating, and tracking ML/DL models**

[Features](#-features) • [Installation](#️-complete-setup-guide-from-scratch) • [Usage](#-all-cli-commands) • [Configuration](#-configuration-files) • [Contributing](#-contributing)

</div>

---

`mlcli` is a modular, configuration-driven command-line tool for training, evaluating, saving, and tracking both Machine Learning and Deep Learning models. It also includes an **interactive terminal UI** for users who prefer a guided workflow.

---

## 🚀 Features

- **Train ML models:**
  - Logistic Regression
  - SVM
  - Random Forest
  - XGBoost

- **Train Deep Learning models:**
  - TensorFlow DNN
  - CNN models
  - RNN/LSTM/GRU models

- **Unified configuration system** (JSON/YAML)
- **Automatic Model Registry** (plug-and-play trainers)
- **Model saving:**
  - ML → Pickle, Joblib & ONNX
  - DL → SavedModel & H5
- **Built-in experiment tracker** (mini-MLflow with JSON storage)
- **Interactive terminal UI (TUI)**

---

## 📁 Project Structure

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
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── io.py
│   │   ├── metrics.py
│   │   ├── logger.py
│   │   └── registry.py
│   ├── runner/
│   │   ├── __init__.py
│   │   └── experiment_tracker.py
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── app.py
│   │   ├── screens/
│   │   └── widgets/
│   └── models/
├── configs/
├── data/
├── artifacts/
├── logs/
├── runs/
├── scripts/
├── README.md
├── pyproject.toml
└── requirements.txt
```

---

## 🛠️ Complete Setup Guide (From Scratch)

### Step 1: Clone the Repository

```bash
git clone https://github.com/codeMaestro78/MLcli.git
cd mlcli
```

### Step 2: Create Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
```

**Linux/macOS:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Install mlcli in Development Mode

```bash
pip install -e .
```

### Step 5: Verify Installation

```bash
mlcli --help
```

**Expected Output:**
```
Usage: mlcli [OPTIONS] COMMAND [ARGS]...

  MLCLI - Machine Learning Command Line Interface

Options:
  --help  Show this message and exit.

Commands:
  eval         Evaluate a saved model on test data.
  export-runs  Export experiment runs to CSV.
  list-models  List all available model trainers.
  list-runs    List all experiment runs.
  show-run     Show details of a specific experiment run.
  train        Train a model using a configuration file.
  ui           Launch the interactive terminal UI.
```

---

## 📖 All CLI Commands

### 1. List Available Models

View all registered model trainers:

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

---

### 2. Train Models

#### Train with Configuration File

```bash
mlcli train --config <path-to-config.json>
```

#### Train Logistic Regression

```bash
mlcli train --config configs/logistic_config.json
```

#### Train Random Forest

```bash
mlcli train --config configs/rf_config.json
```

#### Train SVM

```bash
mlcli train --config configs/svm_config.json
```

#### Train XGBoost

```bash
mlcli train --config configs/xgb_config.json
```

#### Train TensorFlow DNN

```bash
mlcli train --config configs/tf_dnn_config.json
```

#### Train TensorFlow CNN (for image data)

```bash
mlcli train --config configs/tf_cnn_config.json
```

#### Train TensorFlow RNN (for sequence data)

```bash
mlcli train --config configs/tf_rnn_config.json
```

#### Train with Parameter Overrides

```bash
mlcli train --config configs/tf_dnn_config.json --epochs 50 --batch-size 64
```

---

### 3. Evaluate Models

Evaluate a saved model on test data:

```bash
mlcli eval --model-path <path-to-model> --data-path <path-to-test-data> --model-type <model-type>
```

#### Evaluate Pickle Model

```bash
mlcli eval --model-path artifacts/model.pkl --data-path data/test.csv --model-type logistic_regression
```

#### Evaluate Joblib Model

```bash
mlcli eval --model-path artifacts/model.joblib --data-path data/test.csv --model-type random_forest
```

#### Evaluate TensorFlow Model (H5)

```bash
mlcli eval --model-path artifacts/model.h5 --data-path data/test.csv --model-type tf_dnn
```

---

### 4. Experiment Tracking Commands

#### List All Experiment Runs

```bash
mlcli list-runs
```

**Output:**
```
Experiment Runs:
================================================================================
Run ID                              Model Type           Accuracy    Duration
--------------------------------------------------------------------------------
abc123-def456-789...                random_forest        0.8318      4.2s
xyz789-abc123-456...                xgboost              0.8288      1.1s
...
================================================================================
```

#### Show Details of a Specific Run

```bash
mlcli show-run <run-id>
```

**Example:**
```bash
mlcli show-run abc123-def456-789
```

#### Export All Runs to CSV

```bash
mlcli export-runs --output experiments.csv
```

---

### 5. Interactive Terminal UI

Launch the interactive interface:

```bash
mlcli ui
```

**TUI Features:**
- 🎯 **Train Model** - Select config, model type, and override parameters
- 📊 **Evaluate Model** - Load and evaluate saved models
- 📈 **View Experiments** - Browse, filter, and export experiment history
- 🔧 **List Models** - View all registered trainers with metadata

---

## 📝 Configuration Files

### Create a Configuration File

Configuration files define the model, dataset, training parameters, and output settings.

### Configuration Structure

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

### Example Configurations

#### Logistic Regression (`configs/logistic_config.json`)

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

#### Random Forest (`configs/rf_config.json`)

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

#### XGBoost (`configs/xgb_config.json`)

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

#### SVM (`configs/svm_config.json`)

```json
{
  "model": {
    "type": "svm",
    "params": {
      "kernel": "rbf",
      "C": 1.0,
      "gamma": "scale",
      "probability": true
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

#### TensorFlow DNN (`configs/tf_dnn_config.json`)

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

## 🏨 Real-World Example: Hotel Booking Cancellation Prediction

### Step 1: Prepare Your Data

Place your CSV file in the `data/` directory:
```
data/hotel_bookings.csv
```

### Step 2: Preprocess Data (if needed)

Create a preprocessing script `scripts/preprocess_data.py`:

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv('data/hotel_bookings.csv')

# Handle missing values
df = df.fillna(0)

# Encode categorical columns
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    if col != 'target_column':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

# Save processed data
df.to_csv('data/hotel_bookings_processed.csv', index=False)
print("Preprocessing complete!")
```

Run preprocessing:
```bash
python scripts/preprocess_data.py
```

### Step 3: Create Configuration Files

Create `configs/hotel_rf_config.json`:
```json
{
  "model": {
    "type": "random_forest",
    "params": {
      "n_estimators": 100,
      "max_depth": null,
      "random_state": 42
    }
  },
  "dataset": {
    "path": "data/hotel_bookings_processed.csv",
    "type": "csv",
    "target_column": "is_canceled"
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

### Step 4: Train the Model

```bash
mlcli train --config configs/hotel_rf_config.json
```

### Step 5: View Results

```bash
mlcli list-runs
```

### Step 6: Train Multiple Models for Comparison

```bash
# Train Logistic Regression
mlcli train --config configs/hotel_logistic_config.json

# Train Random Forest
mlcli train --config configs/hotel_rf_config.json

# Train XGBoost
mlcli train --config configs/hotel_xgb_config.json

# Train TensorFlow DNN
mlcli train --config configs/hotel_dnn_config.json
```

### Step 7: Export Results

```bash
mlcli export-runs --output hotel_experiments.csv
```

---

## 📊 Model Comparison Results (Hotel Booking Dataset)

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Training Time |
|-------|----------|-----------|--------|----------|---------|---------------|
| **Random Forest** 🏆 | **83.18%** | 83.80% | 83.18% | 82.51% | **90.90%** | 4.2s |
| XGBoost | 82.88% | 83.31% | 82.88% | 82.27% | 90.45% | 1.1s |
| Logistic Regression | 79.90% | 81.03% | 79.90% | 78.68% | 85.20% | 2.8s |
| TF DNN | 62.43% | 38.97% | 62.43% | 47.99% | 50.00% | 43.1s |

> **Note:** Neural networks require feature standardization for optimal performance.

---

## 🧩 Extending mlcli

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

## 🔧 Troubleshooting

### Common Issues

#### 1. "mlcli: command not found"

**Solution:** Make sure the virtual environment is activated and mlcli is installed:
```bash
.\.venv\Scripts\Activate.ps1
pip install -e .
```

#### 2. "ModuleNotFoundError: No module named 'mlcli'"

**Solution:** Install in development mode:
```bash
pip install -e .
```

#### 3. "FileNotFoundError: data/train.csv"

**Solution:** Ensure your data file exists at the specified path in the config file.

#### 4. TensorFlow DNN Poor Performance

**Solution:** Neural networks need standardized features. Add StandardScaler preprocessing:
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

#### 5. ONNX Export Errors

**Solution:** Install skl2onnx:
```bash
pip install skl2onnx
```

---

## 📚 Quick Reference

| Task | Command |
|------|---------|
| Install mlcli | `pip install -e .` |
| Show help | `mlcli --help` |
| List models | `mlcli list-models` |
| Train model | `mlcli train --config <config.json>` |
| Evaluate model | `mlcli eval --model-path <path> --data-path <path> --model-type <type>` |
| List runs | `mlcli list-runs` |
| Show run details | `mlcli show-run <run-id>` |
| Export runs | `mlcli export-runs --output <file.csv>` |
| Launch UI | `mlcli ui` |

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is licensed under the MIT License.
