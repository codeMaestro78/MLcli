# Getting Started with MLCLI

This guide will help you get started with MLCLI in just a few minutes.

## Prerequisites

- Python 3.10 or higher
- pip package manager

## Installation

### From PyPI (Recommended)

```bash
pip install mlcli-toolkit
```

### From Source

```bash
git clone https://github.com/codeMaestro78/mlcli.git
cd mlcli
pip install -e .
```

## Verify Installation

```bash
mlcli --version
mlcli --help
```

## Your First Model

### 1. Prepare Your Data

Create a CSV file with your data. MLCLI expects:
- Features in columns
- Target variable in one column
- No missing values (or handle them in preprocessing)

Example `data/sample.csv`:
```csv
feature1,feature2,feature3,target
1.2,3.4,5.6,0
2.3,4.5,6.7,1
...
```

### 2. Create a Configuration File

Create `configs/my_config.json`:

```json
{
  "dataset": {
    "path": "data/sample.csv",
    "type": "csv",
    "target_column": "target"
  },
  "model": {
    "type": "random_forest",
    "params": {
      "n_estimators": 100,
      "max_depth": 10,
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

### 3. Train Your Model

```bash
mlcli train --config configs/my_config.json
```

### 4. Evaluate Your Model

```bash
mlcli evaluate --model models/rf_model.pkl --data data/test.csv --target target
```

### 5. Make Predictions

```bash
mlcli predict --model models/rf_model.pkl --data data/new_data.csv --output predictions.csv
```

## Using the Interactive UI

For a guided experience, use the Terminal UI:

```bash
mlcli ui
```

Navigate using arrow keys and Enter to select options.

## Next Steps

- [Configuration Guide](user-guide/configuration.md)
- [Available Models](user-guide/models.md)
- [Hyperparameter Tuning](user-guide/tuning.md)
- [Model Explainability](user-guide/explainability.md)

## Getting Help

```bash
# General help
mlcli --help

# Command-specific help
mlcli train --help
mlcli tune --help
```
