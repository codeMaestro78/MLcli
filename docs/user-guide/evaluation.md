# Model Evaluation

This guide covers how to evaluate trained models using MLCLI.

## Evaluation Overview

Model evaluation involves testing your trained model on unseen data to assess its performance. MLCLI supports various evaluation metrics and provides detailed reports.

## Supported Evaluation Types

MLCLI can evaluate models for:
- **Classification**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Regression**: MSE, MAE, R² Score (when supported)
- **Clustering**: Silhouette Score, Calinski-Harabasz Index (for clustering models)
- **Anomaly Detection**: Precision, Recall, F1-Score for anomaly detection

## Basic Evaluation

### Command Syntax

```bash
mlcli eval --model <model-path> --data <test-data> --type <model-type>
```

### Required Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--model`, `-m` | Path to saved model file | `models/rf_model.pkl` |
| `--data`, `-d` | Path to evaluation data (CSV) | `data/test.csv` |
| `--type`, `-t` | Model type | `random_forest` |

### Optional Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--format`, `-f` | Model format | `pickle` |
| `--target` | Target column name | Auto-detect |
| `--verbose` | Detailed output | `True` |

## Examples

### Evaluate Random Forest Model

```bash
mlcli eval --model artifacts/rf_model.pkl --data data/test.csv --type random_forest
```

### Evaluate TensorFlow Model

```bash
mlcli eval --model artifacts/dnn_model.h5 --data data/test.csv --type tf_dnn --format h5
```

### Evaluate XGBoost Model with Custom Target

```bash
mlcli eval -m models/xgb_model.pkl -d test.csv -t xgboost --target label
```

## Evaluation Output

MLCLI provides comprehensive evaluation results:

### Metrics Table
```
Evaluation Results
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Metric              ┃ Value         ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ Accuracy            │ 0.9125        │
│ Precision           │ 0.8952        │
│ Recall              │ 0.9341        │
│ F1 Score            │ 0.9142        │
│ ROC AUC             │ 0.9578        │
└─────────────────────┴───────────────┘
```

### Classification Report (Detailed)
For classification tasks, MLCLI also shows per-class metrics:
- Precision, Recall, F1-Score for each class
- Support (number of samples per class)
- Macro and weighted averages

## Model Formats

MLCLI supports loading models from various formats:

| Framework | Supported Formats |
|-----------|-------------------|
| scikit-learn | `pickle`, `joblib`, `onnx` |
| TensorFlow | `h5`, `savedmodel`, `onnx` |
| XGBoost | `pickle`, `json`, `onnx` |

## Data Requirements

### Input Data Format
- CSV format with headers
- Same features as training data
- Target column (optional for predictions, required for metrics)

### Missing Values
- Handle missing values before evaluation
- Use preprocessing if needed

## Advanced Evaluation

### Custom Metrics

For custom evaluation scenarios, you can:
1. Load the model manually using MLCLI's Python API
2. Use your own evaluation functions
3. Integrate with other evaluation libraries

### Cross-Validation Evaluation

For more robust evaluation, consider:
1. Using hyperparameter tuning with cross-validation
2. Evaluating on multiple test sets
3. Using ensemble evaluation methods

## Troubleshooting

### Common Issues

1. **"Model file not found"**
   - Check the model path
   - Ensure the model was saved correctly during training

2. **"Unknown model type"**
   - Use `mlcli list-models` to see available types
   - Check spelling in the type parameter

3. **"Feature mismatch"**
   - Ensure test data has same features as training data
   - Check for missing or extra columns

4. **Poor metrics**
   - Verify data preprocessing is consistent
   - Check for data leakage
   - Consider model overfitting

### Data Format Issues

If your data has issues:
```bash
# Check data format
head data/test.csv

# Validate CSV structure
python -c "import pandas as pd; print(pd.read_csv('data/test.csv').info())"
```

## Integration with Training

Evaluation is typically done after training:

```bash
# Train model
mlcli train --config rf_config.json

# Evaluate on test data
mlcli eval --model artifacts/rf_model.pkl --data data/test.csv --type random_forest

# Check training vs test performance
mlcli show-run <run-id>
```

## Best Practices

1. **Use separate test data** - Never evaluate on training data
2. **Maintain data consistency** - Same preprocessing as training
3. **Multiple metrics** - Don't rely on single metrics
4. **Cross-validation** - Use when possible for more reliable estimates
5. **Document results** - Keep track of evaluation results and parameters

## Getting Help

```bash
# Get help with eval command
mlcli eval --help

# List available models for type parameter
mlcli list-models

# View experiment history
mlcli list-runs
```