# Training Models

This guide explains how to train machine learning and deep learning models using MLCLI.

## Supported Models

MLCLI supports various model types across different frameworks:

### Machine Learning Models (scikit-learn)
- `logistic_regression` - Logistic Regression Classifier
- `svm` - Support Vector Machine Classifier
- `random_forest` - Random Forest Classifier
- `xgboost` - XGBoost Gradient Boosting Classifier
- `lightgbm` - LightGBM Gradient Boosting Classifier
- `catboost` - CatBoost Gradient Boosting Classifier

### Deep Learning Models (TensorFlow)
- `tf_dnn` - Dense Neural Network
- `tf_cnn` - Convolutional Neural Network for images
- `tf_rnn` - Recurrent Neural Network for sequences

### Clustering Models
- `kmeans` - K-Means Clustering
- `dbscan` - DBSCAN Clustering

### Anomaly Detection
- `one_class_svm` - One-Class SVM for anomaly detection
- `isolation_forest` - Isolation Forest for anomaly detection

## Training Workflow

### 1. Prepare Your Data

Your data should be in CSV format with:
- Features as columns
- Target variable in one column (for supervised learning)
- No missing values (handle them during preprocessing)
- Proper data types

Example data format:
```csv
feature1,feature2,feature3,target
1.2,3.4,5.6,0
2.3,4.5,6.7,1
7.8,9.0,1.2,0
```

### 2. Create Configuration File

Create a JSON configuration file that specifies your model, data, and training parameters.

#### Basic Structure
```json
{
  "model": {
    "type": "model_name",
    "params": {
      // Model-specific parameters
    }
  },
  "dataset": {
    "path": "path/to/your/data.csv",
    "type": "csv",
    "target_column": "target_column_name"
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

#### Example: Random Forest
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

#### Example: TensorFlow DNN
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

### 3. Train the Model

```bash
mlcli train --config your_config.json
```

#### Training Options

| Option | Description | Example |
|--------|-------------|---------|
| `--config`, `-c` | Path to config file | `--config rf_config.json` |
| `--output`, `-o` | Output directory override | `--output models/` |
| `--name`, `-n` | Run name for tracking | `--name experiment_1` |
| `--epochs`, `-e` | Override epochs (DL only) | `--epochs 50` |
| `--batch-size`, `-b` | Override batch size (DL only) | `--batch-size 64` |
| `--verbose/--quiet` | Control output verbosity | `--verbose` |

### 4. Monitor Training

MLCLI provides real-time feedback during training:
- Model type and framework
- Dataset information
- Training progress
- Evaluation metrics
- Model saving status

### 5. View Training Results

After training completes, you'll see:
- Test accuracy and other metrics
- Model save locations
- Experiment tracking information

## Advanced Configuration

### Custom Data Loading

For more control over data loading:
```json
{
  "dataset": {
    "path": "data/train.csv",
    "type": "csv",
    "target_column": "label",
    "features": ["col1", "col2", "col3"],
    "encoding": "utf-8"
  }
}
```

### Model-Specific Parameters

Each model type has its own parameters. Use `mlcli list-models` to see available models and their metadata.

### Output Formats

Supported save formats by framework:

| Framework | Formats |
|-----------|---------|
| scikit-learn | `pickle`, `joblib`, `onnx` |
| TensorFlow | `h5`, `savedmodel`, `onnx` |
| XGBoost | `pickle`, `joblib`, `json`, `onnx` |

## Experiment Tracking

All training runs are automatically tracked with:
- Run ID and timestamp
- Configuration parameters
- Training metrics
- Model artifacts location

View experiments with:
```bash
mlcli list-runs
mlcli show-run <run-id>
```

## Troubleshooting

### Common Issues

1. **"Unknown model type"**
   - Check spelling in config file
   - Use `mlcli list-models` to see available models

2. **Memory errors with large datasets**
   - Reduce batch size for DL models
   - Use data preprocessing to reduce features

3. **Poor performance**
   - Ensure data is properly preprocessed
   - Try hyperparameter tuning
   - Consider different model types

4. **TensorFlow warnings**
   - Install TensorFlow: `pip install tensorflow`
   - For M1/M2 Macs: `pip install tensorflow-macos`

### Getting Help

```bash
# List all available models
mlcli list-models

# Get help with training command
mlcli train --help

# View training logs
tail -f logs/mlcli.log
```