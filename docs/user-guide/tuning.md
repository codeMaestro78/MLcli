# Hyperparameter Tuning

This guide explains how to optimize model hyperparameters using MLCLI's tuning capabilities.

## Tuning Overview

Hyperparameter tuning is the process of finding the best combination of model parameters to maximize performance. MLCLI supports multiple tuning methods with automatic cross-validation.

## Supported Tuning Methods

| Method | Description | Best For | Speed |
|--------|-------------|----------|-------|
| **Grid Search** | Exhaustive search over all parameter combinations | Small parameter spaces | Slow |
| **Random Search** | Random sampling from parameter distributions | Large parameter spaces | Medium |
| **Bayesian Optimization** | Intelligent search using probabilistic models | Expensive evaluations | Fast |

## Basic Tuning Workflow

### 1. Create Tuning Configuration

Create a JSON config file that extends your training config with tuning parameters:

```json
{
  "model": {
    "type": "random_forest",
    "params": {}
  },
  "dataset": {
    "path": "data/train.csv",
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
      "min_samples_leaf": [1, 2, 4]
    }
  },
  "output": {
    "model_dir": "artifacts"
  }
}
```

### 2. Run Hyperparameter Tuning

```bash
mlcli tune --config tune_rf_config.json --method random --n-trials 100
```

### 3. Review Results

MLCLI displays the best parameters and performance metrics, then optionally trains the final model.

## Configuration Details

### Parameter Space Definition

Define parameter spaces differently for each method:

#### Grid Search (Discrete Values)
```json
{
  "tuning": {
    "param_space": {
      "n_estimators": [50, 100, 200],
      "max_depth": [10, 20, null],
      "criterion": ["gini", "entropy"]
    }
  }
}
```

#### Random/Bayesian Search (Distributions)
```json
{
  "tuning": {
    "param_space": {
      "n_estimators": {"type": "int", "low": 50, "high": 500},
      "max_depth": {"type": "int", "low": 3, "high": 20},
      "learning_rate": {"type": "loguniform", "low": 0.01, "high": 0.3},
      "subsample": {"type": "uniform", "low": 0.6, "high": 1.0}
    }
  }
}
```

### Parameter Distribution Types

| Type | Description | Example |
|------|-------------|---------|
| `int` | Integer range | `{"type": "int", "low": 1, "high": 100}` |
| `uniform` | Uniform float | `{"type": "uniform", "low": 0.0, "high": 1.0}` |
| `loguniform` | Log-uniform | `{"type": "loguniform", "low": 0.001, "high": 1.0}` |
| `categorical` | Choice from list | `{"type": "categorical", "choices": ["a", "b", "c"]}` |

## Tuning Commands

### Basic Syntax

```bash
mlcli tune --config <config.json> --method <method> [options]
```

### Key Options

| Option | Description | Default |
|--------|-------------|---------|
| `--config`, `-c` | Tuning configuration file | Required |
| `--method`, `-m` | Tuning method (grid/random/bayesian) | `random` |
| `--n-trials`, `-n` | Number of trials | `50` |
| `--cv` | Cross-validation folds | `5` |
| `--scoring` | Metric to optimize | `accuracy` |
| `--train-best` | Train final model with best params | `False` |
| `--output`, `-o` | Save tuning results | Auto-generated |

### Available Scoring Metrics

- **Classification**: `accuracy`, `precision`, `recall`, `f1`, `roc_auc`
- **Regression**: `neg_mean_squared_error`, `r2` (when supported)
- **Custom**: Framework-specific metrics

## Examples

### Random Forest Tuning

```json
// tune_rf_config.json
{
  "model": {
    "type": "random_forest",
    "params": {}
  },
  "dataset": {
    "path": "data/train.csv",
    "target_column": "target"
  },
  "training": {
    "test_size": 0.2,
    "random_state": 42
  },
  "tuning": {
    "param_space": {
      "n_estimators": {"type": "int", "low": 50, "high": 300},
      "max_depth": {"type": "int", "low": 5, "high": 30},
      "min_samples_split": {"type": "int", "low": 2, "high": 20},
      "max_features": {"type": "categorical", "choices": ["sqrt", "log2", null]}
    }
  }
}
```

```bash
mlcli tune --config tune_rf_config.json --method random --n-trials 100 --cv 5
```

### XGBoost Tuning

```json
// tune_xgb_config.json
{
  "model": {
    "type": "xgboost",
    "params": {}
  },
  "dataset": {
    "path": "data/train.csv",
    "target_column": "target"
  },
  "tuning": {
    "param_space": {
      "n_estimators": {"type": "int", "low": 50, "high": 500},
      "max_depth": {"type": "int", "low": 3, "high": 15},
      "learning_rate": {"type": "loguniform", "low": 0.01, "high": 0.3},
      "subsample": {"type": "uniform", "low": 0.6, "high": 1.0},
      "colsample_bytree": {"type": "uniform", "low": 0.6, "high": 1.0}
    }
  }
}
```

```bash
mlcli tune --config tune_xgb_config.json --method bayesian --n-trials 200 --scoring f1
```

### Neural Network Tuning

```json
// tune_tf_dnn_config.json
{
  "model": {
    "type": "tf_dnn",
    "params": {
      "optimizer": "adam",
      "epochs": 20
    }
  },
  "dataset": {
    "path": "data/train.csv",
    "target_column": "target"
  },
  "tuning": {
    "param_space": {
      "layers": {"type": "categorical", "choices": [[64, 32], [128, 64], [256, 128, 64]]},
      "learning_rate": {"type": "loguniform", "low": 0.0001, "high": 0.01},
      "dropout": {"type": "uniform", "low": 0.1, "high": 0.5},
      "batch_size": {"type": "categorical", "choices": [16, 32, 64]}
    }
  }
}
```

## Tuning Results

### Best Parameters Display
```
Best Hyperparameters
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃ Parameter           ┃ Value            ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ n_estimators        │ 247              │
│ max_depth           │ 12               │
│ min_samples_split   │ 5                │
│ max_features        │ sqrt             │
└─────────────────────┴──────────────────┘
```

### Performance Summary
```
Tuning Complete!
Best Score (accuracy): 0.9247
Total Trials: 100
Duration: 45.2s
```

### Top Parameter Combinations
Shows the top 5 parameter combinations with their scores.

## Advanced Features

### Saving Tuning Results

Results are automatically saved to `runs/tuning_<model>_<method>.json`

```bash
# Custom output location
mlcli tune --config config.json --method random --output my_tuning_results.json
```

### Training Best Model

```bash
mlcli tune --config config.json --method bayesian --train-best
```

This will:
1. Run tuning to find best parameters
2. Train a final model with those parameters
3. Save the model to the configured output directory

### Cross-Validation Control

```bash
# Use 10-fold CV
mlcli tune --config config.json --method grid --cv 10

# Use custom scoring
mlcli tune --config config.json --method random --scoring f1
```

## Best Practices

### Parameter Space Design

1. **Start small** - Begin with a small parameter space
2. **Use domain knowledge** - Choose reasonable ranges based on your understanding
3. **Include default values** - Always include default parameters as options
4. **Consider correlations** - Some parameters interact (e.g., max_depth and n_estimators)

### Method Selection

- **Grid Search**: For small, discrete parameter spaces
- **Random Search**: For large parameter spaces with continuous values
- **Bayesian**: For expensive evaluations or when you want optimal efficiency

### Computational Considerations

- **Grid Search**: Can be very slow with many parameters
- **Random Search**: Scales better with parameter count
- **Bayesian**: Most efficient but requires installation of Optuna

## Troubleshooting

### Common Issues

1. **"No param_space defined"**
   - Add a `tuning.param_space` section to your config

2. **Slow tuning**
   - Reduce number of trials
   - Use random search instead of grid
   - Simplify parameter space

3. **Memory errors**
   - Reduce CV folds
   - Use smaller datasets for tuning
   - Choose simpler models

4. **Optuna not found**
   ```bash
   pip install optuna
   ```

### Performance Tips

- Use early stopping for deep learning models
- Start with random search before bayesian optimization
- Use cross-validation scores for more reliable estimates
- Parallelize tuning when possible

## Integration with Experiment Tracking

All tuning runs are tracked:

```bash
# View tuning experiments
mlcli list-runs --model xgboost

# Export tuning results
mlcli export-runs --output tuning_experiments.csv
```

## Getting Help

```bash
# List available tuning methods
mlcli list-tuners

# Get help with tune command
mlcli tune --help

# View tuning examples
mlcli tune --help
```