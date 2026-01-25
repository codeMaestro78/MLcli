# Data Preprocessing

This guide covers data preprocessing capabilities in MLCLI, including scaling, normalization, encoding, and feature selection.

## Preprocessing Overview

Data preprocessing transforms raw data into a format suitable for machine learning models. MLCLI provides a comprehensive set of preprocessing methods that can be applied individually or as part of a pipeline.

## Supported Preprocessing Methods

### Scaling Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `standard_scaler` | Standardize to zero mean, unit variance | Most ML algorithms |
| `minmax_scaler` | Scale to range (default 0-1) | Neural networks, distance-based algorithms |
| `robust_scaler` | Scale using median/IQR (outlier-resistant) | Data with outliers |
| `maxabs_scaler` | Scale by maximum absolute value | Sparse data |

### Normalization Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `normalizer` | Normalize samples to unit norm | Text data, sparse matrices |
| `l1_normalizer` | L1 norm normalization | Sparse data |
| `l2_normalizer` | L2 norm normalization | Dense data |

### Encoding Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `label_encoder` | Encode labels to 0 to n_classes-1 | Target variables, ordinal features |
| `onehot_encoder` | One-hot encode categorical features | Nominal categorical features |
| `ordinal_encoder` | Ordinal encode categorical features | Ordinal categorical features |

### Feature Selection Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `select_k_best` | Select top K features by score | High-dimensional data |
| `rfe` | Recursive Feature Elimination | Remove least important features |
| `variance_threshold` | Remove low-variance features | Remove constant/near-constant features |

## Basic Preprocessing

### Single Method Preprocessing

```bash
mlcli preprocess --data data/train.csv --output data/train_scaled.csv --method standard_scaler
```

### Required Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--data`, `-d` | Input CSV file path | `data/train.csv` |
| `--output`, `-o` | Output CSV file path | `data/train_scaled.csv` |
| `--method`, `-m` | Preprocessing method | `standard_scaler` |

### Optional Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--target`, `-t` | Target column name | Auto-detect |
| `--columns`, `-c` | Specific columns to preprocess | All numeric |
| `--k` | Number of features (SelectKBest/RFE) | `10` |
| `--threshold` | Variance threshold | `0.0` |
| `--norm` | Norm type (l1, l2, max) | `l2` |
| `--range-min` | MinMaxScaler min value | `0.0` |
| `--range-max` | MinMaxScaler max value | `1.0` |
| `--save-preprocessor` | Save fitted preprocessor | None |

## Examples

### Standard Scaling

```bash
mlcli preprocess --data data/train.csv --output data/train_scaled.csv --method standard_scaler
```

This transforms features to have zero mean and unit variance.

### Min-Max Scaling

```bash
mlcli preprocess -d data/train.csv -o data/train_minmax.csv -m minmax_scaler --range-min 0 --range-max 1
```

Scales features to a specified range.

### Robust Scaling (Outlier-Resistant)

```bash
mlcli preprocess -d data/train.csv -o data/train_robust.csv -m robust_scaler
```

Uses median and IQR instead of mean and standard deviation.

### Feature Selection with SelectKBest

```bash
mlcli preprocess -d data/train.csv -o data/train_selected.csv -m select_k_best --target label --k 10
```

Selects top 10 features based on statistical tests.

### Recursive Feature Elimination

```bash
mlcli preprocess -d data/train.csv -o data/train_rfe.csv -m rfe --target label --k 15
```

Recursively removes features and builds a model on remaining features.

### Remove Low-Variance Features

```bash
mlcli preprocess -d data/train.csv -o data/train_var.csv -m variance_threshold --threshold 0.1
```

Removes features with variance below the threshold.

## Preprocessing Pipelines

### Multiple Steps in Sequence

```bash
mlcli preprocess-pipeline --data data/train.csv --output data/processed.csv --steps "standard_scaler,select_k_best" --target label
```

### Pipeline Configuration

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--data`, `-d` | Input data file | `data/train.csv` |
| `--output`, `-o` | Output file | `data/processed.csv` |
| `--steps`, `-s` | Comma-separated preprocessing steps | `"standard_scaler,rfe"` |
| `--config`, `-c` | Pipeline config file (JSON) | `pipeline.json` |
| `--target`, `-t` | Target column | `label` |
| `--save-pipeline` | Save fitted pipeline | `models/pipeline.pkl` |

### Advanced Pipeline Example

```bash
# Multiple preprocessing steps
mlcli preprocess-pipeline \
  --data data/train.csv \
  --output data/processed.csv \
  --steps "standard_scaler,variance_threshold,select_k_best" \
  --target target \
  --save-pipeline models/preprocessing_pipeline.pkl
```

## Configuration Files

### Preprocessing Config

```json
{
  "preprocessing": {
    "steps": [
      {
        "method": "standard_scaler",
        "params": {}
      },
      {
        "method": "select_k_best",
        "params": {
          "k": 20
        }
      }
    ]
  }
}
```

## Saving and Loading Preprocessors

### Save Preprocessor

```bash
mlcli preprocess -d data/train.csv -o data/train_scaled.csv -m standard_scaler --save-preprocessor models/scaler.pkl
```

### Load in Python

```python
import joblib

# Load saved preprocessor
scaler = joblib.load('models/scaler.pkl')

# Apply to new data
X_new_scaled = scaler.transform(X_new)
```

## Data Requirements

### Input Format
- CSV files with headers
- Numeric columns for scaling/normalization
- Categorical columns for encoding
- No missing values (handle separately)

### Column Selection
- By default, processes all numeric columns
- Use `--columns` to specify specific columns
- Target column is automatically excluded

### Missing Values
Handle missing values before preprocessing:

```bash
# Fill missing values
python -c "import pandas as pd; df = pd.read_csv('data/train.csv'); df.fillna(df.mean(), inplace=True); df.to_csv('data/train_clean.csv', index=False)"
```

## Integration with Training

### Preprocess Before Training

```bash
# Preprocess data
mlcli preprocess -d data/train.csv -o data/train_scaled.csv -m standard_scaler

# Train model on preprocessed data
mlcli train --config config.json  # config points to train_scaled.csv
```

### Preprocessing in Config

While MLCLI doesn't have built-in preprocessing in training configs, you can:

1. Preprocess data separately
2. Update your training config to use preprocessed data
3. Save preprocessors for later use on test/production data

## Best Practices

### When to Use Each Method

- **StandardScaler**: Most common choice for ML algorithms
- **MinMaxScaler**: When you need bounded values (0-1)
- **RobustScaler**: When data has many outliers
- **Normalizer**: For text data or when you want unit norm samples

### Feature Selection Guidelines

- **SelectKBest**: Good starting point, fast
- **RFE**: More accurate but slower, good for smaller datasets
- **VarianceThreshold**: Very fast, good for removing obviously useless features

### Pipeline Design

1. **Handle missing values first**
2. **Encode categorical features**
3. **Scale/normalize numeric features**
4. **Apply feature selection**
5. **Save preprocessing pipeline**

### Validation

Always validate preprocessing:

```bash
# Check processed data
head data/train_scaled.csv

# Verify statistics
python -c "import pandas as pd; df = pd.read_csv('data/train_scaled.csv'); print(df.describe())"
```

## Troubleshooting

### Common Issues

1. **"Target column required"**
   - Specify `--target` for feature selection methods

2. **"No numeric columns found"**
   - Check data types: `df.dtypes`
   - Convert strings to numeric if needed

3. **Memory errors**
   - Process data in chunks for large datasets
   - Use more efficient methods

4. **Poor model performance after preprocessing**
   - Verify preprocessing is applied consistently
   - Check for data leakage
   - Validate preprocessing parameters

### Data Type Issues

```bash
# Check data types
python -c "import pandas as pd; print(pd.read_csv('data/train.csv').dtypes)"

# Convert to numeric
python -c "import pandas as pd; df = pd.read_csv('data/train.csv'); df['col'] = pd.to_numeric(df['col']); df.to_csv('data/train_fixed.csv', index=False)"
```

## Advanced Usage

### Custom Preprocessing

For custom preprocessing needs:

```python
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load and preprocess manually
df = pd.read_csv('data/train.csv')
scaler = StandardScaler()
numeric_cols = df.select_dtypes(include=['number']).columns
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
df.to_csv('data/train_custom_scaled.csv', index=False)
```

### Integration with ML Pipelines

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Create ML pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])

# Train pipeline
pipeline.fit(X_train, y_train)
```

## Getting Help

```bash
# List available preprocessing methods
mlcli list-preprocessors

# Get help with preprocess command
mlcli preprocess --help

# Get help with pipeline command
mlcli preprocess-pipeline --help
```