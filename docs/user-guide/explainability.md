# Model Explainability

This guide covers model explainability features in MLCLI, including SHAP and LIME methods for understanding model predictions.

## Explainability Overview

Model explainability helps understand why a model makes certain predictions. MLCLI supports two popular methods:

- **SHAP (SHapley Additive exPlanations)**: Game theory-based approach providing global and local explanations
- **LIME (Local Interpretable Model-agnostic Explanations)**: Local surrogate models for individual predictions

## Supported Methods

| Method | Full Name | Type | Best For |
|--------|-----------|------|----------|
| `shap` | SHapley Additive exPlanations | Global + Local | Tree models, feature importance |
| `lime` | Local Interpretable Model-agnostic Explanations | Local | Any model, individual predictions |

## Basic Explainability

### Global Feature Importance (SHAP)

```bash
mlcli explain --model models/rf_model.pkl --data data/train.csv --type random_forest --method shap
```

### Local Explanation (LIME)

```bash
mlcli explain --model models/rf_model.pkl --data data/train.csv --type random_forest --method lime
```

### Single Instance Explanation

```bash
mlcli explain-instance --model models/rf_model.pkl --data data/test.csv --type random_forest --instance 0
```

## Command Parameters

### Common Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--model`, `-m` | Path to saved model | `models/rf_model.pkl` |
| `--data`, `-d` | Path to data file | `data/train.csv` |
| `--type`, `-t` | Model type | `random_forest` |
| `--method`, `-e` | Explanation method | `shap` or `lime` |
| `--target` | Target column name | `label` |

### Additional Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--num-samples`, `-n` | Number of samples to explain | `100` |
| `--output`, `-o` | Save explanation results (JSON) | Auto-generated |
| `--plot/--no-plot` | Generate plots | `True` |
| `--plot-output`, `-p` | Plot output path | Auto-generated |
| `--instance`, `-i` | Instance index (for explain-instance) | `0` |

## Examples

### SHAP on Random Forest

```bash
mlcli explain --model models/rf_model.pkl --data data/train.csv --type random_forest --method shap --plot-output shap_plot.png
```

### LIME on XGBoost

```bash
mlcli explain -m models/xgb_model.pkl -d data/train.csv -t xgboost -e lime -n 50
```

### Explain Specific Instance

```bash
mlcli explain-instance --model models/rf_model.pkl --data data/test.csv --type random_forest --instance 5 --method shap
```

## Explanation Output

### Feature Importance Table

```
Feature Importance (SHAP)
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Rank          ┃ Feature   ┃ Importance   ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ 1             │ age       │ 0.234        │
│ 2             │ income    │ 0.189        │
│ 3             │ education │ 0.145        │
│ 4             │ experience│ 0.098        │
└───────────────┴───────────┴──────────────┘
```

### Instance-Level Explanation

```
Feature Contributions
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ Feature       ┃ Contribution  ┃ Direction      ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ age           │ 0.156         │ ↑ Positive     │
│ income        │ -0.089        │ ↓ Negative     │
│ education     │ 0.067         │ ↑ Positive     │
└───────────────┴───────────────┴────────────────┘
```

## SHAP vs LIME

### SHAP (Recommended for Tree Models)

**Advantages:**
- Mathematically sound (based on game theory)
- Provides global feature importance
- Consistent and accurate
- Works well with tree-based models

**Best for:**
- Random Forest, XGBoost, LightGBM
- When you need feature importance rankings
- When model interpretability is critical

### LIME (Recommended for Complex Models)

**Advantages:**
- Model-agnostic (works with any model)
- Provides local explanations
- Easy to understand
- Works with black-box models

**Best for:**
- Neural networks, SVMs
- When you need per-instance explanations
- When SHAP is too slow or complex

## Visualization

### SHAP Plots

MLCLI generates various SHAP plots:
- **Bar Plot**: Feature importance ranking
- **Bee Swarm Plot**: Distribution of feature impacts
- **Waterfall Plot**: Individual prediction breakdown

### LIME Plots

- **Feature Contribution Plot**: Shows how each feature contributes to the prediction
- **Local Surrogate Plot**: Shows the simple model that approximates the complex model locally

## Model Compatibility

### Supported Model Types

| Model Type | SHAP Support | LIME Support |
|------------|--------------|--------------|
| random_forest | ✅ Excellent | ✅ Good |
| xgboost | ✅ Excellent | ✅ Good |
| lightgbm | ✅ Excellent | ✅ Good |
| logistic_regression | ⚠️ Limited | ✅ Good |
| svm | ⚠️ Limited | ✅ Good |
| tf_dnn | ⚠️ Limited | ✅ Good |
| tf_cnn | ❌ Not supported | ⚠️ Limited |

### Framework-Specific Notes

- **TensorFlow models**: SHAP support is limited; LIME works but may be slow
- **Large datasets**: Use smaller sample sizes for faster explanations
- **High-dimensional data**: Consider feature selection before explainability

## Advanced Usage

### Customizing Explanations

```bash
# Explain subset of features
mlcli explain --model model.pkl --data data.csv --type random_forest --method shap --num-samples 500

# Save detailed results
mlcli explain --model model.pkl --data data.csv --type random_forest --method shap --output detailed_explanation.json
```

### Batch Instance Explanations

```bash
# Explain multiple instances
for i in {0..9}; do
  mlcli explain-instance --model model.pkl --data test.csv --type random_forest --instance $i --output explanation_$i.json
done
```

### Integration with Training

```bash
# Train model
mlcli train --config rf_config.json

# Explain the trained model
mlcli explain --model artifacts/rf_model.pkl --data data/train.csv --type random_forest --method shap
```

## Interpretation Guidelines

### Understanding SHAP Values

- **Positive values**: Feature pushes prediction higher
- **Negative values**: Feature pushes prediction lower
- **Magnitude**: Strength of feature's influence
- **Distribution**: How feature impacts vary across dataset

### Understanding LIME

- **Local surrogate**: Simple model approximating complex model for one instance
- **Feature weights**: How much each feature contributes to the prediction
- **Sparsity**: Only most important features are shown

## Best Practices

### When to Use Explainability

1. **Model validation**: Ensure model uses reasonable features
2. **Bias detection**: Identify if model relies on sensitive features
3. **Feature engineering**: Guide feature selection and creation
4. **Regulatory compliance**: Explain predictions for legal requirements
5. **Debugging**: Understand unexpected model behavior

### Choosing Sample Size

- **Small datasets (< 1000 samples)**: Use all data
- **Medium datasets (1000-10000)**: Use 100-500 samples
- **Large datasets (>10000)**: Use 100-1000 samples, or sample strategically

### Computational Considerations

- **SHAP**: Faster on tree models, slower on neural networks
- **LIME**: Consistent speed across model types
- **Large datasets**: Use smaller sample sizes or sampling
- **High dimensions**: Consider dimensionality reduction first

## Troubleshooting

### Common Issues

1. **"SHAP not compatible with model"**
   - Use LIME instead for complex models
   - Check model type compatibility

2. **Slow explanations**
   - Reduce `--num-samples`
   - Use faster explanation method
   - Sample data strategically

3. **Memory errors**
   - Use smaller datasets
   - Reduce sample size
   - Use LIME instead of SHAP

4. **Poor explanations**
   - Ensure data preprocessing consistency
   - Check for data quality issues
   - Validate model performance first

### Installation Issues

```bash
# Install SHAP
pip install shap

# Install LIME
pip install lime

# Install matplotlib for plots
pip install matplotlib
```

## Integration with Experiment Tracking

Explanations are automatically tracked:

```bash
# View explanation experiments
mlcli list-runs --model random_forest

# Export explanation results
mlcli export-runs --output explanation_results.csv
```

## Advanced Features

### Custom Explainability

For advanced use cases:

```python
import shap
from mlcli.utils.io import load_data
from mlcli import registry

# Load model and data
trainer = registry.get_trainer('random_forest', config={})
trainer.load('model.pkl')
X, y = load_data('data.csv', target_column='target')

# Create custom explainer
explainer = shap.TreeExplainer(trainer.model)
shap_values = explainer.shap_values(X[:100])

# Custom visualization
shap.summary_plot(shap_values, X[:100])
```

## Getting Help

```bash
# List available explanation methods
mlcli list-explainers

# Get help with explain command
mlcli explain --help

# Get help with explain-instance command
mlcli explain-instance --help

# View example usage
mlcli explain --help
```