# MLCLI Documentation

Welcome to the MLCLI documentation!

## Quick Links

- [Getting Started](getting-started.md)
- [Installation](installation.md)
- [User Guide](user-guide/index.md)
- [API Reference](api/index.md)
- [Examples](../examples/README.md)
- [Contributing](../CONTRIBUTING.md)

## What is MLCLI?

MLCLI is a powerful, modular command-line interface for training, evaluating, and tracking machine learning and deep learning models. It provides:

- **Unified CLI** for multiple ML frameworks (scikit-learn, TensorFlow, XGBoost)
- **Configuration-driven** training with JSON/YAML files
- **Built-in experiment tracking** with JSON storage
- **Model explainability** with SHAP and LIME
- **Hyperparameter tuning** with Grid, Random, and Bayesian optimization
- **Interactive TUI** for guided workflows

## Installation

```bash
pip install mlcli-toolkit
```

## Quick Start

```bash
# Train a model
mlcli train --config configs/rf_config.json

# Evaluate a model
mlcli evaluate --model models/rf_model.pkl --data data/test.csv

# Launch interactive UI
mlcli ui
```

## Documentation Structure

```
docs/
├── index.md                 # This file
├── getting-started.md       # Quick start guide
├── installation.md          # Installation instructions
├── user-guide/
│   ├── index.md
│   ├── training.md
│   ├── evaluation.md
│   ├── tuning.md
│   ├── preprocessing.md
│   └── explainability.md
├── api/
│   ├── index.md
│   ├── trainers.md
│   ├── tuners.md
│   └── preprocessors.md
└── tutorials/
    ├── index.md
    └── custom-trainer.md
```

## Support

- [GitHub Issues](https://github.com/codeMaestro78/mlcli/issues)
- [Discussions](https://github.com/codeMaestro78/mlcli/discussions)
