<div align="center">

```
███╗   ███╗██╗      ██████╗██╗     ██╗
████╗ ████║██║     ██╔════╝██║     ██║
██╔████╔██║██║     ██║     ██║     ██║
██║╚██╔╝██║██║     ██║     ██║     ██║
██║ ╚═╝ ██║███████╗╚██████╗███████╗██║
╚═╝     ╚═╝╚══════╝ ╚═════╝╚══════╝╚═╝
```

# MLCLI - Machine Learning Command Line Interface

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyPI](https://img.shields.io/badge/PyPI-mlcli--toolkit-blue?style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/mlcli-toolkit/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

**A powerful, modular CLI tool for training, evaluating, and tracking ML/DL models**

[📖 Documentation](https://mlcli.vercel.app/) • [📦 PyPI](https://pypi.org/project/mlcli-toolkit/) • [📚 Full Docs](docs/DOCUMENTATION.md)

</div>

---

## ✨ Features

- **ML Models:** Logistic Regression, SVM, Random Forest, XGBoost
- **DL Models:** TensorFlow DNN, CNN, RNN/LSTM/GRU
- **Hyperparameter Tuning:** Grid Search, Random Search, Bayesian (Optuna)
- **Model Explainability:** SHAP & LIME
- **Preprocessing:** Scalers, Normalizers, Encoders, Feature Selection
- **Experiment Tracking:** Built-in tracker with JSON storage
- **Interactive TUI:** Terminal-based user interface

---

## 🚀 Quick Start

### Install

```bash
pip install mlcli-toolkit
```

### Verify

```bash
mlcli --help
```

### Train a Model

```bash
mlcli train --config configs/rf_config.json
```

### Launch Interactive UI

```bash
mlcli ui
```

---

## 📋 Commands

| Command | Description |
|---------|-------------|
| `mlcli list-models` | List available model trainers |
| `mlcli train -c <config>` | Train a model |
| `mlcli eval -m <model> -d <data> -t <type>` | Evaluate a model |
| `mlcli tune -c <config> -m <method>` | Hyperparameter tuning |
| `mlcli explain -m <model> -d <data> -e <method>` | Model explainability |
| `mlcli preprocess -d <data> -o <output> -m <method>` | Preprocess data |
| `mlcli list-runs` | List experiment runs |
| `mlcli ui` | Launch interactive TUI |

---

## 📝 Configuration Example

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

---

## 📚 Documentation

For complete documentation including:
- All configuration options
- Hyperparameter tuning guides
- Model explainability (SHAP/LIME)
- Data preprocessing pipeline
- Extending MLCLI with custom trainers
- Troubleshooting

See **[docs/DOCUMENTATION.md](docs/DOCUMENTATION.md)**

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is licensed under the MIT License.
