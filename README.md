# mlcli

`mlcli` is a modular, configuration-driven command-line tool for training, evaluating, saving, and tracking both Machine Learning and Deep Learning models.
It also includes an **interactive terminal UI** for users who prefer a guided workflow.

---

## 🚀 Features

- Train ML models:
  - Logistic Regression
  - SVM
  - Random Forest
  - XGBoost
- Train Deep Learning models:
  - TensorFlow DNN
  - CNN models
  - RNN/LSTM/GRU models
- Unified configuration system (JSON/YAML)
- Automatic Model Registry (plug-and-play trainers)
- Model saving:
  - ML → Pickle & ONNX
  - DL → SavedModel & H5
- Built-in experiment tracker
- Interactive terminal UI:
  ```bash
  mlcli ui

mlcli/
│── mlcli/
│    ├── cli.py
│    ├── ui/
│    ├── config/
│    ├── trainers/
│    ├── utils/
│    ├── runner/
│    ├── models/
│
│── configs/
│── README.md
│── pyproject.toml
│── requirements.txt
