# mlcli

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
  - ML → Pickle & ONNX
  - DL → SavedModel & H5
- **Built-in experiment tracker**
- **Interactive terminal UI:**
  ```bash
  mlcli ui
  ```

---

## 📁 Project Structure

```
mlcli/
├── mlcli/
│   ├── __init__.py
│   ├── cli.py
│   ├── ui/
│   │   └── interactive_ui.py
│   ├── config/
│   │   └── loader.py
│   ├── trainers/
│   │   ├── base_trainer.py
│   │   ├── logistic_trainer.py
│   │   ├── svm_trainer.py
│   │   ├── rf_trainer.py
│   │   ├── xgb_trainer.py
│   │   ├── tf_dnn_trainer.py
│   │   ├── tf_cnn_trainer.py
│   │   └── tf_rnn_trainer.py
│   ├── utils/
│   │   ├── io.py
│   │   ├── metrics.py
│   │   ├── logger.py
│   │   └── registry.py
│   ├── runner/
│   │   └── experiment_tracker.py
│   └── models/
├── configs/
│   ├── sample_sklearn_config.json
│   ├── sample_tf_dnn_config.json
│   ├── sample_tf_cnn_config.json
│   └── sample_tf_rnn_config.json
├── README.md
├── pyproject.toml
└── requirements.txt
```

---

## 🛠️ Installation

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

---

## 📖 Usage

### Command-Line Interface

Train a model using a configuration file:

```bash
mlcli train --config configs/sample_sklearn_config.json
```

### Interactive UI

Launch the interactive terminal interface:

```bash
mlcli ui
```

---

## 📝 Configuration

Configuration files can be written in JSON or YAML format. See the `configs/` directory for examples.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License
This project is licensed under the MIT License.
