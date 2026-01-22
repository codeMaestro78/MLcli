# PROJECT SUMMARY: mlcli-toolkit v0.3.0

## 🎯 Project Overview

**mlcli-toolkit** is a production-ready CLI toolkit for training, evaluating, and tracking Machine Learning and Deep Learning models. It provides a unified interface for experiment tracking, hyperparameter tuning, model explainability, and an interactive TUI.

## 📊 Current Architecture

### Tech Stack

| Component           | Technology                                            |
| ------------------- | ----------------------------------------------------- |
| Language            | Python 3.10+                                          |
| CLI Framework       | Typer + Rich                                          |
| ML Frameworks       | scikit-learn, TensorFlow, XGBoost, LightGBM, CatBoost |
| Serialization       | Pickle, Joblib, ONNX, SavedModel (TF), H5 (TF)        |
| Config              | JSON/YAML via ConfigLoader                            |
| Experiment Tracking | Custom JSON-based (mini-MLflow)                       |
| UI                  | Textual TUI                                           |

### Core Modules

```
mlcli/
├── cli.py              # Main CLI (1602 lines) - train, tune, eval commands
├── trainers/           # 15+ model trainers
│   ├── base_trainer.py    # Abstract BaseTrainer class
│   ├── logistic_trainer.py
│   ├── svm_trainer.py
│   ├── rf_trainer.py      # Random Forest
│   ├── xgb_trainer.py     # XGBoost
│   ├── lightgbm_trainer.py
│   ├── catboost_trainer.py
│   ├── tf_dnn_trainer.py  # TensorFlow DNN
│   ├── tf_cnn_trainer.py  # TensorFlow CNN
│   ├── tf_rnn_trainer.py  # TensorFlow RNN
│   ├── clustering/        # KMeans, DBSCAN
│   └── anomaly/           # IsolationForest, OneClassSVM
├── tuner/              # Hyperparameter tuning
│   ├── base_tuner.py      # Abstract BaseTuner class
│   ├── grid_tuner.py      # GridSearchCV
│   ├── random_tuner.py    # RandomizedSearchCV
│   ├── optuna_tuner.py    # Bayesian (TPE)
│   └── tuner_factory.py   # Factory pattern
├── preprocessor/       # Data preprocessing
│   ├── base_preprocessor.py
│   ├── pipeline.py        # PreprocessingPipeline
│   ├── scalers.py         # StandardScaler, MinMax, etc.
│   ├── encoders.py        # Label, OneHot encoding
│   └── feature_selectors.py # SelectKBest, RFE, VarianceThreshold
├── runner/             # Experiment tracking
│   └── experiment_tracker.py  # JSON-based run tracking
├── explainer/          # Model explainability
│   ├── shap_explainer.py
│   └── lime_explainer.py
├── config/             # Configuration management
│   └── loader.py          # ConfigLoader for JSON/YAML
├── ui/                 # Interactive TUI
│   └── tui.py
└── utils/              # Utilities
    ├── registry.py        # Model auto-registration
    ├── metrics.py         # Compute metrics
    └── io.py              # Data loading
```

## 🔧 Current ML Workflow

### 1. Training Flow

```
Config (JSON/YAML)
    ↓
ConfigLoader.load()
    ↓
Registry.get_trainer(model_type)
    ↓
Trainer.train(X_train, y_train, X_val, y_val)
    ↓
ExperimentTracker.log_metrics()
    ↓
Trainer.save() → [pickle, joblib, onnx, h5, savedmodel]
```

### 2. Tuning Flow

```
Config with param_space
    ↓
TunerFactory.create(method, param_space)
    ↓
Tuner.tune(trainer_class, X, y)
    ↓
[Grid|Random|Bayesian] Search
    ↓
Best params + Optional train_best model
```

### 3. Existing CLI Commands

- `mlcli train --config <file>` - Train a model
- `mlcli tune --config <file> --method <grid|random|bayesian>` - Hyperparameter tuning
- `mlcli eval --model <path> --data <path>` - Evaluate saved model
- `mlcli list-models` - List available models
- `mlcli list-runs` - Show experiment history
- `mlcli ui` - Launch interactive TUI

## 🏗️ Design Patterns Used

1. **Abstract Base Class Pattern**

   - `BaseTrainer` → All trainers inherit
   - `BaseTuner` → All tuners inherit
   - `BasePreprocessor` → All preprocessors inherit

2. **Registry Pattern** (`@register_model` decorator)

   - Auto-registration via decorators
   - Lazy loading for heavy dependencies (TensorFlow)
   - Metadata storage (framework, model_type, description)

3. **Factory Pattern**

   - `TunerFactory.create(method, param_space)` → Tuner instance
   - `PreprocessorFactory` for preprocessors

4. **Pipeline Pattern**
   - `PreprocessingPipeline.add_step().fit_transform()`

## 📈 Key Observations for AutoML Integration

### Strengths (Leverage These)

1. ✅ Abstract base classes allow easy extension
2. ✅ Registry pattern enables dynamic model discovery
3. ✅ Existing tuner infrastructure (Grid/Random/Bayesian)
4. ✅ Experiment tracker ready for AutoML logging
5. ✅ Preprocessing pipeline supports chaining
6. ✅ ONNX export for model portability

### Gaps to Fill for AutoML

1. ❌ No automatic model selection
2. ❌ No automatic preprocessing selection
3. ❌ No feature engineering automation
4. ❌ No ensemble/stacking support
5. ❌ No data type inference
6. ❌ No time budget management
7. ❌ No early stopping for model search
8. ❌ No cross-model comparison reporting

## 📦 Dependencies (from pyproject.toml)

### Core

- numpy>=1.24,<2.0
- pandas>=2.0
- scikit-learn>=1.0

### ML Frameworks

- tensorflow>=2.10
- xgboost>=1.7
- lightgbm>=4.0.0
- catboost>=1.2.7

### Tuning

- optuna>=3.0.0 (implicit via optuna_tuner)

### CLI/UI

- typer[all]>=0.7.0
- rich-click>=1.6.0
- textual>=0.40.0

### Serialization

- onnx>=1.14
- onnxruntime>=1.15
- skl2onnx>=1.14
- joblib>=1.1

---

_Document generated for AutoML Integration Planning_
_Date: Phase 0 - Project Understanding_
