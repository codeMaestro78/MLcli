# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-12-03

### Added
- **CLI Training Pipeline**
  - Train ML models (Logistic Regression, SVM, Random Forest, XGBoost)
  - Train DL models (TensorFlow DNN, CNN, RNN/LSTM/GRU)
  - Configuration-driven training via JSON/YAML files
  - Parameter overrides from command line

- **Hyperparameter Tuning**
  - Grid Search for exhaustive parameter search
  - Random Search for large parameter spaces
  - Bayesian Optimization via Optuna for intelligent search
  - Cross-validation support
  - Auto-train best model after tuning

- **Model Explainability**
  - SHAP (SHapley Additive exPlanations) for global/local explanations
  - LIME (Local Interpretable Model-agnostic Explanations)
  - Feature importance visualization
  - Instance-level explanations

- **Data Preprocessing Pipeline**
  - Scaling: StandardScaler, MinMaxScaler, RobustScaler
  - Normalization: L1, L2, Max norm
  - Encoding: LabelEncoder, OneHotEncoder, OrdinalEncoder
  - Feature Selection: SelectKBest, RFE, VarianceThreshold
  - Pipeline support for chaining preprocessors

- **Experiment Tracking**
  - Automatic experiment logging
  - Run comparison and filtering
  - Export to CSV
  - Mini-MLflow style tracking

- **Model Export**
  - ML models: Pickle, Joblib, ONNX
  - DL models: SavedModel, H5

- **Interactive Terminal UI (TUI)**
  - Train models with guided interface
  - Evaluate saved models
  - Browse experiment history
  - View registered models

- **Model Registry**
  - Automatic model discovery
  - Pluggable trainer architecture
  - Easy extension for custom models

### Technical Details
- Python 3.8+ support
- Type hints throughout codebase
- Rich CLI output with colors and tables
- Comprehensive error handling
