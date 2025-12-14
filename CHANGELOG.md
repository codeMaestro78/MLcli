# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-12-14

### Added
- **Gradient Boosting Algorithms**
  - LightGBM trainer with classification/regression support
  - CatBoost trainer with automatic categorical feature handling
  - Early stopping, feature importance, and native model exports

- **Clustering Algorithms**
  - K-Means trainer with silhouette, calinski-harabasz, davies-bouldin metrics
  - DBSCAN trainer with automatic noise detection
  - Optimal cluster number finder (elbow method) for K-Means
  - Optimal eps finder for DBSCAN

- **Anomaly Detection Algorithms**
  - Isolation Forest trainer for efficient outlier detection
  - One-Class SVM trainer with RBF kernel support
  - Anomaly scoring and threshold tuning capabilities
  - Support for supervised evaluation with ground truth labels

- **Example Configurations**
  - Added lightgbm.json and catboost.json configs
  - Added clustering/kmeans.json and clustering/dbscan.json
  - Added anomaly/isolation_forest.json and anomaly/one_class_svm.json

### Changed
- Updated trainer registry with 6 new algorithm types
- Extended model type support: classification, regression, clustering, anomaly_detection

---

## [0.2.0] - 2025-12-07

### Added
- **Documentation**
  - Created comprehensive `docs/` folder with getting-started guide, installation instructions
  - Added `examples/` folder with sample configs for all model types (RF, XGB, Logistic, DNN)
  - Added example tuning configurations for hyperparameter optimization
  - Added `CONTRIBUTING.md` with detailed contribution guidelines
  - Added `CODE_OF_CONDUCT.md` for community standards
  - Added `SECURITY.md` for vulnerability reporting

- **Testing Infrastructure**
  - Created `tests/` folder with pytest framework
  - Added test suite for trainers (RF, XGBoost, Logistic Regression)
  - Added test suite for tuners (Grid Search, Random Search)
  - Added shared fixtures and configuration in `conftest.py`

### Changed
- Cleaned up project structure for production deployment
- Updated `.gitignore` to exclude generated artifacts properly
- Improved GridSearchTuner to handle `n_trials` parameter gracefully

### Removed
- Hotel-specific datasets and configurations (moved to generic examples)
- Test experiment artifacts from repository
- Empty `scripts/` directory

### Fixed
- GridSearchTuner now properly filters out unsupported `n_trials` parameter
- Configuration files now use generic `data/your_data.csv` paths

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
