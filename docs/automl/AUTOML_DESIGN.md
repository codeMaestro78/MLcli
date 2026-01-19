# AutoML Design Document

## 🎯 Scope Definition

### AutoML Mode: **Model-Centric AutoML**

Based on the existing mlcli architecture, I recommend a **Model-Centric AutoML** approach that:

1. Leverages existing trainers, tuners, and preprocessors
2. Adds intelligent model selection and comparison
3. Maintains backward compatibility
4. Follows existing design patterns

### In Scope

| Feature                         | Priority | Rationale                               |
| ------------------------------- | -------- | --------------------------------------- |
| Automatic Model Selection       | P0       | Core AutoML value proposition           |
| Automatic Hyperparameter Tuning | P0       | Leverage existing tuner infrastructure  |
| Automatic Preprocessing         | P1       | Leverage existing preprocessor pipeline |
| Multi-Model Comparison          | P1       | Essential for AutoML reporting          |
| Time Budget Management          | P1       | Production requirement                  |
| Ensemble Creation               | P2       | Advanced AutoML feature                 |
| Feature Engineering             | P2       | Adds ML value                           |
| Early Stopping                  | P2       | Efficiency optimization                 |

### Out of Scope (v1)

- Neural Architecture Search (NAS)
- AutoML for Deep Learning model architecture
- Distributed/parallel AutoML
- AutoML for computer vision/NLP-specific tasks
- Custom loss function optimization

## 🏗️ Architecture Strategy

### Option Analysis

| Strategy                           | Pros                      | Cons                           | Recommendation  |
| ---------------------------------- | ------------------------- | ------------------------------ | --------------- |
| **A) New AutoML Module**           | Clean separation, focused | Code duplication               | ✅ **Selected** |
| B) Extend Tuners                   | Reuse existing code       | Overloads tuner responsibility | ❌              |
| C) External Library (auto-sklearn) | Battle-tested             | Dependency bloat, less control | ❌              |

### Chosen Architecture: New `mlcli/automl/` Module

```
mlcli/
├── automl/                      # NEW MODULE
│   ├── __init__.py
│   ├── base_automl.py           # BaseAutoML abstract class
│   ├── automl_classifier.py     # AutoML for classification
│   ├── automl_regressor.py      # AutoML for regression (future)
│   ├── model_selector.py        # Model selection logic
│   ├── search_space.py          # Default param spaces per model
│   ├── data_analyzer.py         # Data type/quality analysis
│   ├── preprocessing_selector.py # Auto preprocessing selection
│   ├── ensemble_builder.py      # Voting/Stacking ensemble (P2)
│   └── reporter.py              # AutoML run reports
```

## 🔄 Proposed AutoML Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    mlcli automl --config                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  1. DATA ANALYSIS (data_analyzer.py)                        │
│     • Infer data types (numeric, categorical, text)         │
│     • Detect missing values, class imbalance                │
│     • Determine task type (classification/regression)       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  2. PREPROCESSING SELECTION (preprocessing_selector.py)     │
│     • Select appropriate scalers                            │
│     • Select encoders for categorical features              │
│     • Select feature selection method                       │
│     • Build PreprocessingPipeline                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  3. MODEL SELECTION (model_selector.py)                     │
│     • Filter compatible models from Registry                │
│     • Optionally filter by user preferences (fast/accurate) │
│     • Generate candidate list                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  4. SEARCH SPACE GENERATION (search_space.py)               │
│     • Get default param_space for each model                │
│     • Adjust based on data size (smaller space for big data)│
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  5. MODEL TRAINING & TUNING (Reuse existing tuners)         │
│     • For each candidate model:                             │
│       - Use OptunaTuner (Bayesian) for efficiency           │
│       - Track with ExperimentTracker                        │
│       - Respect time budget                                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  6. COMPARISON & RANKING (reporter.py)                      │
│     • Rank models by scoring metric                         │
│     • Generate comparison report                            │
│     • Return best model or ensemble                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  7. OUTPUT                                                   │
│     • Save best model (via Trainer.save())                  │
│     • Save AutoML report (JSON/HTML)                        │
│     • Log to ExperimentTracker                              │
└─────────────────────────────────────────────────────────────┘
```

## 📋 Configuration Schema

### New `automl_config.json` Format

```json
{
  "dataset": {
    "path": "data/train.csv",
    "target_column": "label"
  },
  "automl": {
    "task": "classification",
    "metric": "accuracy",
    "time_budget_minutes": 30,
    "models": ["random_forest", "xgboost", "lightgbm", "logistic_regression"],
    "models": "auto",
    "tuning_method": "bayesian",
    "n_trials_per_model": 50,
    "cv_folds": 5,
    "preprocessing": "auto",
    "ensemble": false,
    "early_stopping_rounds": 10 
  },
  "training": {
    "test_size": 0.2,
    "random_state": 42
  },
  "output": {
    "model_dir": "artifacts/automl",
    "report_path": "reports/automl_report.html",
    "save_all_models": false
  }
}
```

## 🔌 Integration Points

### 1. CLI Integration

```python
# New command in cli.py
@app.command("automl")
def automl_run(
    config: Path,
    time_budget: int = 30,
    metric: str = "accuracy",
    models: List[str] = None,
    verbose: bool = True,
):
    """Run AutoML pipeline."""
```

### 2. Registry Integration

```python
# Use existing registry to get compatible models
registry = get_registry()
classification_models = registry.get_models_by_type("classification")
```

### 3. Tuner Integration

```python
# Reuse OptunaTuner for each model
tuner = TunerFactory.create("bayesian", param_space, n_trials=50)
results = tuner.tune(trainer_class, X, y)
```

### 4. Tracker Integration

```python
# Log AutoML runs to experiment tracker
tracker.start_run(model_type="automl", config=automl_config)
tracker.log_params({"candidate_models": models, "time_budget": time_budget})
tracker.log_metrics({"best_model": best_model, "best_score": best_score})
```

## 🔐 Assumptions

1. **Classification First**: Initial implementation focuses on classification tasks
2. **Scikit-learn Models**: Only sklearn-compatible models in v1 AutoML
3. **Tabular Data**: Only CSV/tabular data supported
4. **Single Machine**: No distributed AutoML
5. **Existing Models Only**: Use registered trainers, no new model implementations
6. **Bayesian Default**: Use Optuna (TPE) as default tuning method for efficiency

## 📊 Success Metrics

1. AutoML should find model within 5% of manual tuning performance
2. Time budget should be respected (±10%)
3. All existing CLI commands continue to work unchanged
4. Memory usage stays within 2x of single model training

---

_Document generated for AutoML Integration Planning_
_Phase 1 - Design Strategy_
