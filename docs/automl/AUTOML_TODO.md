# AutoML Integration - Complete TODO List

> **⚠️ IMPORTANT**: This document requires approval before any implementation begins.
> No code will be written until the user confirms this plan.

---

## 📁 File Impact Analysis

### Files to CREATE (New)

| File                                     | Purpose                         | Priority | Est. Lines |
| ---------------------------------------- | ------------------------------- | -------- | ---------- |
| `mlcli/automl/__init__.py`               | Module exports                  | P0       | ~30        |
| `mlcli/automl/base_automl.py`            | Abstract BaseAutoML class       | P0       | ~150       |
| `mlcli/automl/automl_classifier.py`      | AutoMLClassifier implementation | P0       | ~400       |
| `mlcli/automl/model_selector.py`         | Model selection logic           | P0       | ~200       |
| `mlcli/automl/search_space.py`           | Default param spaces            | P0       | ~300       |
| `mlcli/automl/data_analyzer.py`          | Data analysis & inference       | P1       | ~250       |
| `mlcli/automl/preprocessing_selector.py` | Auto preprocessing              | P1       | ~200       |
| `mlcli/automl/reporter.py`               | AutoML reporting                | P1       | ~300       |
| `mlcli/automl/ensemble_builder.py`       | Ensemble creation               | P2       | ~250       |
| `configs/automl_config.json`             | Example AutoML config           | P0       | ~40        |
| `tests/test_automl.py`                   | AutoML unit tests               | P0       | ~200       |
| `docs/automl/USER_GUIDE.md`              | User documentation              | P1       | ~300       |

### Files to MODIFY (Existing)

| File                      | Changes                                      | Impact              |
| ------------------------- | -------------------------------------------- | ------------------- |
| `mlcli/cli.py`            | Add `automl` command (~100 lines)            | Low - additive only |
| `mlcli/__init__.py`       | Export automl module                         | Minimal             |
| `mlcli/utils/registry.py` | Add `get_models_by_type()` method if missing | Low                 |
| `pyproject.toml`          | Update version to 0.4.0                      | Minimal             |
| `README.md`               | Add AutoML section                           | Low                 |

### Files to LEAVE UNCHANGED

- All existing trainers (`mlcli/trainers/*`)
- All existing tuners (`mlcli/tuner/*`)
- All existing preprocessors (`mlcli/preprocessor/*`)
- Experiment tracker (`mlcli/runner/experiment_tracker.py`)
- TUI (`mlcli/ui/tui.py`)
- All existing tests

---

## 📋 Detailed TODO List

### PHASE 1: Core Infrastructure (P0)

#### 1.1 Create AutoML Module Structure

- [ ] **TODO-001**: Create `mlcli/automl/` directory
- [ ] **TODO-002**: Create `mlcli/automl/__init__.py` with module exports
  ```python
  from .base_automl import BaseAutoML
  from .automl_classifier import AutoMLClassifier
  from .model_selector import ModelSelector
  from .search_space import SearchSpaceGenerator
  ```

#### 1.2 Implement BaseAutoML Abstract Class

- [ ] **TODO-003**: Create `mlcli/automl/base_automl.py`
  - Abstract methods: `fit()`, `predict()`, `predict_proba()`, `get_best_model()`
  - Properties: `best_model_`, `best_score_`, `best_params_`, `leaderboard_`
  - Time budget management logic
  - Integration with ExperimentTracker

#### 1.3 Implement Search Space Generator

- [ ] **TODO-004**: Create `mlcli/automl/search_space.py`
  - Default param spaces for:
    - `logistic_regression`: C, penalty, solver
    - `svm`: C, kernel, gamma
    - `random_forest`: n_estimators, max_depth, min_samples_split
    - `xgboost`: n_estimators, max_depth, learning_rate, subsample
    - `lightgbm`: n_estimators, num_leaves, learning_rate
    - `catboost`: iterations, depth, learning_rate
  - Optuna-compatible format (type, low, high, choices)
  - Data-size aware adjustments

#### 1.4 Implement Model Selector

- [ ] **TODO-005**: Create `mlcli/automl/model_selector.py`
  - `select_models(task, data_shape, user_prefs)` method
  - Filter by task type (classification/regression)
  - Filter by data size (exclude slow models for large data)
  - Support user model whitelist/blacklist
  - Model priority ranking

#### 1.5 Implement AutoMLClassifier

- [ ] **TODO-006**: Create `mlcli/automl/automl_classifier.py`
  - Inherits from `BaseAutoML`
  - Implements `fit(X, y)` method:
    1. Analyze data
    2. Select candidate models
    3. Generate search spaces
    4. For each model: tune with time budget
    5. Rank results
    6. Store best model
  - Implements `predict(X)`, `predict_proba(X)`
  - Time budget distribution across models
  - Progress reporting with Rich

---

### PHASE 2: Data Analysis & Preprocessing (P1)

#### 2.1 Implement Data Analyzer

- [ ] **TODO-007**: Create `mlcli/automl/data_analyzer.py`
  - Detect column types (numeric, categorical, datetime, text)
  - Count missing values per column
  - Detect class imbalance
  - Infer task type from target
  - Generate data quality report

#### 2.2 Implement Preprocessing Selector

- [ ] **TODO-008**: Create `mlcli/automl/preprocessing_selector.py`
  - Select scalers based on data distribution
  - Select encoders based on categorical cardinality
  - Handle missing values (imputation strategy)
  - Build `PreprocessingPipeline` automatically
  - Leverage existing `mlcli/preprocessor/` components

---

### PHASE 3: CLI Integration (P0)

#### 3.1 Add AutoML Command

- [ ] **TODO-009**: Add `automl` command to `mlcli/cli.py`
  ```python
  @app.command("automl")
  def automl_run(
      config: Path = Option(..., "--config", "-c"),
      time_budget: int = Option(30, "--time-budget", "-t"),
      metric: str = Option("accuracy", "--metric", "-m"),
      models: str = Option(None, "--models"),
      output: Path = Option(None, "--output", "-o"),
      verbose: bool = Option(True, "--verbose/--quiet"),
  ):
  ```
  - Load config with ConfigLoader
  - Instantiate AutoMLClassifier
  - Run fit()
  - Display leaderboard
  - Save best model

---

### PHASE 4: Reporting (P1)

#### 4.1 Implement AutoML Reporter

- [ ] **TODO-010**: Create `mlcli/automl/reporter.py`
  - Generate leaderboard table (Rich)
  - Export JSON report
  - Export HTML report (optional)
  - Include: model name, score, params, training time
  - Comparison charts (text-based with plotext)

---

### PHASE 5: Testing (P0)

#### 5.1 Unit Tests

- [ ] **TODO-011**: Create `tests/test_automl.py`
  - Test `SearchSpaceGenerator` returns valid spaces
  - Test `ModelSelector` filters correctly
  - Test `DataAnalyzer` infers types correctly
  - Test `AutoMLClassifier.fit()` runs without error
  - Test time budget is respected (±20%)
  - Test best model is retrievable
  - Test predictions work after fit

#### 5.2 Integration Tests

- [ ] **TODO-012**: Add integration test for CLI automl command
  - Run `mlcli automl --config <test_config>`
  - Verify model saved
  - Verify report generated

---

### PHASE 6: Configuration & Examples (P0)

#### 6.1 Example Configuration

- [ ] **TODO-013**: Create `configs/automl_config.json`
  ```json
  {
    "dataset": { "path": "data/sample_data.csv", "target_column": "target" },
    "automl": {
      "task": "classification",
      "metric": "accuracy",
      "time_budget_minutes": 10,
      "models": "auto",
      "tuning_method": "bayesian",
      "n_trials_per_model": 20,
      "cv_folds": 5
    }
  }
  ```

---

### PHASE 7: Documentation (P1)

#### 7.1 User Guide

- [ ] **TODO-014**: Create `docs/automl/USER_GUIDE.md`
  - Quick start example
  - Configuration options reference
  - Model selection options
  - Interpreting results
  - Advanced usage (custom models, custom spaces)

#### 7.2 Update README

- [ ] **TODO-015**: Update `README.md`
  - Add AutoML feature to feature list
  - Add AutoML quick example

---

### PHASE 8: Ensemble (P2 - Future)

#### 8.1 Ensemble Builder

- [ ] **TODO-016**: Create `mlcli/automl/ensemble_builder.py` (Future)
  - Voting ensemble from top-N models
  - Stacking ensemble option
  - Automatic weight optimization

---

## 📊 Implementation Order

```
Week 1: Core (P0)
├── TODO-001: Create module structure
├── TODO-002: __init__.py
├── TODO-003: base_automl.py
├── TODO-004: search_space.py
├── TODO-005: model_selector.py
└── TODO-006: automl_classifier.py

Week 2: Integration (P0)
├── TODO-009: CLI automl command
├── TODO-011: Unit tests
├── TODO-012: Integration tests
└── TODO-013: Example config

Week 3: Enhancement (P1)
├── TODO-007: data_analyzer.py
├── TODO-008: preprocessing_selector.py
├── TODO-010: reporter.py
├── TODO-014: User guide
└── TODO-015: Update README

Future (P2)
└── TODO-016: ensemble_builder.py
```

---

## ✅ Approval Checklist

Before implementation, please confirm:

- [ ] **Scope**: Are P0/P1/P2 priorities acceptable?
- [ ] **Architecture**: Is `mlcli/automl/` module structure acceptable?
- [ ] **Models**: Should AutoML include TensorFlow models or sklearn-only?
- [ ] **Time Budget**: Default 30 minutes acceptable?
- [ ] **Backward Compatibility**: Confirmed no changes to existing workflows?
- [ ] **Testing**: pytest testing approach acceptable?
- [ ] **Documentation**: Markdown docs sufficient?

---

## 🚀 Ready to Implement?

**Please reply with:**

- `APPROVED` - Proceed with implementation
- `MODIFY <item>` - Request changes to specific items
- `QUESTION <topic>` - Need clarification

---

_Document generated for AutoML Integration Planning_
_Phase 2 - Complete TODO List_
