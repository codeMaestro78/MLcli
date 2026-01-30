"""
Tests for MLCLI AutoML Module

Comprehensive test suite for all AutoML components:
- SearchSpaceGenerator
- ModelSelector
- DataAnalyzer
- PreprocessingSelector
- AutoMLClassifier
- AutoMLReporter
"""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path


# Fixtures


@pytest.fixture
def binary_data():
    # Generate binary classification data.
    np.random.seed(42)
    X = np.random.randn(300, 10)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


@pytest.fixture
def multiclass_data():
    # Generate multiclass classification data.
    np.random.seed(42)
    X = np.random.randn(300, 10)
    y = (X[:, 0] > 0.5).astype(int) + (X[:, 0] < -0.5).astype(int)
    return X, y


@pytest.fixture
def imbalanced_data():
    # Generate imbalanced classification data.
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.array([0] * 95 + [1] * 5)  # 95:5 imbalance
    return X, y


@pytest.fixture
def data_with_missing():
    # Generate data with missing values.
    np.random.seed(42)
    X = np.random.randn(100, 5)
    X[0:20, 0] = np.nan  # 20% missing in column 0
    y = (X[:, 1] > 0).astype(int)
    return X, y


@pytest.fixture
def temp_output_dir():
    # Create temporary directory for output files.
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# SearchSpaceGenerator Tests


class TestSearchSpaceGenerator:
    """Tests for SearchSpaceGenerator."""

    def test_init(self):
        # Test initialization.
        from mlcli.automl import SearchSpaceGenerator

        gen = SearchSpaceGenerator()
        assert gen is not None

    def test_get_space_logistic_regression(self):
        """Test logistic regression search space."""
        from mlcli.automl import SearchSpaceGenerator

        gen = SearchSpaceGenerator()
        space = gen.get_space("logistic_regression")

        assert len(space) > 0
        assert "C" in space or "max_iter" in space

    def test_get_space_random_forest(self):
        """Test random forest search space."""
        from mlcli.automl import SearchSpaceGenerator

        gen = SearchSpaceGenerator()
        space = gen.get_space("random_forest")

        assert "n_estimators" in space
        assert len(space) >= 3

    def test_get_space_xgboost(self):
        """Test XGBoost search space."""
        from mlcli.automl import SearchSpaceGenerator

        gen = SearchSpaceGenerator()
        space = gen.get_space("xgboost")

        assert len(space) > 0
        assert "n_estimators" in space or "max_depth" in space

    def test_get_space_lightgbm(self):
        """Test LightGBM search space."""
        from mlcli.automl import SearchSpaceGenerator

        gen = SearchSpaceGenerator()
        space = gen.get_space("lightgbm")

        assert len(space) > 0

    def test_get_space_unknown_model(self):
        """Test unknown model returns empty space."""
        from mlcli.automl import SearchSpaceGenerator

        gen = SearchSpaceGenerator()
        space = gen.get_space("unknown_model")

        assert len(space) == 0

    def test_list_models(self):
        """Test listing supported models."""
        from mlcli.automl import SearchSpaceGenerator

        gen = SearchSpaceGenerator()
        models = gen.list_models()

        assert isinstance(models, list)
        assert "random_forest" in models
        assert "logistic_regression" in models

    def test_custom_spaces(self):
        """Test custom search spaces."""
        from mlcli.automl import SearchSpaceGenerator

        custom = {"custom_model": {"param1": [1, 2, 3]}}
        gen = SearchSpaceGenerator(custom_spaces=custom)
        space = gen.get_space("custom_model")

        assert "param1" in space


# ============================================================================
# ModelSelector Tests
# ============================================================================


class TestModelSelector:
    """Tests for ModelSelector."""

    def test_init(self):
        """Test initialization."""
        from mlcli.automl import ModelSelector

        selector = ModelSelector()
        assert selector is not None

    def test_select_models_default(self):
        """Test default model selection."""
        from mlcli.automl import ModelSelector

        selector = ModelSelector()
        candidates = selector.select_models(n_samples=1000, n_features=10)

        assert len(candidates) > 0
        assert all(hasattr(c, "name") for c in candidates)

    def test_select_models_with_include(self):
        """Test model selection with include filter."""
        from mlcli.automl import ModelSelector

        selector = ModelSelector(include_models=["logistic_regression", "xgboost"])
        candidates = selector.select_models(n_samples=1000)

        names = [c.name for c in candidates]
        assert all(n in ["logistic_regression", "xgboost"] for n in names)

    def test_select_models_with_exclude(self):
        """Test model selection with exclude filter."""
        from mlcli.automl import ModelSelector

        selector = ModelSelector(exclude_models=["xgboost", "lightgbm"])
        candidates = selector.select_models(n_samples=1000)

        names = [c.name for c in candidates]
        assert "xgboost" not in names
        assert "lightgbm" not in names

    def test_select_models_fast_mode(self):
        """Test fast mode selection."""
        from mlcli.automl import ModelSelector

        selector = ModelSelector(fast_mode=True)
        candidates = selector.select_models(n_samples=5000)

        # Fast mode should return fewer models
        assert len(candidates) <= 4

    def test_select_models_small_dataset(self):
        """Test selection for small datasets."""
        from mlcli.automl import ModelSelector

        selector = ModelSelector()
        candidates = selector.select_models(n_samples=50, n_features=2)

        assert len(candidates) > 0

    def test_select_models_large_dataset(self):
        """Test selection for large datasets."""
        from mlcli.automl import ModelSelector

        selector = ModelSelector()
        candidates = selector.select_models(n_samples=100000, n_features=100)

        assert len(candidates) > 0

    def test_model_candidate_attributes(self):
        """Test ModelCandidate has required attributes."""
        from mlcli.automl import ModelSelector

        selector = ModelSelector()
        candidates = selector.select_models(n_samples=1000)

        if candidates:
            c = candidates[0]
            assert hasattr(c, "name")
            assert hasattr(c, "framework")
            assert hasattr(c, "priority")


# ============================================================================
# DataAnalyzer Tests
# ============================================================================


class TestDataAnalyzer:
    """Tests for DataAnalyzer."""

    def test_init(self):
        """Test initialization."""
        from mlcli.automl import DataAnalyzer

        analyzer = DataAnalyzer()
        assert analyzer is not None

    def test_analyze_binary(self, binary_data):
        """Test analysis of binary classification data."""
        from mlcli.automl import DataAnalyzer

        X, y = binary_data
        analyzer = DataAnalyzer()
        report = analyzer.analyze(X, y)

        assert report.n_samples == 300
        assert report.n_features == 10
        assert len(report.numeric_columns) == 10

    def test_analyze_single_feature(self):
        """Test analysis with single feature."""
        from mlcli.automl import DataAnalyzer

        np.random.seed(42)
        X = np.random.randn(100, 1)
        y = (X[:, 0] > 0).astype(int)

        analyzer = DataAnalyzer()
        report = analyzer.analyze(X, y)

        assert report.n_features == 1

    def test_analyze_imbalanced(self, imbalanced_data):
        """Test detection of imbalanced classes."""
        from mlcli.automl import DataAnalyzer

        X, y = imbalanced_data
        analyzer = DataAnalyzer()
        report = analyzer.analyze(X, y)

        assert report.is_imbalanced is True

    def test_analyze_missing_values(self, data_with_missing):
        """Test detection of missing values."""
        from mlcli.automl import DataAnalyzer

        X, y = data_with_missing
        analyzer = DataAnalyzer()
        report = analyzer.analyze(X, y)

        assert report.total_missing > 0
        assert report.missing_ratio > 0

    def test_analyze_constant_columns(self):
        """Test handling of constant columns."""
        from mlcli.automl import DataAnalyzer

        X = np.ones((100, 3))
        X[:, 1] = np.random.randn(100)
        y = (X[:, 1] > 0).astype(int)

        analyzer = DataAnalyzer()
        report = analyzer.analyze(X, y)

        assert report is not None

    def test_analyze_with_feature_names(self, binary_data):
        """Test analysis with custom feature names."""
        from mlcli.automl import DataAnalyzer

        X, y = binary_data
        names = [f"feature_{i}" for i in range(10)]

        analyzer = DataAnalyzer()
        report = analyzer.analyze(X, y, feature_names=names)

        # Report should contain feature info
        assert report.n_features == 10


# ============================================================================
# PreprocessingSelector Tests
# ============================================================================


class TestPreprocessingSelector:
    """Tests for PreprocessingSelector."""

    def test_init(self):
        """Test initialization."""
        from mlcli.automl import PreprocessingSelector

        selector = PreprocessingSelector()
        assert selector is not None

    def test_select_numeric_only(self, binary_data):
        """Test plan for numeric-only data."""
        from mlcli.automl import DataAnalyzer, PreprocessingSelector

        X, y = binary_data
        analyzer = DataAnalyzer()
        report = analyzer.analyze(X, y)

        selector = PreprocessingSelector()
        plan = selector.select(report)

        assert plan.scaler_method is not None

    def test_select_with_missing(self, data_with_missing):
        """Test plan for data with missing values."""
        from mlcli.automl import DataAnalyzer, PreprocessingSelector

        X, y = data_with_missing
        analyzer = DataAnalyzer()
        report = analyzer.analyze(X, y)

        selector = PreprocessingSelector()
        plan = selector.select(report)

        assert plan.has_missing_values is True
        assert plan.missing_strategy is not None

    def test_build_pipeline(self, binary_data):
        """Test building preprocessing pipeline."""
        from mlcli.automl import DataAnalyzer, PreprocessingSelector

        X, y = binary_data
        analyzer = DataAnalyzer()
        report = analyzer.analyze(X, y)

        selector = PreprocessingSelector()
        plan = selector.select(report)

        # Build pipeline if method exists
        if hasattr(selector, "build_pipeline"):
            pipeline = selector.build_pipeline(plan)
            assert pipeline is not None

    def test_get_plan(self, binary_data):
        """Test getting the current plan."""
        from mlcli.automl import DataAnalyzer, PreprocessingSelector

        X, y = binary_data
        analyzer = DataAnalyzer()
        report = analyzer.analyze(X, y)

        selector = PreprocessingSelector()
        selector.select(report)

        plan = selector.get_plan()
        assert plan is not None


# ============================================================================
# AutoMLClassifier Tests
# ============================================================================


class TestAutoMLClassifier:
    """Tests for AutoMLClassifier."""

    def test_init_default(self):
        """Test default initialization."""
        from mlcli.automl import AutoMLClassifier

        automl = AutoMLClassifier(verbose=False)
        assert automl is not None

    def test_init_with_config(self):
        """Test initialization with config."""
        from mlcli.automl import AutoMLClassifier

        automl = AutoMLClassifier(
            max_models=3,
            n_trials_per_model=5,
            metric="accuracy",
            cv=3,
            verbose=False,
        )
        assert automl.max_models == 3
        assert automl.n_trials_per_model == 5

    def test_init_with_include_models(self):
        """Test initialization with model filter."""
        from mlcli.automl import AutoMLClassifier

        automl = AutoMLClassifier(
            include_models=["random_forest", "logistic_regression"],
            verbose=False,
        )
        assert "random_forest" in automl.include_models

    def test_init_invalid_metric(self):
        """Test rejection of invalid metric."""
        from mlcli.automl import AutoMLClassifier

        with pytest.raises(ValueError):
            AutoMLClassifier(metric="invalid_metric", verbose=False)

    def test_fit_predict_binary(self, binary_data):
        """Test fit and predict on binary data."""
        from mlcli.automl import AutoMLClassifier

        X, y = binary_data
        automl = AutoMLClassifier(
            max_models=1,
            n_trials_per_model=2,
            include_models=["logistic_regression"],
            cv=2,
            verbose=False,
        )
        automl.fit(X, y)

        assert automl.is_fitted_ is True
        assert automl.best_model_ is not None
        assert automl.best_score_ > 0

        preds = automl.predict(X[:10])
        assert len(preds) == 10

    def test_fit_predict_multiclass(self, multiclass_data):
        """Test fit and predict on multiclass data."""
        from mlcli.automl import AutoMLClassifier

        X, y = multiclass_data
        n_classes = len(np.unique(y))

        automl = AutoMLClassifier(
            max_models=1,
            n_trials_per_model=2,
            include_models=["logistic_regression"],
            cv=2,
            verbose=False,
        )
        automl.fit(X, y)

        proba = automl.predict_proba(X[:10])
        assert proba.shape == (10, n_classes)

    def test_predict_proba(self, binary_data):
        """Test probability predictions."""
        from mlcli.automl import AutoMLClassifier

        X, y = binary_data
        automl = AutoMLClassifier(
            max_models=1,
            n_trials_per_model=2,
            include_models=["logistic_regression"],
            cv=2,
            verbose=False,
        )
        automl.fit(X, y)

        proba = automl.predict_proba(X[:10])
        assert proba.shape == (10, 2)
        # Probabilities should sum to 1
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_predict_unfitted(self, binary_data):
        """Test prediction on unfitted model raises error."""
        from mlcli.automl import AutoMLClassifier

        X, _ = binary_data
        automl = AutoMLClassifier(verbose=False)

        with pytest.raises((RuntimeError, ValueError)):
            automl.predict(X[:10])

    def test_get_best_model_unfitted(self):
        """Test get_best_model on unfitted raises error."""
        from mlcli.automl import AutoMLClassifier

        automl = AutoMLClassifier(verbose=False)

        with pytest.raises((RuntimeError, ValueError)):
            automl.get_best_model()

    def test_fit_invalid_X_none(self, binary_data):
        """Test fit with None X raises error."""
        from mlcli.automl import AutoMLClassifier

        _, y = binary_data
        automl = AutoMLClassifier(verbose=False)

        with pytest.raises((ValueError, TypeError)):
            automl.fit(None, y)

    def test_fit_invalid_y_none(self, binary_data):
        """Test fit with None y raises error."""
        from mlcli.automl import AutoMLClassifier

        X, _ = binary_data
        automl = AutoMLClassifier(verbose=False)

        with pytest.raises((ValueError, TypeError)):
            automl.fit(X, None)

    def test_fit_mismatched_shapes(self):
        """Test fit with mismatched X and y shapes."""
        from mlcli.automl import AutoMLClassifier

        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 50)  # Different length

        automl = AutoMLClassifier(verbose=False)

        with pytest.raises(ValueError):
            automl.fit(X, y)

    def test_get_leaderboard(self, binary_data):
        """Test getting leaderboard."""
        from mlcli.automl import AutoMLClassifier

        X, y = binary_data
        automl = AutoMLClassifier(
            max_models=2,
            n_trials_per_model=2,
            include_models=["logistic_regression", "random_forest"],
            cv=2,
            verbose=False,
        )
        automl.fit(X, y)

        leaderboard = automl.get_leaderboard()
        assert len(leaderboard) == 2
        assert all("rank" in entry for entry in leaderboard)
        assert all("model_name" in entry for entry in leaderboard)

    def test_get_summary(self, binary_data):
        """Test getting summary."""
        from mlcli.automl import AutoMLClassifier

        X, y = binary_data
        automl = AutoMLClassifier(
            max_models=1,
            n_trials_per_model=2,
            include_models=["logistic_regression"],
            cv=2,
            verbose=False,
        )
        automl.fit(X, y)

        summary = automl.get_summary()
        assert "best_model" in summary
        assert "leaderboard" in summary
        assert "config" in summary

    def test_get_data_report(self, binary_data):
        """Test getting data analysis report."""
        from mlcli.automl import AutoMLClassifier

        X, y = binary_data
        automl = AutoMLClassifier(
            max_models=1,
            n_trials_per_model=2,
            include_models=["logistic_regression"],
            cv=2,
            verbose=False,
        )
        automl.fit(X, y)

        report = automl.get_data_report()
        assert report is not None
        assert report.n_samples == 300

    def test_auto_preprocess_false(self, binary_data):
        """Test with auto_preprocess disabled."""
        from mlcli.automl import AutoMLClassifier

        X, y = binary_data
        automl = AutoMLClassifier(
            max_models=1,
            n_trials_per_model=2,
            include_models=["logistic_regression"],
            auto_preprocess=False,
            cv=2,
            verbose=False,
        )
        automl.fit(X, y)

        assert automl.preprocessing_pipeline_ is None


class TestAutoMLReporter:
    """Tests for AutoMLReporter."""

    @pytest.fixture
    def fitted_automl(self, binary_data):
        """Create a fitted AutoML instance."""
        from mlcli.automl import AutoMLClassifier

        X, y = binary_data
        automl = AutoMLClassifier(
            max_models=2,
            n_trials_per_model=2,
            include_models=["logistic_regression", "random_forest"],
            cv=2,
            verbose=False,
        )
        automl.fit(X, y)
        return automl

    def test_init(self, fitted_automl):
        """Test reporter initialization."""
        from mlcli.automl import AutoMLReporter

        reporter = AutoMLReporter(fitted_automl)
        assert reporter is not None
        assert reporter.summary is not None

    def test_init_unfitted_raises(self):
        """Test reporter with unfitted AutoML raises error."""
        from mlcli.automl import AutoMLClassifier, AutoMLReporter

        automl = AutoMLClassifier(verbose=False)

        with pytest.raises(RuntimeError):
            AutoMLReporter(automl)

    def test_init_invalid_object(self):
        """Test reporter with invalid object raises error."""
        from mlcli.automl import AutoMLReporter

        with pytest.raises(ValueError):
            AutoMLReporter("invalid")

    def test_to_json(self, fitted_automl):
        """Test JSON output."""
        from mlcli.automl import AutoMLReporter

        reporter = AutoMLReporter(fitted_automl)
        json_str = reporter.to_json()

        assert isinstance(json_str, str)
        assert len(json_str) > 100
        assert "leaderboard" in json_str
        assert "report_metadata" in json_str

        # Should be valid JSON
        data = json.loads(json_str)
        assert "report_metadata" in data

    def test_to_html(self, fitted_automl):
        # Test HTML output.
        from mlcli.automl import AutoMLReporter

        reporter = AutoMLReporter(fitted_automl)
        html_str = reporter.to_html()

        assert isinstance(html_str, str)
        assert "<!DOCTYPE html>" in html_str
        assert "Leaderboard" in html_str

    def test_save_json(self, fitted_automl, temp_output_dir):
        # Test saving JSON to file.
        from mlcli.automl import AutoMLReporter

        reporter = AutoMLReporter(fitted_automl)
        output_path = temp_output_dir / "report.json"

        result_path = reporter.save_json(output_path)

        assert result_path.exists()
        assert result_path.stat().st_size > 0

        # Verify content
        with open(result_path) as f:
            data = json.load(f)
        assert "report_metadata" in data

    def test_save_html(self, fitted_automl, temp_output_dir):
        # Test saving HTML to file.
        from mlcli.automl import AutoMLReporter

        reporter = AutoMLReporter(fitted_automl)
        output_path = temp_output_dir / "report.html"

        result_path = reporter.save_html(output_path)

        assert result_path.exists()
        assert result_path.stat().st_size > 0

        # Verify content
        content = result_path.read_text()
        assert "<!DOCTYPE html>" in content

    def test_refresh(self, fitted_automl):
        # Test refreshing reporter data.
        from mlcli.automl import AutoMLReporter

        reporter = AutoMLReporter(fitted_automl)
        original = reporter.summary

        reporter.refresh()

        assert reporter.summary == original


# Integration Tests


class TestAutoMLIntegration:
    # Integration tests for full AutoML workflow.

    def test_full_workflow(self, binary_data, temp_output_dir):
        """Test complete AutoML workflow."""
        from mlcli.automl import AutoMLClassifier, AutoMLReporter

        X, y = binary_data

        # 1. Create and fit AutoML
        automl = AutoMLClassifier(
            max_models=2,
            n_trials_per_model=3,
            include_models=["random_forest", "logistic_regression"],
            metric="accuracy",
            cv=3,
            verbose=False,
        )
        automl.fit(X, y)

        # 2. Verify fitted state
        assert automl.is_fitted_
        assert automl.best_model_ is not None
        assert automl.best_score_ > 0.5

        # 3. Make predictions
        test_X = np.random.randn(50, 10)
        preds = automl.predict(test_X)
        proba = automl.predict_proba(test_X)

        assert preds.shape == (50,)
        assert proba.shape == (50, 2)

        # 4. Get summary and leaderboard
        summary = automl.get_summary()
        leaderboard = automl.get_leaderboard()

        assert "best_model" in summary
        assert len(leaderboard) == 2

        # 5. Generate reports
        reporter = AutoMLReporter(automl)

        json_path = reporter.save_json(temp_output_dir / "automl_report.json")
        html_path = reporter.save_html(temp_output_dir / "automl_report.html")

        assert json_path.exists()
        assert html_path.exists()

    def test_different_metrics(self, binary_data):
        """Test with different evaluation metrics."""
        from mlcli.automl import AutoMLClassifier

        X, y = binary_data

        for metric in ["accuracy", "f1", "roc_auc"]:
            automl = AutoMLClassifier(
                max_models=1,
                n_trials_per_model=2,
                include_models=["logistic_regression"],
                metric=metric,
                cv=2,
                verbose=False,
            )
            automl.fit(X, y)

            assert automl.is_fitted_
            assert automl.best_score_ >= 0

    def test_component_interaction(self, binary_data):
        """Test interaction between components."""
        from mlcli.automl import (
            DataAnalyzer,
            PreprocessingSelector,
            ModelSelector,
            SearchSpaceGenerator,
        )

        X, y = binary_data

        # 1. Analyze data
        analyzer = DataAnalyzer()
        report = analyzer.analyze(X, y)

        assert report.n_samples == 300
        assert report.n_features == 10

        # 2. Select preprocessing
        preproc_selector = PreprocessingSelector()
        plan = preproc_selector.select(report)

        assert plan is not None

        # 3. Select models
        model_selector = ModelSelector()
        candidates = model_selector.select_models(
            n_samples=report.n_samples,
            n_features=report.n_features,
        )

        assert len(candidates) > 0

        # 4. Get search spaces
        space_gen = SearchSpaceGenerator()
        for candidate in candidates[:2]:
            space = space_gen.get_space(candidate.name)
            assert isinstance(space, dict)
