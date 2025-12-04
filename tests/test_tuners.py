"""
Tests for MLCLI Tuners
"""

import pytest
import numpy as np


class TestGridTuner:
    """Tests for Grid Search Tuner."""

    def test_init(self):
        """Test tuner initialization."""
        from mlcli.tuner import GridSearchTuner
        
        param_space = {
            "n_estimators": [10, 50],
            "max_depth": [3, 5],
        }
        
        tuner = GridSearchTuner(param_space=param_space, cv=2)
        assert tuner is not None
        assert tuner.n_combinations == 4

    def test_tune(self, sample_data):
        """Test grid search tuning."""
        from mlcli.tuner import GridSearchTuner
        from mlcli.trainers import RFTrainer
        
        X, y = sample_data
        param_space = {
            "n_estimators": [10, 20],
            "max_depth": [3, 5],
        }
        
        tuner = GridSearchTuner(param_space=param_space, cv=2, verbose=0)
        results = tuner.tune(RFTrainer, X, y)
        
        assert "best_params" in results
        assert "best_score" in results
        assert results["best_score"] > 0


class TestRandomTuner:
    """Tests for Random Search Tuner."""

    def test_init(self):
        """Test tuner initialization."""
        from mlcli.tuner import RandomSearchTuner
        
        param_space = {
            "n_estimators": {"type": "int", "low": 10, "high": 100},
            "max_depth": [3, 5, 10],
        }
        
        tuner = RandomSearchTuner(param_space=param_space, n_iter=5, cv=2)
        assert tuner is not None

    def test_tune(self, sample_data):
        """Test random search tuning."""
        from mlcli.tuner import RandomSearchTuner
        from mlcli.trainers import RFTrainer
        
        X, y = sample_data
        param_space = {
            "n_estimators": [10, 20, 30],
            "max_depth": [3, 5],
        }
        
        tuner = RandomSearchTuner(param_space=param_space, n_iter=3, cv=2, verbose=0)
        results = tuner.tune(RFTrainer, X, y)
        
        assert "best_params" in results
        assert "best_score" in results
