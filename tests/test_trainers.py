"""
Tests for MLCLI Trainers
"""

import pytest
import numpy as np
from pathlib import Path


class TestRFTrainer:
    """Tests for Random Forest Trainer."""

    def test_init(self, rf_config):
        """Test trainer initialization."""
        from mlcli.trainers import RFTrainer
        
        trainer = RFTrainer(config=rf_config)
        assert trainer is not None

    def test_train(self, rf_config, sample_data):
        """Test model training."""
        from mlcli.trainers import RFTrainer
        
        X, y = sample_data
        trainer = RFTrainer(config=rf_config)
        trainer.train(X, y)
        
        assert trainer.model is not None

    def test_predict(self, rf_config, sample_data):
        """Test model prediction."""
        from mlcli.trainers import RFTrainer
        
        X, y = sample_data
        trainer = RFTrainer(config=rf_config)
        trainer.train(X, y)
        
        predictions = trainer.predict(X[:10])
        assert len(predictions) == 10

    def test_evaluate(self, rf_config, sample_data):
        """Test model evaluation."""
        from mlcli.trainers import RFTrainer
        
        X, y = sample_data
        trainer = RFTrainer(config=rf_config)
        trainer.train(X, y)
        
        metrics = trainer.evaluate(X, y)
        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_save_load(self, rf_config, sample_data, temp_dir):
        """Test model save and load."""
        from mlcli.trainers import RFTrainer
        
        X, y = sample_data
        trainer = RFTrainer(config=rf_config)
        trainer.train(X, y)
        
        # Save
        model_path = temp_dir / "rf_model.pkl"
        trainer.save(str(model_path), format="pickle")
        assert model_path.exists()
        
        # Load
        new_trainer = RFTrainer(config=rf_config)
        new_trainer.load(str(model_path))
        
        # Verify predictions match
        pred1 = trainer.predict(X[:5])
        pred2 = new_trainer.predict(X[:5])
        np.testing.assert_array_equal(pred1, pred2)


class TestXGBTrainer:
    """Tests for XGBoost Trainer."""

    def test_init(self, xgb_config):
        """Test trainer initialization."""
        from mlcli.trainers import XGBTrainer
        
        trainer = XGBTrainer(config=xgb_config)
        assert trainer is not None

    def test_train(self, xgb_config, sample_data):
        """Test model training."""
        from mlcli.trainers import XGBTrainer
        
        X, y = sample_data
        trainer = XGBTrainer(config=xgb_config)
        trainer.train(X, y)
        
        assert trainer.model is not None

    def test_predict(self, xgb_config, sample_data):
        """Test model prediction."""
        from mlcli.trainers import XGBTrainer
        
        X, y = sample_data
        trainer = XGBTrainer(config=xgb_config)
        trainer.train(X, y)
        
        predictions = trainer.predict(X[:10])
        assert len(predictions) == 10


class TestLogisticTrainer:
    """Tests for Logistic Regression Trainer."""

    def test_init(self, logistic_config):
        """Test trainer initialization."""
        from mlcli.trainers import LogisticRegressionTrainer
        
        trainer = LogisticRegressionTrainer(config=logistic_config)
        assert trainer is not None

    def test_train(self, logistic_config, sample_data):
        """Test model training."""
        from mlcli.trainers import LogisticRegressionTrainer
        
        X, y = sample_data
        trainer = LogisticRegressionTrainer(config=logistic_config)
        trainer.train(X, y)
        
        assert trainer.model is not None
