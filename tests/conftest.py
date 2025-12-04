"""
MLCLI Test Suite

Shared fixtures and configuration for pytest.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil


@pytest.fixture(scope="session")
def sample_data():
    """Generate sample classification data."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    return X, y


@pytest.fixture(scope="session")
def sample_dataframe(sample_data):
    """Generate sample DataFrame."""
    X, y = sample_data
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["target"] = y
    return df


@pytest.fixture(scope="function")
def temp_dir():
    """Create temporary directory for test artifacts."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture(scope="function")
def sample_csv(sample_dataframe, temp_dir):
    """Create temporary CSV file with sample data."""
    csv_path = temp_dir / "sample_data.csv"
    sample_dataframe.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture(scope="session")
def rf_config():
    """Random Forest configuration."""
    return {
        "params": {
            "n_estimators": 10,
            "max_depth": 5,
            "random_state": 42,
        }
    }


@pytest.fixture(scope="session")
def xgb_config():
    """XGBoost configuration."""
    return {
        "params": {
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.1,
            "random_state": 42,
        }
    }


@pytest.fixture(scope="session")
def logistic_config():
    """Logistic Regression configuration."""
    return {
        "params": {
            "C": 1.0,
            "max_iter": 100,
            "random_state": 42,
        }
    }
