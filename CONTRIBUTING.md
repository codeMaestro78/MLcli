# Contributing to MLCLI

Thank you for your interest in contributing to MLCLI! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up the development environment
4. Create a branch for your changes
5. Make your changes
6. Submit a pull request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/mlcli.git
cd mlcli

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Making Changes

### Branch Naming

- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Test additions or modifications

Example: `feature/add-lightgbm-trainer`

### Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Tests
- `chore`: Maintenance

Example:
```
feat(trainers): add LightGBM trainer support

- Implement LightGBMTrainer class
- Add configuration schema
- Include example config file
```

## Pull Request Process

1. Update documentation for any new features
2. Add tests for new functionality
3. Ensure all tests pass: `pytest`
4. Update CHANGELOG.md
5. Request review from maintainers

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] All CI checks pass

## Coding Standards

### Python Style

- Follow [PEP 8](https://pep8.org/)
- Use type hints for function signatures
- Maximum line length: 100 characters
- Use docstrings for all public functions/classes

### Example

```python
from typing import Dict, Optional, Any
import numpy as np

def process_data(
    data: np.ndarray,
    config: Dict[str, Any],
    normalize: bool = True,
    *,
    verbose: Optional[int] = None,
) -> np.ndarray:
    """
    Process input data according to configuration.

    Args:
        data: Input array of shape (n_samples, n_features)
        config: Processing configuration dictionary
        normalize: Whether to normalize the data
        verbose: Verbosity level (0=silent, 1=progress, 2=debug)

    Returns:
        Processed data array

    Raises:
        ValueError: If data is empty or config is invalid

    Example:
        >>> data = np.array([[1, 2], [3, 4]])
        >>> result = process_data(data, {"scale": True})
    """
    if data.size == 0:
        raise ValueError("Data cannot be empty")
    
    # Implementation
    return processed_data
```

### Adding a New Trainer

1. Create `mlcli/trainers/your_trainer.py`
2. Inherit from `BaseTrainer`
3. Implement required methods:
   - `__init__`
   - `train`
   - `predict`
   - `evaluate`
   - `save`
   - `load`
4. Register in `mlcli/trainers/__init__.py`
5. Add example config in `examples/configs/`
6. Add tests in `tests/trainers/`
7. Update documentation

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=mlcli --cov-report=html

# Run specific test file
pytest tests/test_trainers.py

# Run specific test
pytest tests/test_trainers.py::test_rf_trainer
```

### Writing Tests

```python
import pytest
from mlcli.trainers import RFTrainer

class TestRFTrainer:
    @pytest.fixture
    def trainer(self):
        return RFTrainer(config={"params": {"n_estimators": 10}})

    def test_train(self, trainer, sample_data):
        X, y = sample_data
        trainer.train(X, y)
        assert trainer.model is not None

    def test_predict(self, trainer, sample_data):
        X, y = sample_data
        trainer.train(X, y)
        predictions = trainer.predict(X)
        assert len(predictions) == len(y)
```

## Documentation

- Use Markdown for documentation
- Include code examples
- Keep README.md updated
- Add docstrings to all public APIs

### Building Docs Locally

```bash
cd docs
pip install -r requirements.txt
mkdocs serve
```

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions
- Check existing issues before creating new ones

Thank you for contributing! 🎉
