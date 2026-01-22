# MLCLI Tests

This directory contains the test suite for MLCLI.

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=mlcli --cov-report=html

# Run specific test file
pytest tests/test_trainers.py

# Run specific test
pytest tests/test_trainers.py::TestRFTrainer::test_train

# Run with verbose output
pytest -v

# Run and stop on first failure
pytest -x
```

## Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures
├── test_trainers.py         # Trainer tests
├── test_tuners.py           # Tuner tests
├── test_preprocessors.py    # Preprocessor tests
├── test_explainers.py       # Explainer tests
├── test_cli.py              # CLI command tests
└── test_utils.py            # Utility function tests
```

## Writing Tests

See [CONTRIBUTING.md](../CONTRIBUTING.md) for testing guidelines.
