"""
mlcli - Production ML/DL CLI and TUI

A modular, configuration-driven tool for training, evaluating, and tracking
Machine Learning and Deep Learning models with both CLI and interactive TUI interfaces.
"""

__version__ = "0.3.1"
__author__ = "Devarshi Lalani"
__licence__ = "MIT"

from mlcli.utils.registry import ModelRegistry

# Global model registry instance
registry = ModelRegistry()


def _register_models():
    """Register all models lazily without importing heavy dependencies."""
    from mlcli.trainers import register_all_models

    register_all_models()


# Register models on first access
_register_models()

__all__ = ["registry", "__version__"]
