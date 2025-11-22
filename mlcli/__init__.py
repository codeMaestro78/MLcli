"""
mlcli - Production ML/DL CLI and TUI

A modular, configuration-driven tool for training, evaluating, and tracking
Machine Learning and Deep Learning models with both CLI and interactive TUI interfaces.
"""


__version__="0.1.0"
__author__="Devarshi Lalani"
__licence__="MIT"

from mlcli.utils.registry import ModelRegistry

# Global model registry instance

registry=ModelRegistry()
__all__=["registry","__version__"]
