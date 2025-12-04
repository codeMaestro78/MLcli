"""
Tuner Factory

Factory pattern for creating tuner instances based on method name.
"""

from typing import Any, Dict, Optional, Type
import logging

from mlcli.tuner.base_tuner import BaseTuner
from mlcli.tuner.grid_tuner import GridSearchTuner
from mlcli.tuner.random_tuner import RandomSearchTuner
from mlcli.tuner.optuna_tuner import OptunaTuner

logger = logging.getLogger(__name__)


class TunerFactory:
    """
    Factory for creating hyperparameter tuners.

    Supports: grid, random, bayesian/optuna
    """

    _tuners: Dict[str, Type[BaseTuner]] = {
        "grid": GridSearchTuner,
        "grid_search": GridSearchTuner,
        "random": RandomSearchTuner,
        "random_search": RandomSearchTuner,
        "bayesian": OptunaTuner,
        "optuna": OptunaTuner,
        "tpe": OptunaTuner,
    }

    @classmethod
    def create(cls, method: str, param_space: Dict[str, Any], **kwargs) -> BaseTuner:
        """
        Create a tuner instance.

        Args:
            method: Tuning method ('grid', 'random', 'bayesian')
            param_space: Parameter search space
            **kwargs: Additional arguments for the tuner

        Returns:
            Tuner instance

        Raises:
            ValueError: If method is unknown
        """
        method_lower = method.lower().replace("-", "_").replace(" ", "_")

        if method_lower not in cls._tuners:
            available = ", ".join(sorted(set(cls._tuners.keys())))
            raise ValueError(
                f"Unknown tuning method '{method}'. " f"Available methods: {available}"
            )

        tuner_class = cls._tuners[method_lower]

        # Handle method-specific parameter names
        if method_lower in ["random", "random_search"]:
            # Map n_trials to n_iter for random search
            if "n_trials" in kwargs and "n_iter" not in kwargs:
                kwargs["n_iter"] = kwargs.pop("n_trials")

        logger.info(f"Creating {tuner_class.__name__} tuner")

        return tuner_class(param_space=param_space, **kwargs)

    @classmethod
    def list_methods(cls) -> list:
        """
        List available tuning methods.

        Returns:
            List of method names
        """
        # Return unique methods (not aliases)
        return ["grid", "random", "bayesian"]

    @classmethod
    def get_method_info(cls, method: str) -> Dict[str, str]:
        """
        Get information about a tuning method.

        Args:
            method: Method name

        Returns:
            Dictionary with method info
        """
        info = {
            "grid": {
                "name": "Grid Search",
                "description": "Exhaustive search over all parameter combinations",
                "best_for": "Small parameter spaces with discrete values",
                "complexity": "O(n^k) where n=values per param, k=num params",
            },
            "random": {
                "name": "Random Search",
                "description": "Random sampling from parameter distributions",
                "best_for": "Large parameter spaces, continuous parameters",
                "complexity": "O(n) where n=number of iterations",
            },
            "bayesian": {
                "name": "Bayesian Optimization (Optuna)",
                "description": "Intelligent search using TPE algorithm",
                "best_for": "Expensive evaluations, complex parameter spaces",
                "complexity": "O(n) with smarter sampling",
            },
        }

        method_lower = method.lower().replace("-", "_").replace(" ", "_")

        # Map aliases to base methods
        alias_map = {
            "grid_search": "grid",
            "random_search": "random",
            "optuna": "bayesian",
            "tpe": "bayesian",
        }

        method_key = alias_map.get(method_lower, method_lower)

        return info.get(method_key, {"name": method, "description": "Unknown method"})


def get_tuner(method: str, param_space: Dict[str, Any], **kwargs) -> BaseTuner:
    """
    Convenience function to create a tuner.

    Args:
        method: Tuning method ('grid', 'random', 'bayesian')
        param_space: Parameter search space
        **kwargs: Additional tuner arguments

    Returns:
        Tuner instance

    Example:
        tuner = get_tuner(
            method="random",
            param_space={"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
            n_iter=50,
            cv=5
        )
    """
    return TunerFactory.create(method, param_space, **kwargs)
