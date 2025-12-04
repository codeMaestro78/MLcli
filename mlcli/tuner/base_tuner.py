"""
Base Tuner Abstract Class

Defines the interface for all hyperparameter tuning strategies.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np
from pathlib import Path
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class BaseTuner(ABC):
    """
    Abstract base class for hyperparameter tuners.

    All tuning strategies must implement: tune(), get_best_params(), get_results()
    """

    def __init__(
        self,
        param_space: Dict[str, Any],
        scoring: str = "accuracy",
        cv: int = 5,
        n_jobs: int = -1,
        verbose: int = 1,
        random_state: Optional[int] = 42,
    ):
        """
        Initialize base tuner.

        Args:
            param_space: Dictionary defining the hyperparameter search space
            scoring: Metric to optimize ('accuracy', 'f1', 'roc_auc', 'precision', 'recall')
            cv: Number of cross-validation folds
            n_jobs: Number of parallel jobs (-1 for all cores)
            verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
            random_state: Random seed for reproducibility
        """
        self.param_space = param_space
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

        # Results storage
        self.best_params_: Optional[Dict[str, Any]] = None
        self.best_score_: Optional[float] = None
        self.cv_results_: Optional[Dict[str, Any]] = None
        self.tuning_history_: List[Dict[str, Any]] = []

        # Timing
        self.start_time_: Optional[datetime] = None
        self.end_time_: Optional[datetime] = None

        logger.debug(f"Initialized {self.__class__.__name__} with {len(param_space)} parameters")

    @abstractmethod
    def tune(
        self,
        trainer_class: type,
        X: np.ndarray,
        y: np.ndarray,
        trainer_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning.

        Args:
            trainer_class: The trainer class to tune
            X: Feature matrix
            y: Target vector
            trainer_config: Base configuration for trainer

        Returns:
            Dictionary containing best parameters and results
        """

    @abstractmethod
    def get_best_params(self) -> Dict[str, Any]:
        """
        Get the best hyperparameters found.

        Returns:
            Dictionary of best parameters
        """

    @abstractmethod
    def get_results(self) -> Dict[str, Any]:
        """
        Get detailed tuning results.

        Returns:
            Dictionary with all trial results
        """

    def get_best_score(self) -> float:
        """
        Get the best score achieved.

        Returns:
            Best cross-validation score
        """
        if self.best_score_ is None:
            raise RuntimeError("No tuning has been performed yet. Call tune() first.")
        return self.best_score_

    def get_tuning_duration(self) -> float:
        """
        Get total tuning duration in seconds.

        Returns:
            Duration in seconds
        """
        if self.start_time_ is None or self.end_time_ is None:
            return 0.0
        return (self.end_time_ - self.start_time_).total_seconds()

    def save_results(self, filepath: Path) -> None:
        """
        Save tuning results to JSON file.

        Args:
            filepath: Path to save results
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        results = {
            "tuner_type": self.__class__.__name__,
            "param_space": self._serialize_param_space(),
            "scoring": self.scoring,
            "cv": self.cv,
            "best_params": self.best_params_,
            "best_score": self.best_score_,
            "tuning_history": self.tuning_history_,
            "duration_seconds": self.get_tuning_duration(),
            "timestamp": datetime.now().isoformat(),
        }

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Saved tuning results to {filepath}")

    def load_results(self, filepath: Path) -> Dict[str, Any]:
        """
        Load tuning results from JSON file.

        Args:
            filepath: Path to results file

        Returns:
            Loaded results dictionary
        """
        with open(filepath, "r") as f:
            results = json.load(f)

        self.best_params_ = results.get("best_params")
        self.best_score_ = results.get("best_score")
        self.tuning_history_ = results.get("tuning_history", [])

        return results

    def _serialize_param_space(self) -> Dict[str, Any]:
        """
        Convert param space to JSON-serializable format.

        Returns:
            Serializable param space
        """
        serialized = {}
        for param, value in self.param_space.items():
            if hasattr(value, "__iter__") and not isinstance(value, str):
                serialized[param] = list(value)
            else:
                serialized[param] = str(value)
        return serialized

    def _log_trial(
        self, trial_num: int, params: Dict[str, Any], score: float, duration: float
    ) -> None:
        """
        Log a trial result.

        Args:
            trial_num: Trial number
            params: Parameters used
            score: Score achieved
            duration: Trial duration in seconds
        """
        trial_result = {
            "trial": trial_num,
            "params": params,
            "score": score,
            "duration_seconds": duration,
            "timestamp": datetime.now().isoformat(),
        }
        self.tuning_history_.append(trial_result)

        if self.verbose >= 2:
            logger.info(f"Trial {trial_num}: score={score:.4f}, params={params}")

    def get_top_n_params(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        Get top N parameter combinations by score.

        Args:
            n: Number of top results to return

        Returns:
            List of top parameter combinations with scores
        """
        if not self.tuning_history_:
            return []

        sorted_history = sorted(self.tuning_history_, key=lambda x: x["score"], reverse=True)

        return sorted_history[:n]

    def __repr__(self) -> str:
        """String representation."""
        status = "tuned" if self.best_params_ is not None else "not tuned"
        return f"{self.__class__.__name__}(status={status}, scoring={self.scoring})"
