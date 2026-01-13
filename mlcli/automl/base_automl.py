"""
Base AutoML Abstract Class

Defines the interface and shared utilities for AutoML implementations.
Tracks best model, best score, best params, and maintains a leaderboard.
"""

from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class LeaderboardEntry:
    # One row in the AutoML leaderboard.
    model_name: str
    framework: str
    score: float
    params: Dict[str, Any]
    duration_seconds: float
    extra: Dict[str, Any] = field(default_factory=dict)


class BaseAutoML(ABC):
    """
    Abstract base class for AutoML implementations.

    Subclasses must implement:
    - fit(X, y) -> self
    - predict(X) -> np.ndarray
    - predict_proba(X) -> Optional[np.ndarray]
    - get_best_model() -> Any
    """

    def __init__(
        self,
        task: str = "classification",
        metric: str = "accuracy",
        time_budget_minutes: Optional[Union[int, float]] = None,
        random_state: Optional[int] = 42,
        tracker: Optional[Any] = None,
        verbose: bool = True,
    ) -> None:
        """
        Initialize BaseAutoML.

        Args:
            task: 'classification' or 'regression'
            metric: Scoring metric (accuracy, f1, roc_auc, etc.)
            time_budget_minutes: Max time for AutoML run (None = no limit)
            random_state: Random seed for reproducibility
            tracker: Optional ExperimentTracker instance
            verbose: Whether to print progress
        """
        self.task = task.lower()
        self.metric = metric
        self.time_budget_minutes = time_budget_minutes
        self.random_state = random_state
        self.verbose = verbose

        # Optinal ExperimentTracker
        self.tracker = tracker

        # Time state
        self._start_time: Optional[datetime] = None
        self._end_time: Optional[datetime] = None

        # Best model state
        self.best_model_: Optional[Any] = None
        self.best_model_name_: Optional[str] = None
        self.best_score_: Optional[float] = None
        self.best_params_: Dict[str, Any] = {}

        # Leaderboard
        self.leaderboard_: List[LeaderboardEntry] = []

        # Fitted flag
        self.is_fitted_: bool = False

        logger.debug(
            f"Initialized {self.__class__.__name__}(task={self.task}, "
            f"metric={self.metric}, time_budget={self.time_budget_minutes})"
        )

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "BaseAutoML":
        """
        Run AutoML search and fit the best model.

        Args:
            X: Feature matrix
            y: Target vector
            **kwargs: Additional arguments

        Returns:
            self
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the best model.

        Args:
            X: Feature matrix

        Returns:
            Predictions array
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        Predict probabilities using the best model.

        Args:
            X: Feature matrix

        Returns:
            Probability array or None if not supported
        """
        pass

    @abstractmethod
    def get_best_model(self) -> Any:
        """
        Get the best fitted model/trainer.

        Returns:
            Best model object
        """
        pass

    def get_leaderboard(self) -> List[Dict[str, Any]]:
        """
        Get leaderboard as list of dicts (JSON-serializable).

        Returns:
            List of leaderboard entries
        """

        return [
            {
                "rank": i + 1,
                "model_name": entry.model_name,
                "framework": entry.framework,
                "score": round(entry.score, 6),
                "params": entry.params,
                "duration_seconds": round(entry.duration_seconds, 2),
                "extra": entry.extra,
            }
            for i, entry in enumerate(self.leaderboard_)
        ]

    # Get total AutoML run duration in seconds.
    def get_run_duration_seconds(self) -> Optional[float]:
        if self._start_time is None:
            return None
        end = self._end_time or datetime.now()
        return (end - self._start_time).total_seconds()

    def set_tracker(self, tracker: Any) -> None:
        # Attach an ExperimentTracker after initialization.
        self.tracker = tracker

    # Time Utilities

    def _start_timer(self) -> None:
        # Start the run timer.
        self._start_time = datetime.now()
        self._end_time = None

    def _end_timer(self) -> None:
        # End the run timer
        self._end_time = datetime.now()

    def _time_budget_seconds(self) -> Optional[float]:
        # Get time budget in seconds
        if self.time_budget_minutes is None:
            return None
        return self.time_budget_minutes * 60.0

    def _elapsed_seconds(self) -> float:
        # Get elapsed time since start.
        if self._start_time is None:
            return 0.0
        return (datetime.now() - self._start_time).total_seconds()

    def _remaining_seconds(self) -> Optional[float]:
        # Get remaining time budget in seconds.
        budget = self._time_budget_seconds()
        if budget is None:
            return None
        return max(0.0, budget - self._elapsed_seconds())

    def _time_budget_exceeded(self) -> bool:
        # Check if the time budget is exceeded.
        budget = self._time_budget_seconds()
        if budget is None:
            return False
        return self._elapsed_seconds() >= budget

    #  Tracker Utilities

    def _maybe_log_params(self, params: Dict[str, Any]) -> None:
        # Log params to tracker if available.
        if self.tracker is None:
            return
        try:
            self.tracker.log_params(params)
        except Exception as e:
            logger.debug(f"Tracker log_params failed: {e}")

    def _maybe_log_metrics(self, metrics: Dict[str, Any], prefix: str = "") -> None:
        # Log metrics to tracker if available
        if self.tracker is None:
            return
        try:
            clean: Dict[str, float] = {}
            for k, v in metrics.items():
                if isinstance(v, (int, float, np.floating, np.integer)):
                    clean[k] = float(v)
            self.tracker.log_metrics(clean, prefix=prefix)
        except Exception as e:
            logger.debug(f"Tracker log_metrics failed: {e}")

    # Leaderboard Utilities
    def _add_leaderboard_entry(
        self,
        model_name: str,
        framework: str,
        score: float,
        params: Dict[str, Any],
        duration_seconds: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Add an entry to the leaderboard and keep sorted.
        entry = LeaderboardEntry(
            model_name=model_name,
            framework=framework,
            score=float(score),
            params=params or {},
            duration_seconds=float(duration_seconds),
            extra=extra or {},
        )
        self.leaderboard_.append(entry)
        # Sort by score descending (best first)
        self.leaderboard_.sort(key=lambda e: e.score, reverse=True)

    def _update_best_if_improved(
        self,
        model_name: str,
        model_obj: Any,
        score: float,
        params: Dict[str, Any],
    ) -> bool:
        """
        Update best model if score improves.

        Returns:
            True if best was updated, False otherwise
        """

        if self.best_score_ is None or score > self.best_score_:
            self.best_score_ = float(score)
            self.best_model_ = model_obj
            self.best_model_name_ = model_name
            self.best_params_ = params if params is not None else {}
            logger.info(f"New best model: {model_name} with score {score:.4f}")
            return True
        return False

    # Validation Utilities

    def _validate_inputs(self, X: np.ndarray, y: np.ndarray) -> None:
        # Validate input arrays.
        if X is None or y is None:
            raise ValueError("X and y must not be None")
        if not isinstance(X, np.ndarray):
            raise TypeError(f"X must be np.ndarray, got {type(X)}")
        if not isinstance(y, np.ndarray):
            raise TypeError(f"y must be np.ndarray, got {type(y)}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have same n_samples: {X.shape[0]} vs {y.shape[0]}")

    def _check_is_fitted(self) -> None:
        # Raise error if not fitted.
        if not self.is_fitted_:
            raise RuntimeError(f"{self.__class__.__name__} is not fitted. Call fit() first.")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"task={self.task}, metric={self.metric}, "
            f"time_budget={self.time_budget_minutes}, "
            f"best_model={self.best_model_name_}, "
            f"best_score={self.best_score_})"
        )
