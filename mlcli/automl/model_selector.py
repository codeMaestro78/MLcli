"""
Model Selector

Selects candidate models for AutoML based on task type, data characteristics,
and user preferences. Filters and ranks models from the registry.
"""

from typing import Dict, Optional, List, Set
from dataclasses import dataclass
import copy
import logging
logger = logging.getLogger(__name__)


@dataclass
class ModelCandidate:
    # Represents a candidate model for AutoML.
    name: str
    framework: str
    model_type: str
    priority: int  # lower = higher priority
    estimated_time: int
    description: str = ""


# Default model configurations for different scenarios
CLASSIFICATION_MODELS: List[str] = [
    "logistic_regression",
    "random_forest",
    "xgboost",
    "lightgbm",
    "catboost",
    "svm",
]

REGRESSION_MODELS: List[str] = [
    "random_forest",
    "xgboost",
    "lightgbm",
    "catboost",
]

# Models to skip for the large datasets (>100k)
SLOW_MODELS: Set[str] = {"svm", "catboost"}

# Models to skip for small datasets (<1000)
COMPLEX_MODELS: Set[str] = {"catboost", "lightgbm"}

# Fast models for quick experiments
FAST_MODELS: List[str] = ["logistic_regression", "random_forest"]

# Model priority rankings (lower = (try first)))
MODEL_PRIORITY = Dict[str, int] = {
    "lightgbm": 1,
    "xgboost": 2,
    "random_forest": 3,
    "catboost": 4,
    "logistic_regression": 5,
    "svm": 6,
}

# Estimated relative training time (1-10 scale)
MODEL_TIME_ESTIMATE: Dict[str, int] = {
    "logistic_regression": 1,
    "random_forest": 3,
    "xgboost": 5,
    "lightgbm": 4,
    "catboost": 6,
    "svm": 7,
}


class ModelSelector:
    """
    Selects and ranks models for AutoML pipeline.

    Considers:
    - Task type (classification/regression)
    - Data size and shape
    - Time budget
    - User preferences (whitelist/blacklist)
    """

    def __init__(
        self,
        task: str = "classification",
        include_models: Optional[List[str]] = None,
        exclude_models: Optional[List[str]] = None,
        fast_mode: bool = False,
    ):
        """
        Initialize model selector.

        Args:
            task: 'classification' or 'regression'
            include_models: Whitelist of models to consider (None = all)
            exclude_models: Blacklist of models to skip
            fast_mode: If True, only use fast models
        """

        self.task = task.lower()
        self.include_models = set(include_models) if include_models else None
        self.exclude_models = set(exclude_models) if exclude_models else set()
        self.fast_mode = fast_mode

        # Get base model list for task

        if self.task == "classification":
            self._base_models = copy.deepcopy(CLASSIFICATION_MODELS)
        elif self.task == "regression":
            self._base_models = copy.deepcopy(REGRESSION_MODELS)
        else:
            raise ValueError(f"Unknown task :{task}. Use 'classification' or 'regression'")

        logger.debug(
            f"ModelSelector initialized for {task} with {len(self._base_models)} base models"
        )

    def select_models(
        self,
        n_samples: Optional[int] = None,
        n_features: Optional[int] = None,
        time_budget_minutes: Optional[int] = None,
        max_models: Optional[int] = None,
    ) -> List[ModelCandidate]:
        """
        Select candidate models based on data characteristics.

        Args:
            n_samples: Number of samples in dataset
            n_features: Number of features
            time_budget_minutes: Total time budget in minutes
            max_models: Maximum number of models to return

            Returns:
                List of ModelCandidate objects, sorted by priority
        """

        candidates: List[ModelCandidate] = []

        for model_name in self._base_models:
            # check whitelist
            if self.include_models and model_name not in self.include_models:
                continue

            # check blacklist
            if model_name in self.exclude_models:
                continue

            # Fast mode filter
            if self.fast_mode and model_name not in FAST_MODELS:
                continue

            # Data size filters
            if n_samples is not None:
                # skip slow models for the large datasets
                if n_samples > 100000 and model_name in SLOW_MODELS:
                    logger.debug(f"Skipping {model_name}: too slow for {n_samples} samples")
                    continue

                # skip complex models for the small datasets
                if n_samples < 1000 and model_name in COMPLEX_MODELS:
                    logger.debug(f"Skipping {model_name}: too complex for {n_samples} samples")
                    continue

            # time budget filter
            if time_budget_minutes is not None and time_budget_minutes < 10:
                if MODEL_TIME_ESTIMATE.get(model_name, 5) > 4:
                    logger.debug(f"Skipping {model_name}: insufficient time bydget")
                    continue

            # create candidate
            candidate = ModelCandidate(
                name=model_name,
                framework=self._get_framework(model_name),
                model_type=self.task,
                priority=MODEL_PRIORITY.get(model_name, 10),
                estimated_time=MODEL_TIME_ESTIMATE.get(model_name, 5),
                description=self._get_description(model_name),
            )
            candidates.append(candidate)

            # sort by priority
            candidates.sort(key=lambda x: x.priority)

            # limit number of models
            if max_models is not None and len(candidates) > max_models:
                candidates = candidates[:max_models]

            logger.info(
                f"Selected {len(candidates)} candidates models: {[c.name for c in candidates]}"
            )
        return candidates

    def _get_description(self, model_name: str) -> str:
        # get description for a model

        description = {
            "logistic_regression": "Linear classifier with regularization",
            "svm": "Support Vector Machine with kernel trick",
            "random_forest": "Ensemble of decision trees with bagging",
            "xgboost": "Gradient boosting with regularization",
            "lightgbm": "Fast gradient boosting with leaf-wise growth",
            "catboost": "Gradient boosting with categorical feature support",
        }
        return description.get(model_name, "")

    def get_model_names(
        self,
        n_samples: Optional[int] = None,
        n_features: Optional[int] = None,
        time_budget_minutes: Optional[int] = None,
        max_models: Optional[int] = None,
    ) -> List[str]:
        """
        Get list of selected model names.

        Convenience method that returns just the names.
        """

        candidates = self.select_models(
            n_samples=n_samples,
            n_features=n_features,
            time_budget_minutes=time_budget_minutes,
            max_models=max_models,
        )
        return [c.name for c in candidates]

    def estimate_total_time(
        self,
        n_samples: int,
        n_features: int,
        candidates: List[ModelCandidate],
        n_trials_per_model: int = 20,
    ) -> float:
        """
        Estimate total time for AutoML run.

        Args:
            n_samples: Number of samples
            n_features: Number of features
            candidates: List of model candidates
            n_trials_per_model: Tuning trials per model

        Returns:
            Estimated time in minutes
        """

        # base time per trial (in seconds) - rough estimate
        base_time_per_trial = 2.0

        # scale by data size
        size_factor = (n_samples / 10000) * (n_features / 50)
        size_factor = max(0.1, min(size_factor, 10.0))  # clamp

        total_seconds = 0.0
        for candidate in candidates:
            model_factor = candidate.estimated_time
            trial_time = base_time_per_trial * n_trials_per_model * model_factor * size_factor
            total_seconds += trial_time

        return total_seconds / 60.0

    def allocate_time_budget(
        self,
        candidates: List[ModelCandidate],
        total_budget_minutes=float,
    ) -> Dict[str, float]:
        """
        Allocate time budget across models.

        Distributes time proportionally based on model priority.
        Higher priority models get slightly more time.

        Args:
            candidates: List of model candidates
            total_budget_minutes: Total time budget

        Returns:
            Dict mapping model name to allocated minutes
        """

        if not candidates:
            return {}

        # base allocation : equal split

        base_per_model = total_budget_minutes / len(candidates)

        # weight by inverse priority (lower priority number = more time )
        total_weight = sum(1.0 / c.priority for c in candidates)

        allocation = {}

        for candidate in candidates:
            weight = (1.0 / candidate.priority) / total_weight
            # blend equal split with weighted (70% equal , 30% weighted)

            allocated = 0.7 * base_per_model + 0.3 * (weight * total_budget_minutes)
            allocation[candidate.name] = max(allocated, 1.0)

        logger.debug(f"Time allocation: {allocation}")
        return allocation

    def __repr__(self) -> str:
        return f"ModelSelector(task={self.task}, fast_mode={self.fast_mode})"
