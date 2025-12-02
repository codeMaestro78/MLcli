"""
Grid Search Tuner

Exhaustive search over specified parameter grid.
"""

import numpy as np
from typing import Any, Dict, List, Optional
from itertools import product
from datetime import datetime
import logging
from sklearn.model_selection import StratifiedKFold, cross_val_score
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

from mlcli.tuner.base_tuner import BaseTuner

logger = logging.getLogger(__name__)


class GridSearchTuner(BaseTuner):
    """
    Grid Search hyperparameter tuner.
    
    Performs exhaustive search over all parameter combinations.
    Best for small parameter spaces with discrete values.
    """

    def __init__(
        self,
        param_space: Dict[str, List[Any]],
        scoring: str = "accuracy",
        cv: int = 5,
        n_jobs: int = -1,
        verbose: int = 1,
        random_state: Optional[int] = 42
    ):
        """
        Initialize Grid Search tuner.
        
        Args:
            param_space: Dictionary mapping parameter names to lists of values
                        Example: {"C": [0.1, 1.0, 10.0], "kernel": ["rbf", "linear"]}
            scoring: Metric to optimize
            cv: Number of cross-validation folds
            n_jobs: Number of parallel jobs
            verbose: Verbosity level
            random_state: Random seed
        """
        super().__init__(param_space, scoring, cv, n_jobs, verbose, random_state)
        
        # Validate param_space format
        for param, values in param_space.items():
            if not isinstance(values, (list, tuple)):
                raise ValueError(
                    f"Grid search requires list of values for each parameter. "
                    f"Got {type(values)} for '{param}'"
                )
        
        # Calculate total combinations
        self.n_combinations = 1
        for values in param_space.values():
            self.n_combinations *= len(values)
        
        logger.info(f"Grid Search initialized with {self.n_combinations} parameter combinations")

    def tune(
        self,
        trainer_class: type,
        X: np.ndarray,
        y: np.ndarray,
        trainer_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform grid search hyperparameter tuning.
        
        Args:
            trainer_class: The trainer class to tune
            X: Feature matrix
            y: Target vector
            trainer_config: Base configuration for trainer
            
        Returns:
            Dictionary containing best parameters and results
        """
        self.start_time_ = datetime.now()
        trainer_config = trainer_config or {}
        
        if self.verbose >= 1:
            logger.info(f"Starting Grid Search with {self.n_combinations} combinations")
            logger.info(f"Scoring: {self.scoring}, CV: {self.cv}-fold")
        
        # Generate all parameter combinations
        param_names = list(self.param_space.keys())
        param_values = list(self.param_space.values())
        all_combinations = list(product(*param_values))
        
        best_score = -np.inf
        best_params = None
        all_results = []
        
        # Cross-validation splitter
        cv_splitter = StratifiedKFold(
            n_splits=self.cv,
            shuffle=True,
            random_state=self.random_state
        )
        
        # Iterate through all combinations
        for trial_num, param_combo in enumerate(all_combinations, 1):
            trial_start = datetime.now()
            
            # Create parameter dictionary
            params = dict(zip(param_names, param_combo))
            
            if self.verbose >= 1:
                progress = f"[{trial_num}/{self.n_combinations}]"
                logger.info(f"{progress} Testing: {params}")
            
            try:
                # Create trainer with current params
                config = {**trainer_config, "params": params}
                
                # Perform cross-validation
                cv_scores = self._cross_validate(
                    trainer_class, config, X, y, cv_splitter
                )
                
                mean_score = np.mean(cv_scores)
                std_score = np.std(cv_scores)
                
                trial_duration = (datetime.now() - trial_start).total_seconds()
                
                # Log trial
                self._log_trial(trial_num, params, mean_score, trial_duration)
                
                # Store result
                result = {
                    "params": params,
                    "mean_score": mean_score,
                    "std_score": std_score,
                    "cv_scores": cv_scores.tolist(),
                    "trial": trial_num
                }
                all_results.append(result)
                
                # Update best
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params.copy()
                    
                    if self.verbose >= 1:
                        logger.info(f"  New best score: {mean_score:.4f} (+/- {std_score:.4f})")
                
            except Exception as e:
                logger.warning(f"Trial {trial_num} failed with params {params}: {e}")
                continue
        
        self.end_time_ = datetime.now()
        
        # Store results
        self.best_params_ = best_params
        self.best_score_ = best_score
        self.cv_results_ = {
            "all_results": all_results,
            "n_trials": len(all_results),
            "n_failed": self.n_combinations - len(all_results)
        }
        
        if self.verbose >= 1:
            duration = self.get_tuning_duration()
            logger.info(f"\nGrid Search Complete!")
            logger.info(f"Best Score: {best_score:.4f}")
            logger.info(f"Best Params: {best_params}")
            logger.info(f"Total Duration: {duration:.1f}s")
        
        return {
            "best_params": self.best_params_,
            "best_score": self.best_score_,
            "cv_results": self.cv_results_,
            "duration": self.get_tuning_duration()
        }

    def _cross_validate(
        self,
        trainer_class: type,
        config: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        cv_splitter
    ) -> np.ndarray:
        """
        Perform cross-validation for a single parameter set.
        
        Args:
            trainer_class: Trainer class
            config: Trainer configuration
            X: Features
            y: Labels
            cv_splitter: CV splitter object
            
        Returns:
            Array of CV scores
        """
        scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create and train
            trainer = trainer_class(config=config)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                trainer.train(X_train, y_train, X_val, y_val)
            
            # Evaluate
            metrics = trainer.evaluate(X_val, y_val)
            
            # Get score based on scoring metric
            score = self._get_score_from_metrics(metrics)
            scores.append(score)
        
        return np.array(scores)

    def _get_score_from_metrics(self, metrics: Dict[str, float]) -> float:
        """
        Extract the relevant score from metrics dict.
        
        Args:
            metrics: Metrics dictionary
            
        Returns:
            Score value
        """
        scoring_map = {
            "accuracy": "accuracy",
            "f1": "f1",
            "f1_score": "f1",
            "roc_auc": "roc_auc",
            "auc": "roc_auc",
            "precision": "precision",
            "recall": "recall"
        }
        
        metric_key = scoring_map.get(self.scoring, self.scoring)
        
        if metric_key in metrics:
            return metrics[metric_key]
        elif "accuracy" in metrics:
            return metrics["accuracy"]
        else:
            # Return first numeric metric
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    return value
            raise ValueError(f"Could not find score for metric '{self.scoring}'")

    def get_best_params(self) -> Dict[str, Any]:
        """Get best parameters."""
        if self.best_params_ is None:
            raise RuntimeError("No tuning performed yet. Call tune() first.")
        return self.best_params_.copy()

    def get_results(self) -> Dict[str, Any]:
        """Get all tuning results."""
        if self.cv_results_ is None:
            raise RuntimeError("No tuning performed yet. Call tune() first.")
        return {
            "best_params": self.best_params_,
            "best_score": self.best_score_,
            "cv_results": self.cv_results_,
            "tuning_history": self.tuning_history_,
            "duration": self.get_tuning_duration()
        }

    def get_param_scores_df(self):
        """
        Get results as a pandas DataFrame.
        
        Returns:
            DataFrame with all trial results
        """
        try:
            import pandas as pd
            
            if not self.tuning_history_:
                return pd.DataFrame()
            
            records = []
            for trial in self.tuning_history_:
                record = {"trial": trial["trial"], "score": trial["score"]}
                record.update(trial["params"])
                records.append(record)
            
            return pd.DataFrame(records)
        
        except ImportError:
            logger.warning("pandas not available for DataFrame export")
            return None
