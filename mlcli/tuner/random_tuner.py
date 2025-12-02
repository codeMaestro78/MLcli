"""
Random Search Tuner

Random sampling from parameter distributions for efficient hyperparameter search.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import logging
from sklearn.model_selection import StratifiedKFold
import warnings
from scipy import stats

from mlcli.tuner.base_tuner import BaseTuner

logger = logging.getLogger(__name__)


class RandomSearchTuner(BaseTuner):
    """
    Random Search hyperparameter tuner.
    
    Samples parameters randomly from distributions.
    More efficient than grid search for large parameter spaces.
    """

    def __init__(
        self,
        param_space: Dict[str, Any],
        n_iter: int = 50,
        scoring: str = "accuracy",
        cv: int = 5,
        n_jobs: int = -1,
        verbose: int = 1,
        random_state: Optional[int] = 42
    ):
        """
        Initialize Random Search tuner.
        
        Args:
            param_space: Dictionary mapping parameter names to distributions/lists
                        Example: {
                            "C": [0.1, 1.0, 10.0],  # discrete choices
                            "gamma": {"type": "loguniform", "low": 1e-4, "high": 1e-1},
                            "n_estimators": {"type": "int_uniform", "low": 50, "high": 500}
                        }
            n_iter: Number of parameter combinations to try
            scoring: Metric to optimize
            cv: Number of cross-validation folds
            n_jobs: Number of parallel jobs
            verbose: Verbosity level
            random_state: Random seed
        """
        super().__init__(param_space, scoring, cv, n_jobs, verbose, random_state)
        
        self.n_iter = n_iter
        self.rng = np.random.RandomState(random_state)
        
        logger.info(f"Random Search initialized with {n_iter} iterations")

    def tune(
        self,
        trainer_class: type,
        X: np.ndarray,
        y: np.ndarray,
        trainer_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform random search hyperparameter tuning.
        
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
            logger.info(f"Starting Random Search with {self.n_iter} iterations")
            logger.info(f"Scoring: {self.scoring}, CV: {self.cv}-fold")
        
        best_score = -np.inf
        best_params = None
        all_results = []
        
        # Cross-validation splitter
        cv_splitter = StratifiedKFold(
            n_splits=self.cv,
            shuffle=True,
            random_state=self.random_state
        )
        
        # Random search iterations
        for trial_num in range(1, self.n_iter + 1):
            trial_start = datetime.now()
            
            # Sample parameters
            params = self._sample_params()
            
            if self.verbose >= 1:
                progress = f"[{trial_num}/{self.n_iter}]"
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
            "n_failed": self.n_iter - len(all_results)
        }
        
        if self.verbose >= 1:
            duration = self.get_tuning_duration()
            logger.info(f"\nRandom Search Complete!")
            logger.info(f"Best Score: {best_score:.4f}")
            logger.info(f"Best Params: {best_params}")
            logger.info(f"Total Duration: {duration:.1f}s")
        
        return {
            "best_params": self.best_params_,
            "best_score": self.best_score_,
            "cv_results": self.cv_results_,
            "duration": self.get_tuning_duration()
        }

    def _sample_params(self) -> Dict[str, Any]:
        """
        Sample a parameter combination from the search space.
        
        Returns:
            Dictionary of sampled parameters
        """
        params = {}
        
        for param_name, param_config in self.param_space.items():
            params[param_name] = self._sample_single_param(param_config)
        
        return params

    def _sample_single_param(self, param_config: Any) -> Any:
        """
        Sample a single parameter value.
        
        Args:
            param_config: Parameter configuration (list, dict, or value)
            
        Returns:
            Sampled value
        """
        # List/tuple: uniform choice
        if isinstance(param_config, (list, tuple)):
            return self.rng.choice(param_config)
        
        # Dictionary: distribution specification
        if isinstance(param_config, dict):
            dist_type = param_config.get("type", "uniform")
            
            if dist_type == "uniform":
                low = param_config.get("low", 0)
                high = param_config.get("high", 1)
                return self.rng.uniform(low, high)
            
            elif dist_type == "int_uniform":
                low = param_config.get("low", 0)
                high = param_config.get("high", 100)
                return int(self.rng.randint(low, high + 1))
            
            elif dist_type == "loguniform":
                low = param_config.get("low", 1e-5)
                high = param_config.get("high", 1)
                return float(10 ** self.rng.uniform(np.log10(low), np.log10(high)))
            
            elif dist_type == "normal":
                mean = param_config.get("mean", 0)
                std = param_config.get("std", 1)
                return self.rng.normal(mean, std)
            
            elif dist_type == "choice":
                choices = param_config.get("values", [])
                return self.rng.choice(choices)
            
            else:
                raise ValueError(f"Unknown distribution type: {dist_type}")
        
        # Single value: return as-is
        return param_config

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
        """
        scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            trainer = trainer_class(config=config)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                trainer.train(X_train, y_train, X_val, y_val)
            
            metrics = trainer.evaluate(X_val, y_val)
            score = self._get_score_from_metrics(metrics)
            scores.append(score)
        
        return np.array(scores)

    def _get_score_from_metrics(self, metrics: Dict[str, float]) -> float:
        """Extract score from metrics dictionary."""
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

    def get_convergence_plot_data(self) -> Dict[str, List]:
        """
        Get data for convergence plot.
        
        Returns:
            Dictionary with trials and best scores
        """
        if not self.tuning_history_:
            return {"trials": [], "scores": [], "best_so_far": []}
        
        trials = []
        scores = []
        best_so_far = []
        current_best = -np.inf
        
        for trial in self.tuning_history_:
            trials.append(trial["trial"])
            scores.append(trial["score"])
            current_best = max(current_best, trial["score"])
            best_so_far.append(current_best)
        
        return {
            "trials": trials,
            "scores": scores,
            "best_so_far": best_so_far
        }
