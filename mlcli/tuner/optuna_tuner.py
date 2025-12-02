"""
Optuna-based Bayesian Optimization Tuner

Uses Optuna for efficient hyperparameter search with Tree-structured 
Parzen Estimator (TPE) algorithm.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
import logging
from sklearn.model_selection import StratifiedKFold
import warnings

from mlcli.tuner.base_tuner import BaseTuner

logger = logging.getLogger(__name__)


class OptunaTuner(BaseTuner):
    """
    Optuna-based Bayesian Optimization tuner.
    
    Uses Tree-structured Parzen Estimator (TPE) for intelligent 
    hyperparameter search. Most efficient for large search spaces.
    """

    def __init__(
        self,
        param_space: Dict[str, Any],
        n_trials: int = 100,
        scoring: str = "accuracy",
        cv: int = 5,
        n_jobs: int = 1,
        verbose: int = 1,
        random_state: Optional[int] = 42,
        timeout: Optional[int] = None,
        pruning: bool = True
    ):
        """
        Initialize Optuna tuner.
        
        Args:
            param_space: Dictionary mapping parameter names to search specs
                        Example: {
                            "C": {"type": "loguniform", "low": 1e-4, "high": 100},
                            "kernel": {"type": "categorical", "choices": ["rbf", "linear"]},
                            "gamma": {"type": "float", "low": 0.001, "high": 1.0}
                        }
            n_trials: Number of trials to run
            scoring: Metric to optimize
            cv: Number of cross-validation folds
            n_jobs: Number of parallel trials (1 for sequential)
            verbose: Verbosity level
            random_state: Random seed
            timeout: Maximum time in seconds (None for no limit)
            pruning: Whether to use early stopping for unpromising trials
        """
        super().__init__(param_space, scoring, cv, n_jobs, verbose, random_state)
        
        self.n_trials = n_trials
        self.timeout = timeout
        self.pruning = pruning
        self.study_ = None
        
        # Check if optuna is available
        try:
            import optuna
            self.optuna = optuna
            
            # Set optuna verbosity
            if verbose < 2:
                optuna.logging.set_verbosity(optuna.logging.WARNING)
            
        except ImportError:
            raise ImportError(
                "Optuna is required for Bayesian optimization. "
                "Install it with: pip install optuna"
            )
        
        logger.info(f"Optuna tuner initialized with {n_trials} trials")

    def tune(
        self,
        trainer_class: type,
        X: np.ndarray,
        y: np.ndarray,
        trainer_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform Bayesian optimization hyperparameter tuning.
        
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
            logger.info(f"Starting Optuna Bayesian Optimization with {self.n_trials} trials")
            logger.info(f"Scoring: {self.scoring}, CV: {self.cv}-fold")
        
        # Cross-validation splitter
        cv_splitter = StratifiedKFold(
            n_splits=self.cv,
            shuffle=True,
            random_state=self.random_state
        )
        
        # Store references for objective function
        self._trainer_class = trainer_class
        self._trainer_config = trainer_config
        self._X = X
        self._y = y
        self._cv_splitter = cv_splitter
        self._trial_count = 0
        
        # Create Optuna study
        sampler = self.optuna.samplers.TPESampler(seed=self.random_state)
        
        self.study_ = self.optuna.create_study(
            direction="maximize",
            sampler=sampler,
            study_name="mlcli_hyperparameter_tuning"
        )
        
        # Run optimization
        self.study_.optimize(
            self._objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
            show_progress_bar=(self.verbose >= 1)
        )
        
        self.end_time_ = datetime.now()
        
        # Store results
        self.best_params_ = self.study_.best_params
        self.best_score_ = self.study_.best_value
        
        self.cv_results_ = {
            "all_results": [
                {
                    "params": trial.params,
                    "score": trial.value if trial.value is not None else 0,
                    "trial": trial.number,
                    "state": str(trial.state)
                }
                for trial in self.study_.trials
            ],
            "n_trials": len(self.study_.trials),
            "n_completed": len([t for t in self.study_.trials 
                               if t.state == self.optuna.trial.TrialState.COMPLETE]),
            "n_pruned": len([t for t in self.study_.trials 
                           if t.state == self.optuna.trial.TrialState.PRUNED])
        }
        
        if self.verbose >= 1:
            duration = self.get_tuning_duration()
            logger.info(f"\nOptuna Optimization Complete!")
            logger.info(f"Best Score: {self.best_score_:.4f}")
            logger.info(f"Best Params: {self.best_params_}")
            logger.info(f"Total Duration: {duration:.1f}s")
            logger.info(f"Trials: {self.cv_results_['n_completed']} completed, "
                       f"{self.cv_results_['n_pruned']} pruned")
        
        return {
            "best_params": self.best_params_,
            "best_score": self.best_score_,
            "cv_results": self.cv_results_,
            "duration": self.get_tuning_duration()
        }

    def _objective(self, trial) -> float:
        """
        Optuna objective function.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Score to maximize
        """
        self._trial_count += 1
        trial_start = datetime.now()
        
        # Sample parameters
        params = self._suggest_params(trial)
        
        if self.verbose >= 2:
            logger.info(f"Trial {self._trial_count}: {params}")
        
        try:
            # Create trainer config
            config = {**self._trainer_config, "params": params}
            
            # Cross-validation with optional pruning
            scores = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(
                self._cv_splitter.split(self._X, self._y)
            ):
                X_train = self._X[train_idx]
                X_val = self._X[val_idx]
                y_train = self._y[train_idx]
                y_val = self._y[val_idx]
                
                # Train
                trainer = self._trainer_class(config=config)
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    trainer.train(X_train, y_train, X_val, y_val)
                
                # Evaluate
                metrics = trainer.evaluate(X_val, y_val)
                score = self._get_score_from_metrics(metrics)
                scores.append(score)
                
                # Pruning: report intermediate value
                if self.pruning:
                    intermediate_score = np.mean(scores)
                    trial.report(intermediate_score, fold_idx)
                    
                    if trial.should_prune():
                        raise self.optuna.TrialPruned()
            
            mean_score = np.mean(scores)
            
            # Log trial
            trial_duration = (datetime.now() - trial_start).total_seconds()
            self._log_trial(self._trial_count, params, mean_score, trial_duration)
            
            return mean_score
            
        except self.optuna.TrialPruned:
            raise
        except Exception as e:
            logger.warning(f"Trial {self._trial_count} failed: {e}")
            return 0.0

    def _suggest_params(self, trial) -> Dict[str, Any]:
        """
        Suggest parameters using Optuna trial.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of suggested parameters
        """
        params = {}
        
        for param_name, param_config in self.param_space.items():
            params[param_name] = self._suggest_single_param(
                trial, param_name, param_config
            )
        
        return params

    def _suggest_single_param(
        self,
        trial,
        param_name: str,
        param_config: Any
    ) -> Any:
        """
        Suggest a single parameter value.
        
        Args:
            trial: Optuna trial object
            param_name: Parameter name
            param_config: Parameter configuration
            
        Returns:
            Suggested value
        """
        # List/tuple: categorical choice
        if isinstance(param_config, (list, tuple)):
            return trial.suggest_categorical(param_name, param_config)
        
        # Dictionary: distribution specification
        if isinstance(param_config, dict):
            dist_type = param_config.get("type", "float")
            
            if dist_type in ["categorical", "choice"]:
                choices = param_config.get("choices", param_config.get("values", []))
                return trial.suggest_categorical(param_name, choices)
            
            elif dist_type == "int":
                low = param_config.get("low", 1)
                high = param_config.get("high", 100)
                step = param_config.get("step", 1)
                return trial.suggest_int(param_name, low, high, step=step)
            
            elif dist_type == "int_uniform":
                low = param_config.get("low", 1)
                high = param_config.get("high", 100)
                return trial.suggest_int(param_name, low, high)
            
            elif dist_type == "float":
                low = param_config.get("low", 0.0)
                high = param_config.get("high", 1.0)
                step = param_config.get("step")
                return trial.suggest_float(param_name, low, high, step=step)
            
            elif dist_type == "uniform":
                low = param_config.get("low", 0.0)
                high = param_config.get("high", 1.0)
                return trial.suggest_float(param_name, low, high)
            
            elif dist_type == "loguniform":
                low = param_config.get("low", 1e-5)
                high = param_config.get("high", 1.0)
                return trial.suggest_float(param_name, low, high, log=True)
            
            else:
                raise ValueError(f"Unknown distribution type: {dist_type}")
        
        # Single value: return as constant
        return param_config

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

    def get_importance(self) -> Dict[str, float]:
        """
        Get hyperparameter importance scores.
        
        Returns:
            Dictionary mapping parameter names to importance scores
        """
        if self.study_ is None:
            raise RuntimeError("No tuning performed yet. Call tune() first.")
        
        try:
            importance = self.optuna.importance.get_param_importances(self.study_)
            return dict(importance)
        except Exception as e:
            logger.warning(f"Could not compute parameter importance: {e}")
            return {}

    def plot_optimization_history(self, save_path: Optional[str] = None):
        """
        Plot optimization history.
        
        Args:
            save_path: Optional path to save the plot
        """
        if self.study_ is None:
            raise RuntimeError("No tuning performed yet. Call tune() first.")
        
        try:
            fig = self.optuna.visualization.plot_optimization_history(self.study_)
            if save_path:
                fig.write_image(save_path)
            return fig
        except Exception as e:
            logger.warning(f"Could not create optimization history plot: {e}")
            return None

    def plot_param_importances(self, save_path: Optional[str] = None):
        """
        Plot parameter importances.
        
        Args:
            save_path: Optional path to save the plot
        """
        if self.study_ is None:
            raise RuntimeError("No tuning performed yet. Call tune() first.")
        
        try:
            fig = self.optuna.visualization.plot_param_importances(self.study_)
            if save_path:
                fig.write_image(save_path)
            return fig
        except Exception as e:
            logger.warning(f"Could not create param importance plot: {e}")
            return None
