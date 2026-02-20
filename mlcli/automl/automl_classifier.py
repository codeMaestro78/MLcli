"""
AutoML Classifier

Main AutoML implementation for classification tasks.
Orchestrates data analysis, preprocessing, model selection,
hyperparameter tuning, and model ranking.

This module ties together all AutoML components:
- DataAnalyzer : Analyzes dataset characteristics
- PreprocessingSelector : Selects preprocessing steps
- ModelSelector : Selects candidate models
- SearchSpaceGenerator : Provides hyperparameter spaces
- BaseAutoML : Abstract base class with utilities

Usage:
    from mlcli.automl import AutoMLClassifier

    automl = AutoMLClassifier(
        metric="accuracy",
        time_budget_minutes=30,
        max_models=5,
    )
    automl.fit(X_train, y_train)

    predictions = automl.predict(X_test)
    print(automl.get_leaderboard())
"""

from typing import Any, Dict, List, Optional
import time
import warnings

import numpy as np
import logging

from mlcli.automl.base_automl import BaseAutoML
from mlcli.automl.data_analyzer import DataAnalyzer, DataAnalysisReport
from mlcli.automl.model_selector import ModelSelector, ModelCandidate
from mlcli.automl.preprocessing_selector import PreprocessingSelector, PreprocessingPlan
from mlcli.automl.search_space import SearchSpaceGenerator
from mlcli.preprocessor.pipeline import PreprocessingPipeline
from mlcli.tuner.tuner_factory import TunerFactory
from mlcli.trainers import get_trainer_class

logger = logging.getLogger(__name__)


# Default configuration constants
DEFAULT_CV_FOLDS = 5
DEFAULT_N_TRIALS = 20  # Per model, not total
MIN_TRIALS_PER_MODEL = 5
MAX_TRIALS_PER_MODEL = 50


class AutoMLClassifier(BaseAutoML):
    """
    Automated Machine Learning classifier.

    Automatically:
    - Analyzes data characteristics
    - Selects appropriate preprocessing
    - Chooses candidate models based on task and data size
    - Tunes hyperparameters using Bayesian optimization
    - Ranks models by cross-validation score
    - Stores best model for prediction

    Inherits from BaseAutoML which provides:
    - Leaderboard management
    - Time budget tracking
    - Best model tracking
    - Experiment tracker integration

    Attributes:
        metric (str): Scoring metric for optimization
        time_budget_minutes (float): Total time budget in minutes
        max_models (int): Maximum number of models to try
        n_trials_per_model (int): Number of tuning trials per model
        cv (int): Cross-validation folds
        preprocessing_pipeline_ (PreprocessingPipeline): Fitted pipeline
        data_report_ (DataAnalysisReport): Analysis report
    """

    def __init__(
        self,
        metric: str = "accuracy",
        time_budget_minutes: Optional[float] = None,
        max_models: int = 5,
        n_trials_per_model: int = DEFAULT_N_TRIALS,
        cv: int = DEFAULT_CV_FOLDS,
        include_models: Optional[List[str]] = None,
        exclude_models: Optional[List[str]] = None,
        random_state: Optional[int] = 42,
        verbose: bool = True,
        auto_preprocess: bool = True,
        tuning_method: str = "bayesian",
        tracker: Optional[Any] = None,
    ) -> None:
        """
        Initialize AutoMLClassifier.

        Args:
            metric: Scoring metric ('accuracy', 'f1', 'roc_auc', 'precision', 'recall').
                    Used for model comparison and hyperparameter optimization.
            time_budget_minutes: Total time budget in minutes. None means no limit.
                                Time is distributed across candidate models.
            max_models: Maximum number of models to evaluate. Limited by available
                       models and time budget.
            n_trials_per_model: Number of Optuna trials per model for hyperparameter
                               tuning. More trials = better params but slower.
            cv: Number of cross-validation folds for model evaluation.
            include_models: Whitelist of models to consider. If None, all suitable
                           models are candidates. Example: ['xgboost', 'lightgbm']
            exclude_models: Blacklist of models to skip. Example: ['svm', 'catboost']
            random_state: Random seed for reproducibility.
            verbose: If True, print progress information.
            auto_preprocess: If True, automatically apply preprocessing.
                            If False, expects already-preprocessed data.
            tuning_method: Hyperparameter tuning method ('bayesian', 'random', 'grid').
                          'bayesian' (Optuna) is recommended for efficiency.
            tracker: Optional ExperimentTracker for logging runs.

        Raises:
            ValueError: If metric is not supported.

        Example:
            >>> automl = AutoMLClassifier(
            ...     metric="f1",
            ...     time_budget_minutes=10,
            ...     max_models=3,
            ... )
            >>> automl.fit(X_train, y_train)
            >>> preds = automl.predict(X_test)
        """
        # Validate metric
        valid_metrics = {"accuracy", "f1", "roc_auc", "precision", "recall"}
        if metric.lower() not in valid_metrics:
            raise ValueError(
                f"Invalid metric '{metric}'. "
                f"Supported metrics: {', '.join(sorted(valid_metrics))}"
            )

        super().__init__(
            task="classification",
            metric=metric.lower(),
            time_budget_minutes=time_budget_minutes,
            random_state=random_state,
            tracker=tracker,
            verbose=verbose,
        )

        # Model selection parameters
        self.max_models = max_models
        self.n_trials_per_model = max(
            MIN_TRIALS_PER_MODEL, min(n_trials_per_model, MAX_TRIALS_PER_MODEL)
        )
        self.cv = cv
        self.include_models = include_models
        self.exclude_models = exclude_models
        self.tuning_method = tuning_method.lower()
        self.auto_preprocess = auto_preprocess

        # State tracking (populated during fit)
        self.data_report_: Optional[DataAnalysisReport] = None
        self.preprocessing_plan_: Optional[PreprocessingPlan] = None
        self.preprocessing_pipeline_: Optional[PreprocessingPipeline] = None
        self.candidates_: List[ModelCandidate] = []
        self.tuning_results_: Dict[str, Dict[str, Any]] = {}

        # Store fitted trainers for prediction
        self._fitted_trainers: Dict[str, Any] = {}

        logger.info(
            f"AutoMLClassifier initialized: metric={self.metric}, "
            f"time_budget={self.time_budget_minutes}min, "
            f"max_models={self.max_models}, trials_per_model={self.n_trials_per_model}"
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        **kwargs,
    ) -> "AutoMLClassifier":
        """
        Run AutoML pipeline: analyze, preprocess, select models, tune, rank.

        This is the main entry point for AutoML. It performs:
        1. Input validation
        2. Data analysis
        3. Preprocessing selection and application
        4. Model candidate selection
        5. Hyperparameter tuning for each candidate
        6. Leaderboard ranking
        7. Best model selection

        Args:
            X: Feature matrix (n_samples, n_features). Must be numeric after
               preprocessing. Can contain NaN if imputed externally.
            y: Target vector (n_samples,). Must be integer class labels.
            feature_names: Optional feature names. If None, auto-generated.
            **kwargs: Additional arguments (reserved for future use).

        Returns:
            self: Fitted AutoMLClassifier instance.

        Raises:
            ValueError: If X and y have incompatible shapes.
            RuntimeError: If no models could be trained successfully.

        Example:
            >>> automl = AutoMLClassifier(time_budget_minutes=5)
            >>> automl.fit(X_train, y_train)
            >>> print(f"Best model: {automl.best_model_name_}")
            >>> print(f"Best score: {automl.best_score_:.4f}")
        """
        # === 1. Validate inputs ===
        self._validate_inputs(X, y)
        self._start_timer()

        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        if self.verbose:
            logger.info(f"\n{'='*60}")
            logger.info("AutoML Classification Started")
            logger.info(f"{'='*60}")
            logger.info(f"Samples: {n_samples}, Features: {n_features}, Classes: {n_classes}")
            if self.time_budget_minutes:
                logger.info(f"Time budget: {self.time_budget_minutes} minutes")

        # === 2. Analyze data ===
        if self.verbose:
            logger.info("\n[Step 1/5] Analyzing data...")

        analyzer = DataAnalyzer()
        self.data_report_ = analyzer.analyze(X, y, feature_names)

        if self.verbose:
            logger.info(f"  - Numeric columns: {len(self.data_report_.numeric_columns)}")
            logger.info(f"  - Categorical columns: {len(self.data_report_.categorical_columns)}")
            logger.info(f"  - Imbalanced: {self.data_report_.is_imbalanced}")
            if self.data_report_.warnings:
                for warn in self.data_report_.warnings:
                    logger.warning(f"  ⚠ {warn}")

        # Log to tracker
        self._maybe_log_params(
            {
                "n_samples": n_samples,
                "n_features": n_features,
                "n_classes": n_classes,
                "n_numeric": len(self.data_report_.numeric_columns),
                "n_categorical": len(self.data_report_.categorical_columns),
                "is_imbalanced": self.data_report_.is_imbalanced,
            }
        )

        # === 3. Preprocessing ===
        X_processed = X

        if self.auto_preprocess:
            if self.verbose:
                logger.info("\n[Step 2/5] Selecting preprocessing...")

            X_processed = self._apply_preprocessing(X, y)

            if self.verbose and self.preprocessing_plan_:
                logger.info(f"  - Scaler: {self.preprocessing_plan_.scaler_method}")
                logger.info(f"  - Steps: {len(self.preprocessing_plan_.recommendations)}")
        else:
            if self.verbose:
                logger.info("\n[Step 2/5] Preprocessing skipped (auto_preprocess=False)")

        # Check for remaining issues
        if np.isnan(X_processed).any():
            logger.warning(
                "Data contains NaN values after preprocessing. "
                "Some models may fail. Consider imputation."
            )

        # === 4. Select candidate models ===
        if self.verbose:
            logger.info("\n[Step 3/5] Selecting candidate models...")

        self.candidates_ = self._select_candidates(n_samples, n_features)

        if len(self.candidates_) == 0:
            raise RuntimeError(
                "No candidate models available. "
                "Check include_models/exclude_models settings and data size."
            )

        if self.verbose:
            logger.info(f"  - Selected {len(self.candidates_)} models:")
            for c in self.candidates_:
                logger.info(f"    • {c.name} ({c.framework}, priority={c.priority})")

        # === 5. Tune each model ===
        if self.verbose:
            logger.info("\n[Step 4/5] Tuning hyperparameters...")

        self._tune_candidates(X_processed, y)

        # === 6. Finalize ===
        self._end_timer()
        self.is_fitted_ = True

        if self.verbose:
            logger.info("\n[Step 5/5] AutoML Complete!")
            logger.info(f"{'='*60}")
            logger.info(f"Best model: {self.best_model_name_}")
            logger.info(f"Best score ({self.metric}): {self.best_score_:.4f}")
            logger.info(f"Total time: {self.get_run_duration_seconds():.1f}s")
            logger.info(f"Models evaluated: {len(self.leaderboard_)}")
            logger.info(f"{'='*60}\n")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels using the best model.

        Applies the same preprocessing pipeline fitted during training,
        then uses the best model's predict method.

        Args:
            X: Feature matrix (n_samples, n_features). Must have same
               structure as training data.

        Returns:
            Predicted class labels (n_samples,).

        Raises:
            RuntimeError: If fit() has not been called.
            ValueError: If X has wrong number of features.

        Example:
            >>> predictions = automl.predict(X_test)
            >>> print(f"Predictions: {predictions[:5]}")
        """
        self._check_is_fitted()

        # Apply preprocessing
        X_processed = self._transform_for_prediction(X)

        # Get best trainer
        if self.best_model_name_ not in self._fitted_trainers:
            raise RuntimeError(
                f"Best model '{self.best_model_name_}' not found in fitted trainers. "
                "This is an internal error - please report."
            )

        trainer = self._fitted_trainers[self.best_model_name_]
        return trainer.predict(X_processed)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        Predict class probabilities using the best model.

        Not all models support probability prediction. If the best model
        doesn't support it, returns None.

        Args:
            X: Feature matrix (n_samples, n_features).

        Returns:
            Probability array (n_samples, n_classes) or None if not supported.

        Raises:
            RuntimeError: If fit() has not been called.
        """
        self._check_is_fitted()

        X_processed = self._transform_for_prediction(X)

        if self.best_model_name_ not in self._fitted_trainers:
            return None

        trainer = self._fitted_trainers[self.best_model_name_]

        try:
            return trainer.predict_proba(X_processed)
        except (AttributeError, NotImplementedError):
            logger.warning(f"Model '{self.best_model_name_}' does not support predict_proba.")
            return None

    def get_best_model(self) -> Any:
        """
        Get the best fitted model/trainer.

        Returns the trainer object for the best model, allowing
        direct access to the underlying model.

        Returns:
            Best trainer object.

        Raises:
            RuntimeError: If fit() has not been called.

        Example:
            >>> best_trainer = automl.get_best_model()
            >>> print(type(best_trainer))  # e.g., XGBTrainer
            >>> model = best_trainer.get_model()  # underlying sklearn/xgb model
        """
        self._check_is_fitted()

        if self.best_model_ is None:
            raise RuntimeError("No best model available. Training may have failed.")

        return self.best_model_

    def get_fitted_trainer(self, model_name: str) -> Optional[Any]:
        """
        Get a specific fitted trainer by name.

        Useful for accessing models other than the best one.

        Args:
            model_name: Name of the model (e.g., 'xgboost', 'random_forest')

        Returns:
            Trainer object or None if not found.
        """
        return self._fitted_trainers.get(model_name)

    def get_tuning_results(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get hyperparameter tuning results.

        Args:
            model_name: Specific model name, or None for all results.

        Returns:
            Dictionary with tuning results.
        """
        if model_name:
            return self.tuning_results_.get(model_name, {})
        return self.tuning_results_.copy()

    def get_data_report(self) -> Optional[DataAnalysisReport]:
        """Get the data analysis report from training."""
        return self.data_report_

    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Get a summary of the preprocessing plan."""
        if self.preprocessing_plan_ is None:
            return {}

        selector = PreprocessingSelector()
        selector._plan = self.preprocessing_plan_
        return selector.get_summary()

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _apply_preprocessing(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """
        Apply automatic preprocessing based on data analysis.

        Creates a preprocessing plan using PreprocessingSelector and
        builds/fits a preprocessing pipeline.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            Preprocessed feature matrix
        """
        if self.data_report_ is None:
            raise RuntimeError("Data analysis required before preprocessing.")

        selector = PreprocessingSelector()
        self.preprocessing_plan_ = selector.select(self.data_report_)

        # Build pipeline
        self.preprocessing_pipeline_ = selector.build_pipeline(self.preprocessing_plan_)

        # Fit and transform
        try:
            X_transformed = self.preprocessing_pipeline_.fit_transform(X, y)
            logger.info(f"Preprocessing complete: {X.shape} -> {X_transformed.shape}")
            return X_transformed
        except Exception as e:
            logger.warning(f"Preprocessing failed: {e}. Using raw data.")
            self.preprocessing_pipeline_ = None
            return X

    def _transform_for_prediction(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the fitted preprocessing pipeline for prediction.

        Args:
            X: Raw feature matrix

        Returns:
            Preprocessed feature matrix
        """
        if self.preprocessing_pipeline_ is not None:
            try:
                return self.preprocessing_pipeline_.transform(X)
            except Exception as e:
                logger.warning(f"Preprocessing transform failed: {e}. Using raw data.")

        return X

    def _select_candidates(
        self,
        n_samples: int,
        n_features: int,
    ) -> List[ModelCandidate]:
        """
        Select candidate models based on task, data size, and user preferences.

        Uses ModelSelector from Step 3 to filter and rank models.

        Args:
            n_samples: Number of samples
            n_features: Number of features

        Returns:
            List of ModelCandidate objects, sorted by priority
        """
        selector = ModelSelector(
            task=self.task,
            include_models=self.include_models,
            exclude_models=self.exclude_models,
            fast_mode=(self.time_budget_minutes is not None and self.time_budget_minutes < 5),
        )

        # Calculate time budget per model (if applicable)
        time_budget_minutes = None
        if self.time_budget_minutes:
            # Reserve 10% for overhead
            usable_time = self.time_budget_minutes * 0.9
            time_budget_minutes = usable_time / max(1, self.max_models)

        candidates = selector.select_models(
            n_samples=n_samples,
            n_features=n_features,
            time_budget_minutes=int(time_budget_minutes) if time_budget_minutes else None,
            max_models=self.max_models,
        )

        return candidates

    def _tune_candidates(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """
        Tune hyperparameters for each candidate model.

        For each candidate:
        1. Generate search space (Step 2)
        2. Create tuner (existing infrastructure)
        3. Run hyperparameter optimization
        4. Train final model with best params
        5. Update leaderboard

        Args:
            X: Preprocessed feature matrix
            y: Target vector
        """
        n_samples, n_features = X.shape
        space_gen = SearchSpaceGenerator()

        # Calculate time budget per model
        time_per_model_seconds = None
        if self.time_budget_minutes:
            total_seconds = self.time_budget_minutes * 60
            # Reserve 20% for final training and overhead
            tuning_seconds = total_seconds * 0.8
            time_per_model_seconds = tuning_seconds / max(1, len(self.candidates_))

        for idx, candidate in enumerate(self.candidates_):
            # Check time budget
            if self._time_budget_exceeded():
                logger.warning(f"Time budget exceeded. Stopping after {idx} models.")
                break

            if self.verbose:
                remaining = self._remaining_seconds()
                remaining_str = f"{remaining:.0f}s" if remaining else "unlimited"
                logger.info(
                    f"\n  [{idx+1}/{len(self.candidates_)}] Tuning {candidate.name} "
                    f"(time remaining: {remaining_str})"
                )

            try:
                result = self._tune_single_model(
                    candidate=candidate,
                    X=X,
                    y=y,
                    n_samples=n_samples,
                    n_features=n_features,
                    space_gen=space_gen,
                    timeout_seconds=time_per_model_seconds,
                )

                if result is not None:
                    self.tuning_results_[candidate.name] = result

            except Exception as e:
                logger.error(f"Failed to tune {candidate.name}: {e}")
                self.tuning_results_[candidate.name] = {
                    "status": "failed",
                    "error": str(e),
                }
                continue

    def _tune_single_model(
        self,
        candidate: ModelCandidate,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int,
        n_features: int,
        space_gen: SearchSpaceGenerator,
        timeout_seconds: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Tune a single model candidate.

        Steps:
        1. Get trainer class
        2. Generate search space
        3. Create tuner
        4. Run optimization
        5. Train final model with best params
        6. Add to leaderboard

        Args:
            candidate: ModelCandidate to tune
            X: Feature matrix
            y: Target vector
            n_samples: Number of samples
            n_features: Number of features
            space_gen: SearchSpaceGenerator instance
            timeout_seconds: Optional timeout for this model

        Returns:
            Dictionary with tuning results, or None on failure
        """
        start_time = time.time()

        # === 1. Get trainer class ===
        try:
            trainer_class = get_trainer_class(candidate.name)
        except (ValueError, ImportError) as e:
            logger.warning(f"Could not load trainer for {candidate.name}: {e}")
            return None

        # === 2. Get search space ===
        search_space = space_gen.get_space(
            model_name=candidate.name,
            n_samples=n_samples,
            n_features=n_features,
        )

        if not search_space:
            logger.warning(f"No search space defined for {candidate.name}. Using defaults.")
            # Fallback: train with default params
            return self._train_with_defaults(candidate, trainer_class, X, y, start_time)

        # === 3. Create tuner ===
        # Calculate timeout for this model
        timeout = None
        if timeout_seconds:
            timeout = int(timeout_seconds)

        try:
            tuner = TunerFactory.create(
                method=self.tuning_method,
                param_space=search_space,
                n_trials=self.n_trials_per_model,
                scoring=self.metric,
                cv=self.cv,
                random_state=self.random_state,
                verbose=1 if self.verbose else 0,
                timeout=timeout,
            )
        except ImportError as e:
            logger.warning(f"Tuner creation failed for {candidate.name}: {e}")
            return self._train_with_defaults(candidate, trainer_class, X, y, start_time)

        # === 4. Run optimization ===
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tune_result = tuner.tune(
                    trainer_class=trainer_class,
                    X=X,
                    y=y,
                    trainer_config={},
                )
        except Exception as e:
            logger.warning(f"Tuning failed for {candidate.name}: {e}")
            return self._train_with_defaults(candidate, trainer_class, X, y, start_time)

        best_params = tune_result.get("best_params", {})
        best_score = tune_result.get("best_score", 0.0)

        if self.verbose:
            logger.info(f"    Best score: {best_score:.4f}")
            logger.info(f"    Best params: {best_params}")

        # === 5. Train final model with best params ===
        trainer = trainer_class(config={"params": best_params})

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                trainer.train(X, y)
        except Exception as e:
            logger.error(f"Final training failed for {candidate.name}: {e}")
            return {
                "status": "tuned_but_training_failed",
                "best_params": best_params,
                "best_score": best_score,
                "error": str(e),
            }

        # Store fitted trainer
        self._fitted_trainers[candidate.name] = trainer

        # === 6. Update leaderboard ===
        duration = time.time() - start_time

        self._add_leaderboard_entry(
            model_name=candidate.name,
            framework=candidate.framework,
            score=best_score,
            params=best_params,
            duration_seconds=duration,
            extra={
                "n_trials": tune_result.get("cv_results", {}).get(
                    "n_trials", self.n_trials_per_model
                ),
                "tuning_method": self.tuning_method,
            },
        )

        # Update best model if improved
        self._update_best_if_improved(
            model_name=candidate.name,
            model_obj=trainer,
            score=best_score,
            params=best_params,
        )

        # Log to tracker
        self._maybe_log_metrics({f"{candidate.name}_{self.metric}": best_score}, prefix="model_")

        return {
            "status": "success",
            "best_params": best_params,
            "best_score": best_score,
            "duration_seconds": duration,
            "n_trials": tune_result.get("cv_results", {}).get("n_trials", self.n_trials_per_model),
        }

    def _train_with_defaults(
        self,
        candidate: ModelCandidate,
        trainer_class: type,
        X: np.ndarray,
        y: np.ndarray,
        start_time: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Train a model with default hyperparameters (fallback when tuning fails).

        Args:
            candidate: Model candidate
            trainer_class: Trainer class
            X: Feature matrix
            y: Target vector
            start_time: Start timestamp

        Returns:
            Training result dictionary
        """
        try:
            # Get default params from trainer
            default_params = trainer_class.get_default_params()
        except (AttributeError, NotImplementedError):
            default_params = {}

        trainer = trainer_class(config={"params": default_params})

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                history = trainer.train(X, y)

            # Get score from training history
            score = 0.0
            if "train_metrics" in history:
                score = history["train_metrics"].get(self.metric, 0.0)
            elif "val_metrics" in history:
                score = history["val_metrics"].get(self.metric, 0.0)

        except Exception as e:
            logger.warning(f"Default training failed for {candidate.name}: {e}")
            return None

        # Store trainer
        self._fitted_trainers[candidate.name] = trainer

        duration = time.time() - start_time

        # Update leaderboard
        self._add_leaderboard_entry(
            model_name=candidate.name,
            framework=candidate.framework,
            score=score,
            params=default_params,
            duration_seconds=duration,
            extra={"tuning": "default_params"},
        )

        self._update_best_if_improved(
            model_name=candidate.name,
            model_obj=trainer,
            score=score,
            params=default_params,
        )

        if self.verbose:
            logger.info(f"    Trained with defaults. Score: {score:.4f}")

        return {
            "status": "default_params",
            "best_params": default_params,
            "best_score": score,
            "duration_seconds": duration,
        }

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a JSON-serializable summary of the AutoML run.

        Returns:
            Dictionary with run summary including:
            - Configuration
            - Data report summary
            - Preprocessing summary
            - Leaderboard
            - Best model info
            - Timing
        """
        return {
            "config": {
                "metric": self.metric,
                "time_budget_minutes": self.time_budget_minutes,
                "max_models": self.max_models,
                "n_trials_per_model": self.n_trials_per_model,
                "cv": self.cv,
                "tuning_method": self.tuning_method,
                "auto_preprocess": self.auto_preprocess,
            },
            "data": {
                "n_samples": self.data_report_.n_samples if self.data_report_ else None,
                "n_features": self.data_report_.n_features if self.data_report_ else None,
                "n_numeric": len(self.data_report_.numeric_columns) if self.data_report_ else None,
                "n_categorical": (
                    len(self.data_report_.categorical_columns) if self.data_report_ else None
                ),
                "is_imbalanced": self.data_report_.is_imbalanced if self.data_report_ else None,
            },
            "preprocessing": self.get_preprocessing_summary(),
            "models_evaluated": len(self.leaderboard_),
            "leaderboard": self.get_leaderboard(),
            "best_model": {
                "name": self.best_model_name_,
                "score": self.best_score_,
                "params": self.best_params_,
            },
            "duration_seconds": self.get_run_duration_seconds(),
            "is_fitted": self.is_fitted_,
        }
