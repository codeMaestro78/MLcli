"""
Feature Selection Preprocessors

SelectKBest, RFE, and VarianceThreshold for feature selection.
"""

from typing import Optional, List, Dict, Any, Callable
import numpy as np
from sklearn.feature_selection import (
    SelectKBest,
    RFE,
    VarianceThreshold,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
    chi2
)
from sklearn.linear_model import LogisticRegression, LinearRegression
import logging

from mlcli.preprocessor.base_preprocessor import BasePreprocessor

logger = logging.getLogger(__name__)

# Score function mapping
SCORE_FUNCTIONS = {
    "f_classif": f_classif,
    "f_regression": f_regression,
    "mutual_info_classif": mutual_info_classif,
    "mutual_info_regression": mutual_info_regression,
    "chi2": chi2
}


class SelectKBestProcessor(BasePreprocessor):
    """
    SelectKBest - Select features according to the k highest scores.

    Uses statistical tests to select features.
    """

    def __init__(
        self,
        k: int = 10,
        score_func: str = "f_classif",
        **kwargs
    ):
        """
        Initialize SelectKBest.

        Args:
            k: Number of top features to select
            score_func: Score function name ('f_classif', 'f_regression',
                       'mutual_info_classif', 'mutual_info_regression', 'chi2')
            **kwargs: Additional configuration
        """
        super().__init__(name="select_k_best", **kwargs)
        self.k = k
        self.score_func_name = score_func

        if score_func not in SCORE_FUNCTIONS:
            raise ValueError(
                f"Unknown score function: {score_func}. "
                f"Available: {list(SCORE_FUNCTIONS.keys())}"
            )

        self._preprocessor = SelectKBest(
            score_func=SCORE_FUNCTIONS[score_func],
            k=k
        )
        self._scores: Optional[np.ndarray] = None
        self._selected_features: Optional[List[int]] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SelectKBestProcessor":
        """
        Fit the selector to data.

        Args:
            X: Feature data
            y: Target data (required for scoring)

        Returns:
            Self
        """
        if y is None:
            raise ValueError("SelectKBest requires target variable y")

        logger.info(f"Fitting SelectKBest (k={self.k}, score_func={self.score_func_name})")
        self._preprocessor.fit(X, y)
        self._fitted = True

        self._scores = self._preprocessor.scores_
        self._selected_features = list(self._preprocessor.get_support(indices=True))

        if self._feature_names_in is None:
            self._feature_names_in = [f"feature_{i}" for i in range(X.shape[1])]

        self._feature_names_out = [
            self._feature_names_in[i] for i in self._selected_features
        ]

        logger.info(f"SelectKBest fitted. Selected {len(self._selected_features)} features")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data by selecting k best features.

        Args:
            X: Feature data to transform

        Returns:
            Transformed data with k features
        """
        if not self._fitted:
            raise ValueError("SelectKBest must be fitted before transform")

        return self._preprocessor.transform(X)

    def get_support(self, indices: bool = False) -> np.ndarray:
        """
        Get mask or indices of selected features.

        Args:
            indices: If True, return indices instead of mask

        Returns:
            Boolean mask or indices array
        """
        if not self._fitted:
            raise ValueError("SelectKBest must be fitted first")

        return self._preprocessor.get_support(indices=indices)

    def get_scores(self) -> Dict[str, float]:
        """
        Get feature scores.

        Returns:
            Dictionary mapping feature names to scores
        """
        if not self._fitted or self._scores is None:
            raise ValueError("SelectKBest must be fitted first")

        return dict(zip(self._feature_names_in, self._scores.tolist()))

    def get_params(self) -> Dict[str, Any]:
        """Get selector parameters."""
        params = super().get_params()
        params["k"] = self.k
        params["score_func"] = self.score_func_name
        if self._fitted:
            params["selected_features"] = self._selected_features
            params["scores"] = self._scores.tolist()
        return params


class RFEProcessor(BasePreprocessor):
    """
    RFE - Recursive Feature Elimination.

    Recursively removes features based on estimator importance.
    """

    def __init__(
        self,
        n_features_to_select: int = 10,
        step: int = 1,
        estimator_type: str = "logistic",
        **kwargs
    ):
        """
        Initialize RFE.

        Args:
            n_features_to_select: Number of features to select
            step: Number of features to remove at each iteration
            estimator_type: 'logistic' for classification, 'linear' for regression
            **kwargs: Additional configuration
        """
        super().__init__(name="rfe", **kwargs)
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.estimator_type = estimator_type

        # Create estimator
        if estimator_type == "logistic":
            estimator = LogisticRegression(max_iter=1000, random_state=42)
        elif estimator_type == "linear":
            estimator = LinearRegression()
        else:
            raise ValueError(f"Unknown estimator type: {estimator_type}")

        self._preprocessor = RFE(
            estimator=estimator,
            n_features_to_select=n_features_to_select,
            step=step
        )
        self._ranking: Optional[np.ndarray] = None
        self._selected_features: Optional[List[int]] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "RFEProcessor":
        """
        Fit the RFE selector to data.

        Args:
            X: Feature data
            y: Target data (required)

        Returns:
            Self
        """
        if y is None:
            raise ValueError("RFE requires target variable y")

        logger.info(f"Fitting RFE (n_features={self.n_features_to_select}, step={self.step})")
        self._preprocessor.fit(X, y)
        self._fitted = True

        self._ranking = self._preprocessor.ranking_
        self._selected_features = list(self._preprocessor.get_support(indices=True))

        if self._feature_names_in is None:
            self._feature_names_in = [f"feature_{i}" for i in range(X.shape[1])]

        self._feature_names_out = [
            self._feature_names_in[i] for i in self._selected_features
        ]

        logger.info(f"RFE fitted. Selected {len(self._selected_features)} features")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data by selecting RFE features."""
        if not self._fitted:
            raise ValueError("RFE must be fitted before transform")

        return self._preprocessor.transform(X)

    def get_support(self, indices: bool = False) -> np.ndarray:
        """Get mask or indices of selected features."""
        if not self._fitted:
            raise ValueError("RFE must be fitted first")

        return self._preprocessor.get_support(indices=indices)

    def get_ranking(self) -> Dict[str, int]:
        """
        Get feature ranking.

        Returns:
            Dictionary mapping feature names to ranking (1 = selected)
        """
        if not self._fitted or self._ranking is None:
            raise ValueError("RFE must be fitted first")

        return dict(zip(self._feature_names_in, self._ranking.tolist()))

    def get_params(self) -> Dict[str, Any]:
        """Get selector parameters."""
        params = super().get_params()
        params["n_features_to_select"] = self.n_features_to_select
        params["step"] = self.step
        params["estimator_type"] = self.estimator_type
        if self._fitted:
            params["selected_features"] = self._selected_features
            params["ranking"] = self._ranking.tolist()
        return params


class VarianceThresholdProcessor(BasePreprocessor):
    """
    VarianceThreshold - Feature selector that removes low-variance features.

    Removes features with variance below threshold.
    """

    def __init__(self, threshold: float = 0.0, **kwargs):
        """
        Initialize VarianceThreshold.

        Args:
            threshold: Features with variance <= threshold are removed
            **kwargs: Additional configuration
        """
        super().__init__(name="variance_threshold", **kwargs)
        self.threshold = threshold
        self._preprocessor = VarianceThreshold(threshold=threshold)
        self._variances: Optional[np.ndarray] = None
        self._selected_features: Optional[List[int]] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "VarianceThresholdProcessor":
        """
        Fit the selector to data.

        Args:
            X: Feature data
            y: Ignored (unsupervised method)

        Returns:
            Self
        """
        logger.info(f"Fitting VarianceThreshold (threshold={self.threshold})")
        self._preprocessor.fit(X)
        self._fitted = True

        self._variances = self._preprocessor.variances_
        self._selected_features = list(self._preprocessor.get_support(indices=True))

        if self._feature_names_in is None:
            self._feature_names_in = [f"feature_{i}" for i in range(X.shape[1])]

        self._feature_names_out = [
            self._feature_names_in[i] for i in self._selected_features
        ]

        removed = X.shape[1] - len(self._selected_features)
        logger.info(f"VarianceThreshold fitted. Removed {removed} features")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data by removing low-variance features."""
        if not self._fitted:
            raise ValueError("VarianceThreshold must be fitted before transform")

        return self._preprocessor.transform(X)

    def get_support(self, indices: bool = False) -> np.ndarray:
        """Get mask or indices of selected features."""
        if not self._fitted:
            raise ValueError("VarianceThreshold must be fitted first")

        return self._preprocessor.get_support(indices=indices)

    def get_variances(self) -> Dict[str, float]:
        """
        Get feature variances.

        Returns:
            Dictionary mapping feature names to variances
        """
        if not self._fitted or self._variances is None:
            raise ValueError("VarianceThreshold must be fitted first")

        return dict(zip(self._feature_names_in, self._variances.tolist()))

    def get_params(self) -> Dict[str, Any]:
        """Get selector parameters."""
        params = super().get_params()
        params["threshold"] = self.threshold
        if self._fitted:
            params["selected_features"] = self._selected_features
            params["variances"] = self._variances.tolist()
            params["n_features_removed"] = len(self._feature_names_in) - len(self._selected_features)
        return params
