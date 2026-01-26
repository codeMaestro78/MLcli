"""
Preprocessing Selector

Automatically selects preprocessing steps based on DataAnalyzer output.
Builds a PreprocessingPipeline using existing mlcli preprocessors.

Design Goals:
- Leverage existing mlcli/preprocessor infrastructure
- No new preprocessor implementations
- Integrate with DataAnalyzer output
- Return a ready-to-use PreprocessingPipeline
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import logging

from mlcli.preprocessor.pipeline import PreprocessingPipeline
from mlcli.preprocessor.preprocessor_factory import PreprocessorFactory
from mlcli.automl.data_analyzer import DataAnalysisReport

logger = logging.getLogger(__name__)

# Cardinality thresholds for encoder selection
ONEHOT_MAX_CARDINALITY = 10
ORDINAL_MAX_CARDINALITY = 50

# Outlier detection thresholds
OUTLIER_IQR_MULTIPLIER = 1.5
OUTLIER_RATIO_THRESHOLD = 0.05


@dataclass
class PreprocessingRecommendation:
    """
    Represents a single preprocessing step recommendation.

    Attributes:
        step_name: Logical name of the step (e.g., 'scaling', 'encoding_onehot')
        method: Preprocessor method name understood by PreprocessorFactory
        params: Parameters passed to the preprocessor
        reason: Human-readable explanation for this choice
        target_columns: Columns this step should apply to
    """
    step_name: str
    method: str
    params: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    target_columns: List[str] = field(default_factory=list)


@dataclass
class PreprocessingPlan:
    """
    Represents the complete preprocessing plan.

    Attributes:
        recommendations: Ordered list of preprocessing recommendations
        scaler_method: Selected scaler method (if any)
        has_missing_values: Whether missing values were detected
        missing_strategy: Recommended missing-value handling strategy
        warnings: Human-readable warnings about the dataset
    """
    recommendations: List[PreprocessingRecommendation] = field(default_factory=list)
    scaler_method: Optional[str] = None
    has_missing_values: bool = False
    missing_strategy: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


class PreprocessingSelector:
    """
    Automatically selects preprocessing steps based on dataset characteristics.

    This class uses DataAnalysisReport output to decide:
    - Feature scaling strategy
    - Multi-encoder strategy for categorical features (OneHot + Ordinal)
    - Missing-value handling recommendations

    Note:
        The preprocessing pipeline does NOT perform imputation.
        Missing-value handling is only recommended to the user.
    """

    def __init__(
        self,
        prefer_robust_scaling: bool = False,
        onehot_max_cardinality: int = ONEHOT_MAX_CARDINALITY,
        ordinal_max_cardinality: int = ORDINAL_MAX_CARDINALITY,
    ):
        """
        Initialize the PreprocessingSelector.

        Args:
            prefer_robust_scaling: Force RobustScaler regardless of data
            onehot_max_cardinality: Max unique values for OneHot encoding
            ordinal_max_cardinality: Max unique values for Ordinal encoding
        """
        self.prefer_robust_scaling = prefer_robust_scaling
        self.onehot_max_cardinality = onehot_max_cardinality
        self.ordinal_max_cardinality = ordinal_max_cardinality
        self._plan: Optional[PreprocessingPlan] = None

    def select(
        self,
        report: DataAnalysisReport,
        X: Optional[np.ndarray] = None,
    ) -> PreprocessingPlan:
        """
        Generate a preprocessing plan based on dataset analysis.

        Args:
            report: Output from DataAnalyzer
            X: Optional feature matrix for outlier analysis

        Returns:
            A fully populated PreprocessingPlan
        """
        recommendations: List[PreprocessingRecommendation] = []
        warnings: List[str] = []

        logger.info(
            f"Selecting preprocessing for {report.n_features} features: "
            f"{len(report.numeric_columns)} numeric, "
            f"{len(report.categorical_columns)} categorical"
        )

        # Missing values
        has_missing = report.missing_ratio > 0
        missing_strategy = None

        if has_missing:
            missing_strategy = self._recommend_missing_strategy(report)
            warnings.append(
                f"Dataset has {report.missing_ratio:.1%} missing values. "
                f"Recommended strategy: {missing_strategy}. "
                "Please handle missing values before running AutoML."
            )

        # Scaling
        scaler_method = None
        if report.numeric_columns:
            scaler_method, reason = self._select_scaler(report, X)
            recommendations.append(
                PreprocessingRecommendation(
                    step_name="scaling",
                    method=scaler_method,
                    reason=reason,
                    target_columns=report.numeric_columns.copy(),
                )
            )

        # Encoding (multi-encoder)
        if report.categorical_columns:
            encoder_recs, encoder_warnings = self._select_encoders(report)
            recommendations.extend(encoder_recs)
            warnings.extend(encoder_warnings)

        # Step ordering
        priority = {
            "scaling": 0,
            "encoding_onehot": 1,
            "encoding_ordinal": 2,
        }
        recommendations.sort(key=lambda r: priority.get(r.step_name, 99))

        self._plan = PreprocessingPlan(
            recommendations=recommendations,
            scaler_method=scaler_method,
            has_missing_values=has_missing,
            missing_strategy=missing_strategy,
            warnings=warnings,
        )

        return self._plan

    def _select_scaler(
        self,
        report: DataAnalysisReport,
        X: Optional[np.ndarray],
    ) -> Tuple[str, str]:
        """
        Select a scaling strategy based on outliers and target imbalance.

        Returns:
            (scaler_method, explanation)
        """
        if self.prefer_robust_scaling:
            return "robust_scaler", "User preference for robust scaling"

        if X is not None and report.numeric_columns:
            outlier_ratio = self._estimate_outlier_ratio(X, report)
            if outlier_ratio > OUTLIER_RATIO_THRESHOLD:
                return "robust_scaler", f"High outlier ratio detected ({outlier_ratio:.1%})"

        if report.is_imbalanced:
            return "robust_scaler", "Imbalanced target suggests potential outliers"

        return "standard_scaler", "Default choice for normally distributed features"

    def _estimate_outlier_ratio(
        self,
        X: np.ndarray,
        report: DataAnalysisReport,
    ) -> float:
        """
        Estimate outlier ratio using the IQR method.

        Returns:
            Fraction of values classified as outliers
        """
        total_values = 0
        outliers = 0

        column_names = [col.name for col in report.columns]

        for col_name in report.numeric_columns:
            try:
                idx = column_names.index(col_name)
                values = X[:, idx].astype(float)
                values = values[~np.isnan(values)]

                if len(values) < 4:
                    continue

                q1, q3 = np.percentile(values, [25, 75])
                iqr = q3 - q1
                lower = q1 - OUTLIER_IQR_MULTIPLIER * iqr
                upper = q3 + OUTLIER_IQR_MULTIPLIER * iqr

                outliers += ((values < lower) | (values > upper)).sum()
                total_values += len(values)

            except Exception:
                continue

        return 0.0 if total_values == 0 else outliers / total_values

    def _select_encoders(
        self,
        report: DataAnalysisReport,
    ) -> Tuple[List[PreprocessingRecommendation], List[str]]:
        """
        Select encoding strategies based on categorical feature cardinality.

        Returns:
            encoder_recommendations: List of encoding steps
            warnings: High-cardinality warnings
        """
        recommendations: List[PreprocessingRecommendation] = []
        warnings: List[str] = []

        low, medium, high = [], [], []

        for col in report.columns:
            if col.name not in report.categorical_columns:
                continue

            if col.n_unique <= self.onehot_max_cardinality:
                low.append(col.name)
            elif col.n_unique <= self.ordinal_max_cardinality:
                medium.append(col.name)
            else:
                high.append(col.name)

        if low:
            recommendations.append(
                PreprocessingRecommendation(
                    step_name="encoding_onehot",
                    method="onehot_encoder",
                    params={"sparse_output": False, "handle_unknown": "ignore"},
                    reason=f"Low-cardinality categorical features (n_unique <= {self.onehot_max_cardinality})",
                    target_columns=low,
                )
            )

        if medium:
            recommendations.append(
                PreprocessingRecommendation(
                    step_name="encoding_ordinal",
                    method="ordinal_encoder",
                    params={"handle_unknown": "use_encoded_value", "unknown_value": -1},
                    reason=f"Medium-cardinality categorical features ({self.onehot_max_cardinality} < n_unique <= {self.ordinal_max_cardinality})",
                    target_columns=medium,
                )
            )

        if high:
            warnings.append(
                f"High-cardinality categorical columns excluded: {high}. "
                "Consider target encoding or hashing."
            )

        return recommendations, warnings

    def _recommend_missing_strategy(self, report: DataAnalysisReport) -> str:
        """
        Recommend a missing-value handling strategy.

        Returns:
            Strategy name (string)
        """
        if report.missing_ratio > 0.3:
            return "drop_rows_or_columns"

        if len(report.numeric_columns) > len(report.categorical_columns):
            return "median_imputation"

        if len(report.categorical_columns) > len(report.numeric_columns):
            return "most_frequent_imputation"

        return "mixed_imputation"

    def build_pipeline(
        self,
        plan: Optional[PreprocessingPlan] = None,
    ) -> PreprocessingPipeline:
        """
        Build a PreprocessingPipeline from a preprocessing plan.

        Args:
            plan: Optional plan (defaults to last generated plan)

        Returns:
            Configured PreprocessingPipeline

        Note:
            The current PreprocessingPipeline applies transformations to ALL columns.
            Column-specific preprocessing (target_columns) is stored in recommendations
            for documentation purposes but not enforced by the pipeline.
            For mixed-type data, ensure numeric/categorical columns are handled separately.
        """
        plan = plan or self._plan
        if plan is None:
            raise ValueError("No preprocessing plan available. Call select() first.")

        if plan.has_missing_values:
            logger.warning(
                "Dataset has missing values. The pipeline does not include imputation. "
                f"Recommended strategy: {plan.missing_strategy}"
            )

        pipeline = PreprocessingPipeline()

        for rec in plan.recommendations:
            try:
                preprocessor = PreprocessorFactory.create(rec.method, **rec.params)
                pipeline.add_step(rec.step_name, preprocessor)
                logger.info(
                    f"Added pipeline step: {rec.step_name} ({rec.method}) "
                    f"for columns: {rec.target_columns}"
                )
            except Exception as e:
                logger.error(f"Failed to create preprocessor {rec.method}: {e}")
                raise

        return pipeline

    def get_plan(self) -> Optional[PreprocessingPlan]:
        # Return the last generated preprocessing plan.
        return self._plan

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a JSON-serializable summary of the preprocessing plan.

        Returns:
            Dictionary with plan details
        """
        if self._plan is None:
            return {}

        return {
            "scaler_method": self._plan.scaler_method,
            "has_missing_values": self._plan.has_missing_values,
            "missing_strategy": self._plan.missing_strategy,
            "n_steps": len(self._plan.recommendations),
            "steps": [
                {
                    "name": rec.step_name,
                    "method": rec.method,
                    "reason": rec.reason,
                    "target_columns": rec.target_columns,
                }
                for rec in self._plan.recommendations
            ],
            "warnings": self._plan.warnings,
        }

    def __repr__(self) -> str:
        return (
            f"PreprocessingSelector(n_steps={len(self._plan.recommendations)})"
            if self._plan else "PreprocessingSelector(not configured)"
        )
