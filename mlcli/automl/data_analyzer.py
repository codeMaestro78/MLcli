"""
Data Analyzer

Analyzes dataset characteristics to inform AutoML decisions:
- Column type detection (numeric, categorical, datetime, text)
- Missing value analysis
- Class imbalance detection
- Task type inference (classification vs regression)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import logging

logger = logging.getLogger(__name__)


# Thresholds for type inference
CATEGORICAL_THRESHOLD = 20  # Max unique values to consider categorical
IMBALANCE_THRESHOLD = 0.1  # Min ratio for minority class (below = imbalanced)
TEXT_AVG_LENGTH = 50  # Avg string length to consider as text vs categorical


@dataclass
class ColumnInfo:
    """Information about a single column."""

    name: str
    dtype: str
    inferred_type: str  # 'numeric', 'categorical', 'datetime', 'text', 'binary'
    n_unique: int
    n_missing: int
    missing_ratio: float
    sample_values: List[Any] = field(default_factory=list)


@dataclass
class DataAnalysisReport:
    """Complete data analysis report."""

    n_samples: int
    n_features: int
    columns: List[ColumnInfo]
    numeric_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str]
    text_columns: List[str]
    binary_columns: List[str]
    total_missing: int
    missing_ratio: float
    target_info: Optional[Dict[str, Any]] = None
    inferred_task: Optional[str] = None  # 'classification' or 'regression'
    is_imbalanced: bool = False
    warnings: List[str] = field(default_factory=list)


class DataAnalyzer:
    """
    Analyzes datasets for AutoML preprocessing and model selection.

    Works with numpy arrays. For pandas DataFrames, convert first.
    """

    def __init__(
        self,
        categorical_threshold: int = CATEGORICAL_THRESHOLD,
        imbalance_threshold: float = IMBALANCE_THRESHOLD,
    ):
        """
        Initialize DataAnalyzer.

        Args:
            categorical_threshold: Max unique values to consider categorical
            imbalance_threshold: Min class ratio to consider balanced
        """
        self.categorical_threshold = categorical_threshold
        self.imbalance_threshold = imbalance_threshold
        self._report: Optional[DataAnalysisReport] = None

    def analyze(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> DataAnalysisReport:
        """
        Analyze dataset and generate report.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (optional)
            feature_names: Column names (optional)

        Returns:
            DataAnalysisReport with all analysis results
        """
        n_samples, n_features = X.shape

        # Generate feature names if not provided
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        logger.info(f"Analyzing dataset: {n_samples} samples, {n_features} features")

        # Analyze each column
        columns: List[ColumnInfo] = []
        numeric_cols: List[str] = []
        categorical_cols: List[str] = []
        datetime_cols: List[str] = []
        text_cols: List[str] = []
        binary_cols: List[str] = []
        total_missing = 0
        warnings: List[str] = []

        for i, name in enumerate(feature_names):
            col_data = X[:, i]
            col_info = self._analyze_column(col_data, name)
            columns.append(col_info)
            total_missing += col_info.n_missing

            # Categorize column
            if col_info.inferred_type == "numeric":
                numeric_cols.append(name)
            elif col_info.inferred_type == "categorical":
                categorical_cols.append(name)
            elif col_info.inferred_type == "datetime":
                datetime_cols.append(name)
            elif col_info.inferred_type == "text":
                text_cols.append(name)
            elif col_info.inferred_type == "binary":
                binary_cols.append(name)

            # Warn about high missing ratio
            if col_info.missing_ratio > 0.5:
                warnings.append(f"Column '{name}' has >50% missing values")

        # Calculate overall missing ratio
        total_cells = n_samples * n_features
        missing_ratio = total_missing / total_cells if total_cells > 0 else 0.0

        # Analyze target if provided
        target_info = None
        inferred_task = None
        is_imbalanced = False

        if y is not None:
            target_info, inferred_task, is_imbalanced = self._analyze_target(y)
            if is_imbalanced:
                warnings.append("Target variable is imbalanced")

        # Build report
        self._report = DataAnalysisReport(
            n_samples=n_samples,
            n_features=n_features,
            columns=columns,
            numeric_columns=numeric_cols,
            categorical_columns=categorical_cols,
            datetime_columns=datetime_cols,
            text_columns=text_cols,
            binary_columns=binary_cols,
            total_missing=total_missing,
            missing_ratio=missing_ratio,
            target_info=target_info,
            inferred_task=inferred_task,
            is_imbalanced=is_imbalanced,
            warnings=warnings,
        )

        logger.info(
            f"Analysis complete: {len(numeric_cols)} numeric, "
            f"{len(categorical_cols)} categorical, "
            f"task={inferred_task}, imbalanced={is_imbalanced}"
        )

        return self._report

    def _analyze_column(self, data: np.ndarray, name: str) -> ColumnInfo:
        # Analyze a single column.
        # Handle missing values (None, NaN)
        mask = self._get_valid_mask(data)
        n_missing = int((~mask).sum())
        valid_data = data[mask]

        # Get dtype
        dtype = str(data.dtype)

        # Count unique values
        try:
            unique_values = np.unique(valid_data[~self._is_nan_array(valid_data)])
            n_unique = len(unique_values)
        except (TypeError, ValueError):
            # For object arrays with mixed types
            unique_values = list(set(str(v) for v in valid_data if v is not None))
            n_unique = len(unique_values)

        # Infer type
        inferred_type = self._infer_column_type(valid_data, n_unique, dtype)

        # Sample values (up to 5)
        sample_values = []
        if len(valid_data) > 0:
            indices = np.linspace(0, len(valid_data) - 1, min(5, len(valid_data)), dtype=int)
            sample_values = [valid_data[i] for i in indices]

        return ColumnInfo(
            name=name,
            dtype=dtype,
            inferred_type=inferred_type,
            n_unique=n_unique,
            n_missing=n_missing,
            missing_ratio=n_missing / len(data) if len(data) > 0 else 0.0,
            sample_values=sample_values,
        )

    def _infer_column_type(self, data: np.ndarray, n_unique: int, dtype: str) -> str:
        # Infer the semantic type of a column.
        if len(data) == 0:
            return "numeric"

        # Check for binary
        if n_unique == 2:
            return "binary"

        # Check for numeric
        if np.issubdtype(data.dtype, np.number):
            # Even numeric columns with few unique values might be categorical
            if n_unique <= self.categorical_threshold and n_unique < len(data) * 0.05:
                return "categorical"
            return "numeric"

        # Check for datetime
        if "datetime" in dtype.lower():
            return "datetime"

        # Object/string types
        if data.dtype == object:
            # Check if it's text (long strings) vs categorical (short strings)
            try:
                avg_len = np.mean([len(str(v)) for v in data[:100] if v is not None])
                if avg_len > TEXT_AVG_LENGTH:
                    return "text"
            except (TypeError, ValueError):
                pass

            # Check if low cardinality -> categorical
            if n_unique <= self.categorical_threshold:
                return "categorical"

            return "text"

        # Default to categorical for other types
        if n_unique <= self.categorical_threshold:
            return "categorical"

        return "numeric"

    def _analyze_target(self, y: np.ndarray) -> Tuple[Dict[str, Any], str, bool]:
        """
        Analyze target variable.

        Returns:
            (target_info dict, inferred_task, is_imbalanced)
        """
        # Get valid values
        mask = self._get_valid_mask(y)
        valid_y = y[mask]

        # Count unique values
        unique_values, counts = np.unique(valid_y, return_counts=True)
        n_unique = len(unique_values)

        # Infer task type
        if n_unique == 2:
            inferred_task = "classification"  # Binary
        elif n_unique <= self.categorical_threshold:
            inferred_task = "classification"  # Multi-class
        elif np.issubdtype(valid_y.dtype, np.number):
            inferred_task = "regression"
        else:
            inferred_task = "classification"

        # Check for class imbalance (only for classification)
        is_imbalanced = False
        class_distribution = {}

        if inferred_task == "classification":
            total = len(valid_y)
            for val, count in zip(unique_values, counts):
                ratio = count / total
                class_distribution[str(val)] = {
                    "count": int(count),
                    "ratio": round(ratio, 4),
                }
                if ratio < self.imbalance_threshold:
                    is_imbalanced = True

        target_info = {
            "n_unique": n_unique,
            "dtype": str(y.dtype),
            "n_missing": int((~mask).sum()),
            "class_distribution": class_distribution if inferred_task == "classification" else None,
        }

        return target_info, inferred_task, is_imbalanced

    def _get_valid_mask(self, data: np.ndarray) -> np.ndarray:
        # Get mask of valid (non-null) values.
        if np.issubdtype(data.dtype, np.floating):
            return ~np.isnan(data)
        elif data.dtype == object:
            return np.array([v is not None and v == v for v in data])  # v == v catches NaN
        else:
            return np.ones(len(data), dtype=bool)

    def _is_nan_array(self, data: np.ndarray) -> np.ndarray:
        # Check for NaN values in array.
        if np.issubdtype(data.dtype, np.floating):
            return np.isnan(data)
        return np.zeros(len(data), dtype=bool)

    def get_report(self) -> Optional[DataAnalysisReport]:
        # Get the last analysis report.
        return self._report

    def get_summary(self) -> Dict[str, Any]:
        # Get a JSON-serializable summary of the analysis.
        if self._report is None:
            return {}

        return {
            "n_samples": self._report.n_samples,
            "n_features": self._report.n_features,
            "numeric_columns": self._report.numeric_columns,
            "categorical_columns": self._report.categorical_columns,
            "binary_columns": self._report.binary_columns,
            "text_columns": self._report.text_columns,
            "datetime_columns": self._report.datetime_columns,
            "total_missing": self._report.total_missing,
            "missing_ratio": round(self._report.missing_ratio, 4),
            "inferred_task": self._report.inferred_task,
            "is_imbalanced": self._report.is_imbalanced,
            "target_info": self._report.target_info,
            "warnings": self._report.warnings,
        }

    def __repr__(self) -> str:
        if self._report:
            return (
                f"DataAnalyzer(analyzed={self._report.n_samples}x{self._report.n_features}, "
                f"task={self._report.inferred_task})"
            )
        return "DataAnalyzer(not analyzed)"
