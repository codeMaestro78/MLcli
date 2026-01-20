"""
Data Analyzer

Analyzes dataset characteristics to inform AutoML decisions:
- Column type detection (numeric, categorical, datetime, text)
- Missing value analysis
- Class imbalance detection
- Task type inference (classification vs regression)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


# Thresholds for type inference
CATEGORICAL_THRESHOLD = 20  # Max unique values to consider categorical
IMBALANCE_THRESHOLD = 0.1  # Min ratio for minority class (below = imbalanced)
TEXT_AVG_LENGTH = 50  # Avg string length to consider as text vs categorical
MISSING_RATIO_WARNING_THRESHOLD = 0.5  # Missing ratio threshold for warnings (>50%)


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
        missing_ratio_warning_threshold: float = MISSING_RATIO_WARNING_THRESHOLD,
    ):
        """
        Initialize DataAnalyzer.

        Args:
            categorical_threshold: Max unique values to consider categorical
            imbalance_threshold: Min class ratio to consider balanced
            missing_ratio_warning_threshold: Missing ratio threshold for warnings
        """
        self.categorical_threshold = categorical_threshold
        self.imbalance_threshold = imbalance_threshold
        self.missing_ratio_warning_threshold = missing_ratio_warning_threshold
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
        # Validate and normalize inputs
        if not isinstance(X, np.ndarray):
            try:
                X = np.asarray(X)
            except (TypeError, ValueError) as e:
                raise ValueError(f"X must be convertible to a numpy array. Error: {e}")

        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got {X.ndim}D array")

        n_samples, n_features = X.shape

        # Generate feature names if not provided
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]
        else:
            if len(feature_names) != n_features:
                raise ValueError(
                    f"feature_names length ({len(feature_names)}) must match "
                    f"number of features ({n_features})"
                )

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
            if col_info.missing_ratio > self.missing_ratio_warning_threshold:
                threshold_pct = int(self.missing_ratio_warning_threshold * 100)
                warnings.append(f"Column '{name}' has >{threshold_pct}% missing values")

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
        """
        Analyze a single column and summarize its characteristics.

        This helper inspects a 1D array representing one column of the dataset,
        handling missing values, counting unique values, inferring a semantic
        column type, and collecting example values.

        Args:
            data: One-dimensional NumPy array containing the column values. May be of
                numeric, boolean, datetime-like, or object dtype, and may include
                missing values (for example, None or np.nan).
            name: The column name as it appears in the original dataset.

        Returns:
            ColumnInfo: A summary of the column, including:
                - name: the original column name.
                - dtype: the NumPy dtype of the input array.
                - inferred_type: a high-level type label (for example
                  "numeric", "categorical", "datetime", "text", or "binary")
                  as determined by _infer_column_type.
                - n_unique: the number of distinct non-missing values.
                - n_missing: the count of missing entries.
                - missing_ratio: the fraction of missing entries over the total
                  number of rows (0.0 if the column is empty).
                - sample_values: up to five representative non-missing values
                  sampled evenly across the valid data.
        """
        # Handle missing values (None, NaN)
        mask = self._get_valid_mask(data)
        n_missing = int((~mask).sum())
        valid_data = data[mask]

        # Get dtype
        dtype = str(data.dtype)

        # Count unique values
        try:
            unique_values = np.unique(valid_data)
            n_unique = len(unique_values)
        except (TypeError, ValueError) as exc:
            # For object arrays with mixed types where np.unique fails,
            # fall back to string-based uniqueness and log a warning so that
            # users are aware this may mask type-related differences.
            logger.warning(
                "Failed to compute unique values for column '%s' with dtype '%s': %s. "
                "Falling back to string-based uniqueness, which may mask type "
                "differences (e.g., 1 vs '1').",
                name,
                dtype,
                exc,
            )
            unique_values = list({str(v) for v in valid_data if v is not None})
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
        """
        Infer the semantic type of a column based on its values and metadata.

        This method analyzes the data type, number of unique values, and content
        characteristics to determine the most appropriate semantic type for AutoML
        preprocessing and model selection.

        Args:
            data: One-dimensional numpy array containing the non-missing values of the column.
            n_unique: Number of distinct values observed in the data.
            dtype: String representation of the numpy data type.

        Returns:
            str: The inferred semantic type. One of:
                - "binary": Exactly 2 unique values
                - "numeric": Numeric data types (int, float) with high cardinality
                - "categorical": Low cardinality data or numeric data with few unique values
                - "datetime": Data with datetime in the type name
                - "text": Object arrays with long average string length (>50 chars)
                - "unknown": Empty data where type inference is not possible
                - "numeric": Default fallback for high cardinality data

        Notes:
            The inference logic follows this priority order:
            1. Empty data detection (returns "unknown")
            2. Binary detection (n_unique == 2)
            3. Numeric type checking with restrictive categorical threshold (≤5 unique values)
            4. Datetime type detection via dtype string
            5. Object/string type analysis using average string length
            6. Cardinality-based categorical detection
            7. Default to numeric for high cardinality unknown types

            Text vs categorical distinction for object and unicode string arrays uses
            random sampling of up to 100 values to compute average string length.
        """
        if len(data) == 0:
            return "unknown"

        # Check for binary
        if n_unique == 2:
            return "binary"

        # Check for numeric
        if np.issubdtype(data.dtype, np.number):
            # Numeric columns with very few unique values might be categorical
            # Use a very restrictive threshold to avoid misclassifying rating scales
            # Only consider categorical if ≤ 3 unique values (very clear categories)
            if n_unique <= 3:
                return "categorical"
            return "numeric"

        # Check for datetime
        if "datetime" in dtype.lower():
            return "datetime"

        # Object/string types
        if data.dtype == object or data.dtype.kind == "U":
            # Check if it's text (long strings) vs categorical (short strings)
            try:
                # Use random sampling across the entire dataset for better accuracy
                non_null_values = [str(v) for v in data if v is not None]
                if non_null_values:
                    sample_size = min(100, len(non_null_values))
                    if len(non_null_values) > sample_size:
                        rng = np.random.default_rng()
                        indices = rng.choice(len(non_null_values), size=sample_size, replace=False)
                        lengths = [len(non_null_values[i]) for i in indices]
                    else:
                        lengths = [len(v) for v in non_null_values]
                    avg_len = np.mean(lengths)
                    if avg_len > TEXT_AVG_LENGTH:
                        return "text"
            except (TypeError, ValueError) as exc:
                # If length computation fails, fall back to cardinality-based inference below.
                logger.debug("Failed to compute average string length: %s", exc)
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
        Analyze target variable and infer the ML task type.

        Performs comprehensive analysis of the target variable including:
        - Task type inference (classification vs regression)
        - Class distribution analysis for classification tasks
        - Imbalance detection
        - Validation of target appropriateness

        Args:
            y: Target variable array

        Returns:
            Tuple containing:
            - target_info: Dict with n_unique, dtype, n_missing, and class_distribution
            - inferred_task: "classification" or "regression"
            - is_imbalanced: Boolean indicating if classification target is imbalanced

        Notes:
            Task inference logic:
            - Binary (2 unique values): classification
            - Multi-class (≤20 unique values): classification
            - Numeric types: regression
            - Non-numeric with high cardinality: classification (with warning if >100 classes)

            class_distribution (only present for classification tasks) is a list of dicts
            with keys: "label" (class value), "count" (number of samples), "ratio" (proportion)
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
            # Non-numeric target with high cardinality - likely problematic
            inferred_task = "classification"
            if n_unique > 100:
                logger.warning(
                    f"Target variable has {n_unique} unique values and is non-numeric. "
                    f"This may indicate an ID column or other inappropriate target variable "
                    f"was selected for classification."
                )

        # Check for class imbalance (only for classification)
        is_imbalanced = False
        class_distribution = []

        if inferred_task == "classification":
            total = len(valid_y)
            for val, count in zip(unique_values, counts):
                ratio = count / total
                class_distribution.append(
                    {
                        "label": val,
                        "count": int(count),
                        "ratio": round(ratio, 4),
                    }
                )
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
        """Get a boolean mask indicating valid (non-null) values in the data array.

        Args:
            data: Input numpy array to check for valid values.

        Returns:
            np.ndarray: Boolean array where True indicates valid values and False indicates
            missing/null values. Uses pd.isna() for comprehensive NaN/null detection
            across all data types.
        """
        # Use pandas.isna() for comprehensive null/NaN detection
        return ~pd.isna(data)

    def _is_nan_array(self, data: np.ndarray) -> np.ndarray:
        """Check for NaN (Not a Number) and null values in the input array.

        Args:
            data: Input numpy array to check for NaN/null values.

        Returns:
            np.ndarray: Boolean array where True indicates NaN or null values.
            Uses pd.isna() for comprehensive detection across all data types.
        """
        return pd.isna(data)

    def get_report(self) -> Optional[DataAnalysisReport]:
        """Get the last generated data analysis report.

        Returns:
            Optional[DataAnalysisReport]: The most recent analysis report if analyze() has been
            called, otherwise None.
        """
        return self._report

    def get_summary(self) -> Dict[str, Any]:
        """Get a JSON-serializable summary of the data analysis.

        Returns:
            Dict[str, Any]: A dictionary containing key analysis metrics including
            sample count, feature count, column types, missing data statistics,
            inferred task type, imbalance status, target information, and warnings.
            Returns an empty dictionary if no analysis has been performed yet.
        """
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
