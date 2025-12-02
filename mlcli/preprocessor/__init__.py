"""
MLCLI Data Preprocessing Module

Provides data preprocessing utilities including:
- Scaling (StandardScaler, MinMaxScaler, RobustScaler)
- Normalization (L1, L2, Max)
- Encoding (LabelEncoder, OneHotEncoder, OrdinalEncoder)
- Feature Selection (SelectKBest, RFE, VarianceThreshold)
"""

from mlcli.preprocessor.base_preprocessor import BasePreprocessor
from mlcli.preprocessor.scalers import (
    StandardScalerProcessor,
    MinMaxScalerProcessor,
    RobustScalerProcessor
)
from mlcli.preprocessor.normalizers import NormalizerProcessor
from mlcli.preprocessor.encoders import (
    LabelEncoderProcessor,
    OneHotEncoderProcessor,
    OrdinalEncoderProcessor
)
from mlcli.preprocessor.feature_selectors import (
    SelectKBestProcessor,
    RFEProcessor,
    VarianceThresholdProcessor
)
from mlcli.preprocessor.preprocessor_factory import PreprocessorFactory, get_preprocessor
from mlcli.preprocessor.pipeline import PreprocessingPipeline

__all__ = [
    "BasePreprocessor",
    "StandardScalerProcessor",
    "MinMaxScalerProcessor",
    "RobustScalerProcessor",
    "NormalizerProcessor",
    "LabelEncoderProcessor",
    "OneHotEncoderProcessor",
    "OrdinalEncoderProcessor",
    "SelectKBestProcessor",
    "RFEProcessor",
    "VarianceThresholdProcessor",
    "PreprocessorFactory",
    "get_preprocessor",
    "PreprocessingPipeline"
]
