"""
Input/Output Utilities

Handles data loading (CSV, images), model saving/loading,
and file path management.
"""

import pickle
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union, Any, Dict, List
import logging
import json

logger = logging.getLogger(__name__)


def load_data(
    data_path: Union[str, Path],
    data_type: str = "csv",
    target_column: Optional[str] = None,
    features: Optional[List[str]] = None,
    **kwargs,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load dataset from various formats.

    Args:
        data_path: Path to data file or directory
        data_type: Type of data ('csv', 'image', 'numpy')
        target_column: Name of target column (for CSV)
        features: List of feature column names (for CSV)
        **kwargs: Additional arguments for specific loaders

    Returns:
        Tuple of (features, targets) as numpy arrays

    Raises:
        ValueError: If data type is unsupported
        FileNotFoundError: If data path doesn't exist
    """
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found :{data_path}")

    logger.info(f"Loading {data_path} data from : {data_path}")

    if data_type == "csv":
        return _load_csv(data_path, target_column, features, **kwargs)
    elif data_type == "image":
        return _load_images(data_path, **kwargs)
    elif data_type == "numpy":
        return _load_numpy(data_path, **kwargs)
    else:
        raise ValueError(
            f"Unsupported data type :{data_path}. ", f"Supported types: csv,image,numpy"
        )


def _load_csv(
    csv_path: Path,
    target_column: Optional[str] = None,
    features: Optional[List[str]] = None,
    **kwargs,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load data from CSV file.

    Args:
        csv_path: Path to CSV file
        target_column: Name of target column
        features: List of feature columns
        **kwargs: Additional pandas read_csv arguments

    Returns:
        Tuple of (X, y) as numpy arrays
    """

    df = pd.read_csv(csv_path, **kwargs)
    logger.info(f"Loaded CSV with shape: {df.shape}")

    # Extract features
    if features:
        X = df[features].values
    elif target_column:
        X = df.drop(columns=[target_column]).values
    else:
        X = df.values

    # Extract target
    if target_column:
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        y = df[target_column].values
        logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    else:
        logger.info(f"Features shape: {X.shape} (no target column specified)")

    return X, y


def _load_images(
    image_dir: Path, image_size: Tuple[int, int] = (224, 224), color_mode: str = "rgb", **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load images from directory structure.

    Expected structure:
        image_dir/
            class1/
                img1.jpg
                img2.jpg
            class2/
                img3.jpg

    Args:
        image_dir: Path to image directory
        image_size: Target size for images (height, width)
        color_mode: 'rgb' or 'grayscale'
        **kwargs: Additional arguments

    Returns:
        Tuple of (images, labels) as numpy arrays
    """
    try:
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
    except ImportError:
        raise ImportError(
            "TensorFlow is required for image loading. " "Install with: pip install tensorflow"
        )

    # Use ImageDataGenerator to load images
    datagen = ImageDataGenerator(rescale=1.0 / 255)

    generator = datagen.flow_from_directory(
        str(image_dir),
        target_size=image_size,
        color_mode=color_mode,
        class_mode="sparse",
        batch_size=32,
        shuffle=False,
    )

    # Load all images
    images = []
    labels = []

    for i in range(len(generator)):
        batch_images, batch_labels = generator[i]
        images.append(batch_images)
        labels.append(batch_labels)

    X = np.vstack(images)
    y = np.hstack(labels)

    logger.info(f"Loaded {len(X)} images with shape: {X.shape}")
    logger.info(f"Classses: {generator.class_indices}")

    return X, y


def _load_numpy(
    numpy_path: Path, features_file: str = "X.npy", target_file: str = "y.npy", **kwargs
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load data from numpy files.

    Args:
        numpy_path: Path to directory containing numpy files
        features_file: Name of features file
        target_file: Name of target file
        **kwargs: Additional arguments

    Returns:
        Tuple of (X, y) as numpy arrays
    """

    X_path = numpy_path / features_file
    y_path = numpy_path / target_file

    if not X_path.exists():
        raise FileNotFoundError(f"Features file not found: {X_path}")

    X = np.load(X_path)
    logger.info(f"Loaded fetures with shape: {X.shape}")

    y = None
    if y_path.exists():
        y = np.load(y_path)
        logger.info(f"Loaded targets with shape: {y.shape}")

    return X, y


def save_model(
    model: Any,
    save_path: Union[str, Path],
    model_format: str = "pickle",
    metadata: Optional[dict] = None,
) -> Path:
    """
    Save model to disk in specified format.

    Args:
        model: Model object to save
        save_path: Path to save model
        model_format: Format ('pickle', 'joblib', 'onnx', 'h5', 'savedmodel')
        metadata: Optional metadata to save alongside model

    Returns:
        Path where model was saved

    Raises:
        ValueError: If format is unsupported
    """

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving model in {model_format} format to: {save_path}")

    if model_format == "pickle":
        with open(save_path, "wb") as f:
            pickle.dump(model, f)

    elif model_format == "joblib":
        joblib.dump(model, save_path)

    elif model_format == "onnx":
        # ONNX save handled by trainer (requires conversion)
        raise NotImplementedError("ONNX save should be handled by trainer")

    elif model_format in ["h5", "savedmodel"]:
        # Tensorflow foramts handled by trainer
        raise NotImplementedError(f"{model_format} save should be handled by trainer")

    else:
        raise ValueError(f"Unsupported format: {model_format}")

    # Save metadata if provided

    if metadata:
        metadata_path = save_path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.debug(f"Saved metadata to :{metadata_path}")

    logger.info(f"Model saved successfully")
    return save_path


def load_model(model_path: Union[str, Path], model_format: str = "pickle") -> Any:
    """
    Load model from disk.

    Args:
        model_path: Path to model file
        model_format: Format ('pickle', 'joblib', 'onnx', 'h5', 'savedmodel')

    Returns:
        Loaded model object

    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If format is unsupported
    """

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logger.info(f"Loading model from :{model_path}")

    if model_format == "pickle":
        with open(model_path, "rb") as f:
            model = pickle.load(f)

    elif model_format == "joblib":
        model = joblib.load(model_path)

    elif model_format == "onnx":
        import onnxruntime as ort

        model = ort.InferenceSession(str(model_path))

    elif model_format == "h5":
        from tensorflow.keras.models import load_model as keras_load

        model = keras_load(str(model_path))

    elif model_format == "savedmodel":
        import tensorflow as tf

        model = tf.keras.models.load_model(str(model_path))

    else:
        raise ValueError(f"Unsupported foramt: {model_format}")

    logger.info("Model loaded successfully")
    return model


def ensure_dir(dir_path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if not.

    Args:
        dir_path: Directory path

    Returns:
        Path object for the directory
    """
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_available_configs(config_dir: Union[str, Path] = "configs") -> List[Path]:
    """
    Get list of available configuration files.

    Args:
        config_dir: Directory containing config files

    Returns:
        List of config file paths
    """
    config_dir = Path(config_dir)

    if not config_dir.exists():
        logger.warning(f"Config directory not found: {config_dir}")
        return []

    configs = []
    for ext in [".json", ".yaml", ".yml"]:
        configs.extend(config_dir.glob(f"*{ext}"))

    return sorted(configs)


def fet_available_models(model_dir: Union[str, Path] = "mlcli/models") -> Dict[str, List[Path]]:
    """
    Get list of saved models by format.

    Args:
        model_dir: Directory containing model files

    Returns:
        Dictionary mapping format to list of model paths
    """
    model_dir = Path(model_dir)

    if not model_dir.exists():
        logger.warning(f"Model directory not found: {model_dir}")
        return {}

    models = {
        "pickle": list(model_dir.glob("*.pkl")),
        "joblib": list(model_dir.glob("*.joblib")),
        "onnx": list(model_dir.glob("*.onnx")),
        "h5": list(model_dir.glob("*.h5")),
        "savedmodel": [d for d in model_dir.glob("*/") if (d / "saved_model.pb").exists()],
    }
    return {k: v for k, v in models.items() if v}
