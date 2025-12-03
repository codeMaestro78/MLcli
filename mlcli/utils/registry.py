"""
Model Registry System

Provides decorator-based auto-registration for all trainer classes,
enabling dynamic model discovery and instantiation from configuration.
Supports lazy loading to avoid importing heavy dependencies like TensorFlow.
"""

from typing import Dict, Type, Optional, List, Any
import logging

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Central registry for all model trainers.

    Maps model type strings (e.g., 'logistic_regression') to their corresponding
    trainer class implementations. Supports automatic registration via decorator
    and lazy loading for heavy dependencies.
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._registry: Dict[str, Type] = {}
        self._lazy_registry: Dict[str, Dict[str, str]] = {}  # For lazy loading
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, trainer_class: Type, description: str = "",
                 framework: str = "unknown", model_type: str = "unknown") -> None:
        """
        Register a trainer class with metadata.

        Args:
            name: Unique identifier for the model (e.g., 'logistic_regression')
            trainer_class: The trainer class to register
            description: Human-readable description of the model
            framework: ML framework (e.g., 'sklearn', 'tensorflow', 'xgboost')
            model_type: Type of model (e.g., 'classification', 'regression')

        Raises:
            ValueError: If name is already registered
        """

        if name in self._registry:
            logger.warning(f"Model '{name}' is already registered. Overwriting")

        self._registry[name] = trainer_class
        self._metadata[name] = {
            "description": description,
            "framework": framework,
            "model_type": model_type,
            "class_name": trainer_class.__name__
        }

        logger.debug(f"Registered model: {name} -> {trainer_class.__name__}")

    def register_lazy(self, name: str, module_path: str, class_name: str,
                      description: str = "", framework: str = "unknown",
                      model_type: str = "unknown") -> None:
        """
        Register a trainer for lazy loading (doesn't import the module yet).

        Args:
            name: Unique identifier for the model
            module_path: Full module path (e.g., 'mlcli.trainers.tf_dnn_trainer')
            class_name: Class name to import from module
            description: Human-readable description
            framework: ML framework name
            model_type: Type of model
        """
        self._lazy_registry[name] = {
            "module_path": module_path,
            "class_name": class_name,
        }
        self._metadata[name] = {
            "description": description,
            "framework": framework,
            "model_type": model_type,
            "class_name": class_name,
            "lazy": True,
        }
        logger.debug(f"Registered lazy model: {name} -> {module_path}.{class_name}")

    def _resolve_lazy(self, name: str) -> Optional[Type]:
        """Resolve a lazy-registered model by importing it."""
        if name not in self._lazy_registry:
            return None
        
        import importlib
        lazy_info = self._lazy_registry[name]
        module = importlib.import_module(lazy_info["module_path"])
        trainer_class = getattr(module, lazy_info["class_name"])
        
        # Move from lazy to regular registry
        self._registry[name] = trainer_class
        del self._lazy_registry[name]
        
        # Update metadata
        if name in self._metadata:
            self._metadata[name]["lazy"] = False
        
        return trainer_class

    def get(self, name: str) -> Optional[Type]:
        """
        Retrieve a trainer class by name.

        Args:
            name: Model identifier

        Returns:
            Trainer class or None if not found
        """
        # Check regular registry first
        if name in self._registry:
            return self._registry.get(name)
        
        # Try lazy loading
        if name in self._lazy_registry:
            return self._resolve_lazy(name)
        
        return None

    def get_trainer(self, name: str, **kwargs) -> Any:
        """
        Instantiate a trainer by name.

        Args:
            name: Model identifier
            **kwargs: Arguments to pass to trainer constructor

        Returns:
            Instantiated trainer object

        Raises:
            KeyError: If model name not found in registry
        """
        trainer_class = self.get(name)
        if trainer_class is None:
            available = ", ".join(self.list_models())
            raise KeyError(f"Model '{name}' not found in registry. "
                          f"Available models: {available}")

        return trainer_class(**kwargs)

    def list_models(self) -> List[str]:
        """
        Get list of all registered model names (including lazy).

        Returns:
            List of model identifiers
        """
        all_models = set(self._registry.keys()) | set(self._lazy_registry.keys())
        return sorted(all_models)


    def get_metadata(self,name:str)->Optional[Dict[str,Any]]:
        """
        Get metadata for a registered model.

        Args:
            name: Model identifier

        Returns:
            Metadata dictionary or None if not found
        """
        return self._metadata.get(name)

    def get_all_metadata(self)->Dict[str,Dict[str,Any]]:
        """
        Get metadata for all registered models.

        Returns:
            Dictionary mapping model names to their metadata
        """
        return self._metadata.copy()
    def get_models_by_framework(self,framework:str)->List[str]:
        """
        Get all models for a specific framework.

        Args:
            framework: Framework name (e.g., 'sklearn', 'tensorflow')

        Returns:
            List of model names
        """
        return [name for name,meta in self._metadata.items() if meta.get("framework")==framework]

    def is_registered(self, name: str) -> bool:
        """
        Check if a model is registered (including lazy).

        Args:
            name: Model identifier

        Returns:
            True if registered, False otherwise
        """
        return name in self._registry or name in self._lazy_registry

    def unregister(self,name:str)->bool:
        """
        Remove a model from the registry.

        Args:
            name: Model identifier

        Returns:
            True if removed, False if not found
        """
        if name in self._registry:
            del self._registry[name]
            del self._metadata[name]
            logger.debug(f"Unregisterd model: {name}")
            return True
        return False

    def __len__(self)->int:
        """Return number of registered models. """
        return len(self._registry)

    def __contains__(self,name:str)->bool:
        """Check if models is registered using 'in' operator. """
        return name in self._registry

    def __repr__(self)->str:
        """String representation of registry. """
        return f"ModelRegistry(models- {len(self._registry)})"

def register_model(name:str,description:str="",framework:str="unknown",model_type:str="classification"):
    """
    Decorator for auto-registering trainer classes.

    Usage:
        @register_model("logistic_regression", description="Logistic Regression",
                       framework="sklearn", model_type="classification")
        class LogisticRegressionTrainer(BaseTrainer):
            pass

    Args:
        name: Unique identifier for the model
        description: Human-readable description
        framework: ML framework name
        model_type: Type of model

    Returns:
        Decorator function
    """

    def decorator(trainer_class:Type)->Type:
        from mlcli import registry

        registry.register(name=name,trainer_class=trainer_class,
                          description=description,framework=framework,
                          model_type=model_type)
        return trainer_class
    return decorator


