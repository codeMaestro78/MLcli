"""
Model Registry System

Provides decorator-based auto-registration for all trainer classes,
enabling dynamic model discovery and instantiation from configuration.
"""

from typing import Dict, Type, Optional, List, Any
import logging

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Central registry for all model trainers.

    Maps model type strings (e.g., 'logistic_regression') to their corresponding
    trainer class implementations. Supports automatic registration via decorator.
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._registry :Dict[str,Type]={}
        self._metadata : Dict[str,Dict[str,Any]]={}

    def register(self,name:str,trainer_class:Type,description:str="",framework:str="unknown",model_type:str="unknown")->None:
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

        self._registry[name]= trainer_class
        self._metadata[name]= {
            "description":description,
            "framework":framework,
            "model_type":model_type,
            "class_name":trainer_class.__name__
        }

        logger.debug(f"Registered model:{name}->{trainer_class.__name__}")

    def get(self,name:str)->Optional[Type]:
        """
        Retrieve a trainer class by name.

        Args:
            name: Model identifier

        Returns:
            Trainer class or None if not found
        """
        return self._registry.get(name)

    def get_trainer(self,name:str,**kwargs)->Any:
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
        trainer_class=self.get(name)
        if trainer_class is None:
            available=", ".join(self.list_models())
            raise KeyError(f"Model '{name}' not found in registry." f"Available models: {available}")

        return trainer_class(**kwargs)

    def list_models(self)->List[str]:
        """
        Get list of all registered model names.

        Returns:
            List of model identifiers
        """
        return sorted(self._registry.keys())


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

    def is_registered(self,name:str)->bool:
        """
        Check if a model is registered.

        Args:
            name: Model identifier

        Returns:
            True if registered, False otherwise
        """
        return name in self._registry

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


