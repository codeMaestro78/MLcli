"""Configuration Loader

This module is responsible for loading configuration files written in JSON or YAML.
It makes sure the configurations are valid, match the expected schema, and include all required fields.
If any values are missing, it automatically fills in the appropriate defaults.
Overall, it ensures that both training and evaluation pipelines always receive clean and reliable configuration settings.
"""

import json
import yaml
from pathlib import Path
from typing import Optional,Dict, Any,Union
import logging

logger=logging.getLogger(__name__)

class ConfigLoader:
    """
    Configuration file loader supporting JSON and YAML formats.

    Handles loading, validation, and merging of configuration files
    with support for defaults and overrides.
    """

    REQUIRED_KEYS=["dataset","model"]
    OPTIONAL_KEYS=["training","evaluation","output"]

    def __init__(self,config_path:Optional[Union[str,Path]]=None):
        """
        Initialize config loader.

        Args:
            config_path: Path to configuration file (optional)
        """
        self.config_path=Path(config_path) if config_path else None
        self.config=Dict[str,Any]={}

    def load(self,config_path:Union[str,Path])->Dict[str,Any]:
        """
        Load configuration from file.

        Args:
            config_path: Path to JSON or YAML config file

        Returns:
            Loaded configuration dictionary

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If file format is unsupported or invalid
        """
        config_path=Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found : {config_path}")
        logger.info(f"Loading configuration from: {config_path}")

        # Determine file type and load

        suffix=config_path.suffix.lower()

        try:
            with open(config_path,"r",encoding="utf-8") as f:
                if suffix == ".json":
                    self.config=json.load(f)
                elif suffix in ['.yaml','.yml']:
                    self.config=yaml.safe_load(f)
                else:
                    raise ValueError(
                        f"Unsupported config format :{suffix}."
                        f"Use .json, .yaml or .yml "
                    )
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {config_path} :{e}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {config_path} : {e}")

        # validate configuration
        self.validate()

        # Apply defaults
        self._apply_defaults()

        logger.info(f"Configuration loaded successfully: model_type={self.get_model_type()}")

        return self.config

    def validate(self)->None:
        """
        Validate loaded configuration.

        Raises:
            ValueError: If configuration is invalid or missing required keys
        """

        if not self.config:
            raise ValueError(f"Configuration is Empty")

        # check required keys
        missing_keys=[key for key in self.REQUIRED_KEYS if key not in self.config]
        if missing_keys:
            raise ValueError(
                f"Missing required configuration keys: {missing_keys}."
                f"Required keys: {self.REQUIRED_KEYS}"
            )
        #  validate dataset section
        if "path" not in self.config["dataset"]:
            raise ValueError(f"'dataset' section must contain 'path key")
        # validate model section
        if "type" not in self.config["model"]:
            raise ValueError(f"'model' section must contain 'type' key")

        logger.debug("Configuration validation passed")

    def _apply_defaults(self)->None:
        """Apply default values for optional sections."""
        # Default training config
        if "training" not in self.config:
            self.config["training"]={}

        training_defaults={
            "test_size":0.2,
            "random_state":42,
            "shuffle":True
        }
        for key ,value in training_defaults.items():
            self.config["training"].setdefault(key,value)

        # Default evaluation config
        if "evaluation" not in self.config:
            self.config["evaluation"]={}

        eval_defaults={
            "metrics":["accuracy","precision","recall","f1"]
        }
        for key, value in eval_defaults.items():
            self.config["evaluation"].setdefault(key,value)

        # Default output config
        if "output" not in self.config:
            self.config["output"]={}

        output_defaults={
            "model_dir":"mlcli/models",
            "save_format":["pickle","onmx"],
            "log_dir":"runs"
        }
        for key,value in output_defaults.items():
            self.config["output"].setdefault(key,value)

    def get(self,key:str,default:Any=None)->Any:
        """
        Get configuration value by key.

        Args:
            key: Configuration key (supports dot notation, e.g., 'model.type')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys=key.split(".")
        value= self.config

        for k in keys:
            if isinstance(value,dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        return value

    def set(self,key:str,value:Any)->None:
        """
        Set configuration value.

        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys=key.split(".")
        config=self.config

        # Navigate to nested key
        for k in keys[:-1]:
            if k not in config:
                config[k]={}
            config=config[k]

        # Set value
        config[keys[-1]]=value
        logger.debug(f"Set config :{key} --> {value}")

    def get_model_type(self)->str:
        """
        Get model type from configuration.

        Returns:
            Model type string
        """
        return self.get("model.params ",{})

    def get_dataset_config(self)->Dict[str,Any]:
        """
        Get dataset configuration.

        Returns:
            Dataset configuration dictionary
        """
        return self.config.get("dataset",{})

    def get_training_config(self)->Dict[str,Any]:
        """
        Get training configuration.

        Returns:
            Training configuration dictionary
        """
        return self.config.get("training",{})

    def get_output_config(self)->Dict[str,Any]:
        """
        Get output configuration.

        Returns:
            Output configuration dictionary
        """
        return self.config.get("output",{})

    def merge(self,overrides:Dict[str,Any])->None:
        """
        Merge override values into configuration.

        Args:
            overrides: Dictionary of values to override
        """
        self._deep_merge(self.config,overrides)
        logger.debug(f"Applied configuration overrides :{list(overrides.keys())}")

    def _deep_merge(self,base:Dict,override:Dict)->Dict:
        """
        Recursively merge two dictionaries.

        Args:
            base: Base dictionary
            override: Override dictionary

        Returns:
            Merged dictionary
        """

        for key,value in override.items():
            if key in base and isinstance(base[key],dict) and isinstance(value,dict):
                self._deep_merge(base[key],value)
            else:
                base[key]=value
        return base


    def save(self,output_path:Union[str,Path])->None:
        """
        Save current configuration to file.

        Args:
            output_path: Path to save configuration
        """

        output_path=Path(output_path)
        suffix=output_path.suffix.lower()

        output_path.mkdir(parents=True,exist_ok=True)

        with open(output_path,"w",encoding="utf-8") as f:
            if suffix == '.json':
                json.dump(self.config,f,indent=2)
            elif suffix  in ['.yaml','.yml']:
                yaml.dump(self.config,f,default_flow_style=False)
            else:
                raise ValueError(f"Unsupported format : {suffix}")

        logger.info(f"Configuration saved to: {output_path}")

    def to_dict(self)->Dict[str,Any]:
        """
        Get configuration as dictionary.

        Returns:
            Configuration dictionary
        """
        return self.config.copy()

    def __repr__(self)->str:
        """String representation"""
        return f"ConfigLoader(model_type='{self.get_model_type()}')"

    def __str__(self)->str:
        """Readable String"""
        return json.dumps(self.config,indent=2)


def load_config(config_path:Union[str,Path])->ConfigLoader:
    """
    Convenience function to load configuration.

    Args:
        config_path: Path to configuration file

    Returns:
        ConfigLoader instance with loaded configuration
    """
    return ConfigLoader(config_path)
