"""
LIME Explainer

LIME (Local Interpretable Model-agnostic Explanations) implementation.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import logging

from mlcli.explainer.base_explainer import BaseExplainer

logger = logging.getLogger(__name__)


class LIMEExplainer(BaseExplainer):
    """
    LIME-based model explainer.
    
    Uses LIME to explain individual predictions by fitting
    interpretable local models.
    """
    
    def __init__(
        self,
        model: Any,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        mode: str = "classification"
    ):
        """
        Initialize LIME explainer.
        
        Args:
            model: Trained model
            feature_names: Names of features
            class_names: Names of target classes
            mode: Either 'classification' or 'regression'
        """
        super().__init__(model, feature_names, class_names)
        self.mode = mode
        self.lime_explainer = None
        self.instance_explanations = []
    
    def _create_explainer(
        self,
        X_train: np.ndarray,
        categorical_features: Optional[List[int]] = None
    ) -> None:
        """
        Create LIME explainer.
        
        Args:
            X_train: Training data for LIME explainer
            categorical_features: Indices of categorical features
        """
        from lime.lime_tabular import LimeTabularExplainer
        
        feature_names = self.feature_names or [
            f"feature_{i}" for i in range(X_train.shape[1])
        ]
        
        self.lime_explainer = LimeTabularExplainer(
            training_data=X_train,
            feature_names=feature_names,
            class_names=self.class_names,
            categorical_features=categorical_features,
            mode=self.mode,
            verbose=False
        )
        
        logger.info(f"Created LIME {self.mode} explainer")
    
    def explain(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        X_train: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        num_features: int = 10,
        num_samples: int = 5000,
        max_instances: int = 100,
        categorical_features: Optional[List[int]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate LIME explanations for multiple instances.
        
        Args:
            X: Input data to explain
            X_train: Training data for LIME explainer
            num_features: Number of features in explanation
            num_samples: Number of samples for LIME
            max_instances: Maximum instances to explain
            categorical_features: Indices of categorical features
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing LIME explanations
        """
        X = self._validate_input(X)
        
        if X_train is not None:
            X_train = self._validate_input(X_train)
        else:
            X_train = X
        
        # Create explainer if not exists
        if self.lime_explainer is None:
            self._create_explainer(X_train, categorical_features)
        
        # Limit instances
        if len(X) > max_instances:
            indices = np.random.choice(len(X), max_instances, replace=False)
            X = X[indices]
        
        logger.info(f"Computing LIME explanations for {len(X)} instances...")
        
        predict_fn = self._get_predict_function()
        
        # Aggregate feature importance across all instances
        feature_importance_sum = {}
        self.instance_explanations = []
        
        for i, instance in enumerate(X):
            exp = self.lime_explainer.explain_instance(
                instance,
                predict_fn,
                num_features=num_features,
                num_samples=num_samples
            )
            
            instance_exp = {
                'instance_idx': i,
                'features': dict(exp.as_list()),
                'score': exp.score if hasattr(exp, 'score') else None,
                'local_pred': exp.local_pred[0] if hasattr(exp, 'local_pred') else None
            }
            self.instance_explanations.append(instance_exp)
            
            # Aggregate importance
            for feature, importance in exp.as_list():
                # Extract feature name (LIME returns conditions like "feature > 5")
                feature_name = feature.split()[0] if ' ' in feature else feature
                
                if feature_name not in feature_importance_sum:
                    feature_importance_sum[feature_name] = []
                feature_importance_sum[feature_name].append(abs(importance))
        
        # Average importance across instances
        feature_importance = {
            k: np.mean(v) for k, v in feature_importance_sum.items()
        }
        sorted_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )
        
        self.explanations = {
            "method": "lime",
            "mode": self.mode,
            "n_instances": len(X),
            "n_features": X.shape[1],
            "num_features_explained": num_features,
            "num_samples_per_instance": num_samples,
            "feature_importance": sorted_importance,
            "top_features": list(sorted_importance.keys())[:10]
        }
        
        logger.info("LIME explanation complete")
        
        return self.explanations
    
    def explain_instance(
        self,
        instance: Union[np.ndarray, pd.Series],
        X_train: Optional[np.ndarray] = None,
        num_features: int = 10,
        num_samples: int = 5000,
        categorical_features: Optional[List[int]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate LIME explanation for a single instance.
        
        Args:
            instance: Single data point to explain
            X_train: Training data for LIME explainer
            num_features: Number of features in explanation
            num_samples: Number of samples for LIME
            categorical_features: Indices of categorical features
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing instance explanation
        """
        if isinstance(instance, pd.Series):
            instance = instance.values
        
        instance = np.asarray(instance).flatten()
        
        # Support X_background as alias for X_train
        if X_train is None:
            X_train = kwargs.get('X_background')
        
        if self.lime_explainer is None:
            if X_train is None:
                raise ValueError("X_train or X_background required for first explanation")
            X_train = self._validate_input(X_train)
            self._create_explainer(X_train, categorical_features)
        
        predict_fn = self._get_predict_function()
        
        exp = self.lime_explainer.explain_instance(
            instance,
            predict_fn,
            num_features=num_features,
            num_samples=num_samples
        )
        
        # Get feature contributions
        contributions = dict(exp.as_list())
        
        # Separate positive and negative contributions
        positive_features = {k: v for k, v in contributions.items() if v > 0}
        negative_features = {k: v for k, v in contributions.items() if v < 0}
        
        # Sort by absolute importance
        sorted_contributions = dict(
            sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        )
        
        result = {
            "method": "lime",
            "instance_values": instance.tolist(),
            "feature_contributions": sorted_contributions,
            "positive_contributions": dict(
                sorted(positive_features.items(), key=lambda x: x[1], reverse=True)
            ),
            "negative_contributions": dict(
                sorted(negative_features.items(), key=lambda x: abs(x[1]), reverse=True)
            ),
            "local_model_score": exp.score if hasattr(exp, 'score') else None
        }
        
        # Add prediction info
        if hasattr(exp, 'local_pred'):
            result['local_prediction'] = exp.local_pred[0]
        
        if self.mode == "classification" and hasattr(exp, 'predict_proba'):
            result['predicted_probabilities'] = exp.predict_proba.tolist()
        
        return result
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get global feature importance from LIME explanations.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.explanations is None:
            raise RuntimeError("Run explain() first to compute feature importance")
        
        return self.explanations.get("feature_importance", {})
    
    def plot(
        self,
        plot_type: str = "bar",
        output_path: Optional[Path] = None,
        max_display: int = 20,
        instance_idx: int = 0,
        **kwargs
    ) -> Optional[str]:
        """
        Generate LIME plots.
        
        Args:
            plot_type: Type of plot ('bar', 'instance')
            output_path: Path to save the plot
            max_display: Maximum features to display
            instance_idx: Index of instance for instance plot
            **kwargs: Additional plot arguments
            
        Returns:
            Path to saved plot or None
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        if self.explanations is None:
            raise RuntimeError("Run explain() first to generate plots")
        
        plt.figure(figsize=(12, 8))
        
        if plot_type == "bar":
            # Global feature importance bar plot
            importance = self.get_feature_importance()
            features = list(importance.keys())[:max_display]
            values = [importance[f] for f in features]
            
            colors = ['steelblue' if v >= 0 else 'coral' for v in values]
            
            plt.barh(features[::-1], values[::-1], color=colors[::-1])
            plt.xlabel('Mean |LIME Weight|')
            plt.title('Feature Importance (LIME)')
            plt.tight_layout()
        
        elif plot_type == "instance":
            # Single instance explanation
            if instance_idx >= len(self.instance_explanations):
                raise ValueError(f"Instance {instance_idx} not found")
            
            instance_exp = self.instance_explanations[instance_idx]
            contributions = instance_exp['features']
            
            features = list(contributions.keys())[:max_display]
            values = [contributions[f] for f in features]
            
            colors = ['green' if v >= 0 else 'red' for v in values]
            
            plt.barh(features[::-1], values[::-1], color=colors[::-1])
            plt.xlabel('LIME Weight')
            plt.title(f'LIME Explanation - Instance {instance_idx}')
            plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            plt.tight_layout()
        
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Plot saved to {output_path}")
            return str(output_path)
        
        plt.close()
        return None
    
    def get_summary_text(self) -> str:
        """
        Get text summary of LIME explanation.
        
        Returns:
            Human-readable summary
        """
        if self.explanations is None:
            return "No explanation computed yet. Run explain() first."
        
        lines = [
            "=" * 60,
            "LIME Explanation Summary",
            "=" * 60,
            f"Mode: {self.explanations['mode']}",
            f"Instances Analyzed: {self.explanations['n_instances']}",
            f"Features per Explanation: {self.explanations['num_features_explained']}",
            f"Samples per Instance: {self.explanations['num_samples_per_instance']}",
            "",
            "Top 10 Most Important Features (Averaged):",
            "-" * 40
        ]
        
        importance = self.explanations['feature_importance']
        for i, (feature, value) in enumerate(list(importance.items())[:10], 1):
            lines.append(f"  {i}. {feature}: {value:.4f}")
        
        lines.extend(["", "=" * 60])
        
        return "\n".join(lines)
