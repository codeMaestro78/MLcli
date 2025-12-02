"""
Experiment Tracker

Tracks all training runs with metrics, hyperparameters, and model paths.
Provides a mini-MLflow experience with JSON-based storage.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    Tracks and manages ML/DL experiments.

    Stores run information including:
    - Timestamp and run ID
    - Model type and framework
    - Hyperparameters
    - Training and evaluation metrics
    - Model save paths
    - Dataset information
    """

    def __init__(self, log_dir: str = "runs"):
        """
        Initialize experiment tracker.

        Args:
            log_dir: Directory to store experiment logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.experiments_file = self.log_dir / "experiments.json"
        self.experiments: List[Dict[str, Any]] = []

        # Load existing experiments
        self._load_experiments()

        # Current run tracking
        self.current_run: Optional[Dict[str, Any]] = None

        logger.debug(f"Initialized ExperimentTracker with {len(self.experiments)} existing runs")

    def _load_experiments(self) -> None:
        """Load existing experiments from file."""
        if self.experiments_file.exists():
            try:
                with open(self.experiments_file, 'r', encoding='utf-8') as f:
                    self.experiments = json.load(f)
                logger.debug(f"Loaded {len(self.experiments)} experiments")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not load experiments file: {e}")
                self.experiments = []
        else:
            self.experiments = []

    def _save_experiments(self) -> None:
        """Save experiments to file."""
        try:
            with open(self.experiments_file, 'w', encoding='utf-8') as f:
                json.dump(self.experiments, f, indent=2, default=str)
            logger.debug(f"Saved {len(self.experiments)} experiments")
        except IOError as e:
            logger.error(f"Could not save experiments file: {e}")

    def start_run(
        self,
        model_type: str,
        framework: str,
        config: Dict[str, Any],
        run_name: Optional[str] = None
    ) -> str:
        """
        Start a new experiment run.

        Args:
            model_type: Type of model (e.g., 'logistic_regression', 'tf_dnn')
            framework: ML framework (e.g., 'sklearn', 'tensorflow')
            config: Full configuration dictionary
            run_name: Optional human-readable name for the run

        Returns:
            Run ID
        """
        run_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().isoformat()

        self.current_run = {
            "run_id": run_id,
            "run_name": run_name or f"{model_type}_{run_id}",
            "timestamp": timestamp,
            "model_type": model_type,
            "framework": framework,
            "status": "running",
            "config": config,
            "hyperparameters": config.get("model", {}).get("params", {}),
            "dataset": config.get("dataset", {}),
            "metrics": {},
            "model_paths": {},
            "training_history": {},
            "duration_seconds": None,
            "error": None
        }

        self._start_time = datetime.now()

        logger.info(f"Started experiment run: {run_id} ({self.current_run['run_name']})")

        return run_id

    def log_metrics(self, metrics: Dict[str, float], prefix: str = "") -> None:
        """
        Log metrics for current run.

        Args:
            metrics: Dictionary of metric name to value
            prefix: Optional prefix for metric names (e.g., 'train_', 'val_')
        """
        if self.current_run is None:
            logger.warning("No active run. Call start_run() first.")
            return

        for name, value in metrics.items():
            metric_name = f"{prefix}{name}" if prefix else name
            self.current_run["metrics"][metric_name] = float(value)

        logger.debug(f"Logged {len(metrics)} metrics with prefix '{prefix}'")

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log additional parameters for current run.

        Args:
            params: Dictionary of parameter name to value
        """
        if self.current_run is None:
            logger.warning("No active run. Call start_run() first.")
            return

        self.current_run["hyperparameters"].update(params)
        logger.debug(f"Logged {len(params)} parameters")

    def log_model_path(self, format_name: str, path: Path) -> None:
        """
        Log saved model path.

        Args:
            format_name: Model format (e.g., 'pickle', 'h5')
            path: Path to saved model
        """
        if self.current_run is None:
            logger.warning("No active run. Call start_run() first.")
            return

        self.current_run["model_paths"][format_name] = str(path)
        logger.debug(f"Logged model path: {format_name} -> {path}")

    def log_training_history(self, history: Dict[str, Any]) -> None:
        """
        Log training history.

        Args:
            history: Training history dictionary
        """
        if self.current_run is None:
            logger.warning("No active run. Call start_run() first.")
            return

        self.current_run["training_history"] = history
        logger.debug("Logged training history")

    def log_artifact(self, name: str, value: Any) -> None:
        """
        Log arbitrary artifact.

        Args:
            name: Artifact name
            value: Artifact value (must be JSON serializable)
        """
        if self.current_run is None:
            logger.warning("No active run. Call start_run() first.")
            return

        if "artifacts" not in self.current_run:
            self.current_run["artifacts"] = {}

        self.current_run["artifacts"][name] = value
        logger.debug(f"Logged artifact: {name}")

    def end_run(self, status: str = "completed", error: Optional[str] = None) -> Dict[str, Any]:
        """
        End current run and save to experiments file.

        Args:
            status: Run status ('completed', 'failed', 'cancelled')
            error: Error message if status is 'failed'

        Returns:
            Completed run data
        """
        if self.current_run is None:
            logger.warning("No active run to end.")
            return {}

        # Calculate duration
        if hasattr(self, '_start_time'):
            duration = (datetime.now() - self._start_time).total_seconds()
            self.current_run["duration_seconds"] = round(duration, 2)

        self.current_run["status"] = status

        if error:
            self.current_run["error"] = error

        # Add to experiments list
        self.experiments.append(self.current_run)

        # Save to file
        self._save_experiments()

        completed_run = self.current_run

        logger.info(
            f"Ended experiment run: {completed_run['run_id']} "
            f"(status={status}, duration={completed_run.get('duration_seconds', 0):.1f}s)"
        )

        # Reset current run
        self.current_run = None

        return completed_run

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific run by ID.

        Args:
            run_id: Run identifier

        Returns:
            Run data or None if not found
        """
        for run in self.experiments:
            if run["run_id"] == run_id:
                return run
        return None

    def get_all_runs(self) -> List[Dict[str, Any]]:
        """
        Get all experiment runs.

        Returns:
            List of all runs
        """
        return self.experiments.copy()

    def get_runs_by_model(self, model_type: str) -> List[Dict[str, Any]]:
        """
        Get runs for a specific model type.

        Args:
            model_type: Model type to filter by

        Returns:
            List of matching runs
        """
        return [run for run in self.experiments if run["model_type"] == model_type]

    def get_runs_by_framework(self, framework: str) -> List[Dict[str, Any]]:
        """
        Get runs for a specific framework.

        Args:
            framework: Framework to filter by

        Returns:
            List of matching runs
        """
        return [run for run in self.experiments if run["framework"] == framework]

    def get_best_run(self, metric: str = "accuracy", higher_is_better: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get the best run based on a metric.

        Args:
            metric: Metric name to compare
            higher_is_better: Whether higher values are better

        Returns:
            Best run or None if no runs
        """
        if not self.experiments:
            return None

        valid_runs = [
            run for run in self.experiments
            if metric in run.get("metrics", {})
        ]

        if not valid_runs:
            return None

        return max(valid_runs, key=lambda r: r["metrics"][metric] * (1 if higher_is_better else -1))

    def get_recent_runs(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get most recent runs.

        Args:
            n: Number of runs to return

        Returns:
            List of recent runs
        """
        sorted_runs = sorted(
            self.experiments,
            key=lambda r: r["timestamp"],
            reverse=True
        )
        return sorted_runs[:n]

    def delete_run(self, run_id: str) -> bool:
        """
        Delete a run by ID.

        Args:
            run_id: Run identifier

        Returns:
            True if deleted, False if not found
        """
        for i, run in enumerate(self.experiments):
            if run["run_id"] == run_id:
                del self.experiments[i]
                self._save_experiments()
                logger.info(f"Deleted run: {run_id}")
                return True
        return False

    def clear_all_runs(self) -> int:
        """
        Clear all experiment runs.

        Returns:
            Number of runs cleared
        """
        count = len(self.experiments)
        self.experiments = []
        self._save_experiments()
        logger.info(f"Cleared {count} experiment runs")
        return count

    def get_run_summary(self, run_id: str) -> str:
        """
        Get human-readable summary of a run.

        Args:
            run_id: Run identifier

        Returns:
            Formatted summary string
        """
        run = self.get_run(run_id)

        if not run:
            return f"Run {run_id} not found."

        lines = [
            f"{'='*60}",
            f"Run: {run['run_name']} ({run['run_id']})",
            f"{'='*60}",
            f"Timestamp:  {run['timestamp']}",
            f"Model:      {run['model_type']}",
            f"Framework:  {run['framework']}",
            f"Status:     {run['status']}",
            f"Duration:   {run.get('duration_seconds', 'N/A')}s",
            "",
            "Hyperparameters:",
            "-" * 40
        ]

        for param, value in run.get("hyperparameters", {}).items():
            lines.append(f"  {param}: {value}")

        lines.extend([
            "",
            "Metrics:",
            "-" * 40
        ])

        for metric, value in run.get("metrics", {}).items():
            if isinstance(value, float):
                lines.append(f"  {metric}: {value:.4f}")
            else:
                lines.append(f"  {metric}: {value}")

        if run.get("model_paths"):
            lines.extend([
                "",
                "Model Paths:",
                "-" * 40
            ])
            for fmt, path in run["model_paths"].items():
                lines.append(f"  {fmt}: {path}")

        lines.append(f"{'='*60}")

        return "\n".join(lines)

    def export_to_csv(self, output_path: str) -> None:
        """
        Export experiments to CSV file.

        Args:
            output_path: Path to output CSV file
        """
        import csv

        if not self.experiments:
            logger.warning("No experiments to export")
            return

        # Flatten experiments for CSV
        rows = []
        for run in self.experiments:
            row = {
                "run_id": run["run_id"],
                "run_name": run["run_name"],
                "timestamp": run["timestamp"],
                "model_type": run["model_type"],
                "framework": run["framework"],
                "status": run["status"],
                "duration_seconds": run.get("duration_seconds", "")
            }

            # Add metrics
            for metric, value in run.get("metrics", {}).items():
                row[f"metric_{metric}"] = value

            rows.append(row)

        # Get all columns
        all_columns = set()
        for row in rows:
            all_columns.update(row.keys())

        columns = sorted(all_columns)

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

        logger.info(f"Exported {len(rows)} experiments to {output_path}")

    def __len__(self) -> int:
        """Return number of experiments."""
        return len(self.experiments)

    def __repr__(self) -> str:
        """String representation."""
        return f"ExperimentTracker(runs={len(self.experiments)}, log_dir='{self.log_dir}')"
