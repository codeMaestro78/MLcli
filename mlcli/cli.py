"""
MLCLI Command Line Interface

Main CLI application with commands for training, evaluation,
model management, hyperparameter tuning, model explainability,
data preprocessing, and interactive UI.
"""

import json
import typer
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from pathlib import Path
from typing import Optional
import pandas as pd

# Initialize Typer app
app = typer.Typer(
    name="mlcli",
    help="Production ML/DL CLI for training, evaluation, hyperparameter tuning, and experiment tracking",
    add_completion=False,
)

console = Console()


def get_registry():
    """Get the model registry with all trainers loaded."""
    from mlcli import registry

    # Import trainers to trigger registration
    from mlcli import trainers  # noqa: F401

    return registry


def get_tracker():
    """Get experiment tracker instance."""
    from mlcli.runner.experiment_tracker import ExperimentTracker

    return ExperimentTracker()


@app.command("train")
def train(
    config: Path = typer.Option(
        ..., "--config", "-c", help="Path to configuration file (JSON or YAML)", exists=True
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output directory for models (overrides config)"
    ),
    run_name: Optional[str] = typer.Option(None, "--name", "-n", help="Name for this training run"),
    epochs: Optional[int] = typer.Option(
        None, "--epochs", "-e", help="Number of epochs (overrides config, for DL models)"
    ),
    batch_size: Optional[int] = typer.Option(
        None, "--batch-size", "-b", help="Batch size (overrides config, for DL models)"
    ),
    verbose: bool = typer.Option(True, "--verbose/--quiet", "-v/-q", help="Verbose output"),
):
    """
    Train a model using configuration file.

    Example:
        mlcli train --config configs/logistic_config.json
        mlcli train -c configs/tf_dnn_config.json --epochs 50 --name "experiment_1"
    """
    from mlcli.config.loader import ConfigLoader
    from mlcli.utils.io import load_data
    from mlcli.utils.logger import setup_logger
    from sklearn.model_selection import train_test_split

    # Setup logger
    log_level = "INFO" if verbose else "WARNING"
    setup_logger("mlcli", level=log_level)

    console.print(Panel.fit("[bold blue]MLCLI Training Pipeline[/bold blue]", border_style="blue"))

    try:
        # Load configuration
        console.print(f"\n[cyan]Loading configuration from:[/cyan] {config}")
        config_loader = ConfigLoader(config)

        # Apply overrides
        if epochs is not None:
            config_loader.set("model.params.epochs", epochs)
            console.print(f"[yellow]Override:[/yellow] epochs = {epochs}")

        if batch_size is not None:
            config_loader.set("model.params.batch_size", batch_size)
            console.print(f"[yellow]Override:[/yellow] batch_size = {batch_size}")

        if output_dir is not None:
            config_loader.set("output.model_dir", str(output_dir))

        # Get model type and framework
        model_type = config_loader.get_model_type()
        registry = get_registry()

        if model_type not in registry:
            console.print(f"[red]Error:[/red] Unknown model type '{model_type}'")
            console.print(f"Available models: {', '.join(registry.list_models())}")
            raise typer.Exit(1)

        metadata = registry.get_metadata(model_type)
        framework = metadata["framework"]

        console.print(f"[green]Model type:[/green] {model_type}")
        console.print(f"[green]Framework:[/green] {framework}")

        # Initialize experiment tracker
        tracker = get_tracker()
        run_id = tracker.start_run(
            model_type=model_type,
            framework=framework,
            config=config_loader.to_dict(),
            run_name=run_name,
        )

        console.print(f"[green]Run ID:[/green] {run_id}")

        # Load dataset
        dataset_config = config_loader.get_dataset_config()
        console.print(f"\n[cyan]Loading dataset from:[/cyan] {dataset_config['path']}")

        X, y = load_data(
            data_path=dataset_config["path"],
            data_type=dataset_config.get("type", "csv"),
            target_column=dataset_config.get("target_column"),
            features=dataset_config.get("features"),
        )

        console.print(
            f"[green]Dataset shape:[/green] X={X.shape}, y={y.shape if y is not None else 'None'}"
        )

        # Train/test split
        training_config = config_loader.get_training_config()
        test_size = training_config.get("test_size", 0.2)
        random_state = training_config.get("random_state", 42)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        console.print(f"[green]Train samples:[/green] {len(X_train)}")
        console.print(f"[green]Test samples:[/green] {len(X_test)}")

        # Create trainer
        console.print("\n[cyan]Initializing trainer...[/cyan]")
        trainer = registry.get_trainer(model_type, config=config_loader.config.get("model", {}))

        # Train model
        console.print("\n[bold cyan]Starting training...[/bold cyan]\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Training model...", total=None)

            training_history = trainer.train(X_train, y_train, X_val=X_test, y_val=y_test)

        # Log training history
        tracker.log_training_history(training_history)

        # Evaluate on test set
        console.print("\n[cyan]Evaluating on test set...[/cyan]")
        test_metrics = trainer.evaluate(X_test, y_test)

        # Log metrics
        tracker.log_metrics(training_history.get("train_metrics", {}), prefix="train_")
        tracker.log_metrics(test_metrics, prefix="test_")

        # Display metrics
        metrics_table = Table(title="Evaluation Metrics", show_header=True)
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="green")

        for metric, value in test_metrics.items():
            if isinstance(value, float):
                metrics_table.add_row(metric, f"{value:.4f}")
            else:
                metrics_table.add_row(metric, str(value))

        console.print(metrics_table)

        # Save model
        output_config = config_loader.get_output_config()
        model_dir = Path(output_config.get("model_dir", "mlcli/models"))
        save_formats = output_config.get("save_format", ["pickle"])

        # Adjust formats based on framework
        if framework == "tensorflow":
            save_formats = ["h5", "savedmodel"]
        elif framework in ["sklearn", "xgboost"]:
            if "pickle" not in save_formats:
                save_formats.append("pickle")

        console.print(f"\n[cyan]Saving model to:[/cyan] {model_dir}")
        console.print(f"[cyan]Formats:[/cyan] {save_formats}")

        saved_paths = trainer.save(model_dir, save_formats)

        # Log model paths
        for fmt, path in saved_paths.items():
            tracker.log_model_path(fmt, path)
            console.print(f"[green]Saved {fmt}:[/green] {path}")

        # End run
        run_data = tracker.end_run(status="completed")

        # Final summary
        console.print(
            Panel.fit(
                f"[bold green]Training Complete![/bold green]\n\n"
                f"Run ID: {run_id}\n"
                f"Test Accuracy: {test_metrics.get('accuracy', 'N/A'):.4f}\n"
                f"Duration: {run_data.get('duration_seconds', 0):.1f}s",
                title="Summary",
                border_style="green",
            )
        )

    except Exception as e:
        console.print(f"\n[red]Error during training:[/red] {str(e)}")

        # End run with error
        try:
            tracker.end_run(status="failed", error=str(e))
        except Exception:
            pass

        if verbose:
            console.print_exception()

        raise typer.Exit(1)


@app.command("tune")
def tune(
    config: Path = typer.Option(
        ..., "--config", "-c", help="Path to tuning configuration file (JSON or YAML)", exists=True
    ),
    method: str = typer.Option(
        "random", "--method", "-m", help="Tuning method: grid, random, or bayesian"
    ),
    n_trials: int = typer.Option(
        50, "--n-trials", "-n", help="Number of trials/iterations for random/bayesian search"
    ),
    cv: int = typer.Option(5, "--cv", help="Number of cross-validation folds"),
    scoring: str = typer.Option(
        "accuracy",
        "--scoring",
        "-s",
        help="Metric to optimize (accuracy, f1, roc_auc, precision, recall)",
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Path to save tuning results (JSON)"
    ),
    train_best: bool = typer.Option(
        False, "--train-best", help="Train a model with the best parameters after tuning"
    ),
    verbose: bool = typer.Option(True, "--verbose/--quiet", "-v/-q", help="Verbose output"),
):
    """
    Tune hyperparameters using Grid Search, Random Search, or Bayesian Optimization.

    Example:
        mlcli tune --config configs/tune_rf_config.json --method random --n-trials 100
        mlcli tune -c configs/tune_logistic.json -m grid --cv 5
        mlcli tune -c configs/tune_xgb.json -m bayesian -n 200 --train-best
    """
    from mlcli.config.loader import ConfigLoader
    from mlcli.utils.io import load_data
    from mlcli.utils.logger import setup_logger
    from mlcli.tuner import get_tuner

    log_level = "INFO" if verbose else "WARNING"
    setup_logger("mlcli", level=log_level)

    console.print(
        Panel.fit(
            "[bold magenta]MLCLI Hyperparameter Tuning[/bold magenta]", border_style="magenta"
        )
    )

    try:
        # Load configuration
        console.print(f"\n[cyan]Loading tuning configuration from:[/cyan] {config}")
        config_loader = ConfigLoader(config)

        # Get model type
        model_type = config_loader.get_model_type()
        registry = get_registry()

        if model_type not in registry:
            console.print(f"[red]Error:[/red] Unknown model type '{model_type}'")
            console.print(f"Available models: {', '.join(registry.list_models())}")
            raise typer.Exit(1)

        metadata = registry.get_metadata(model_type)
        framework = metadata["framework"]

        console.print(f"[green]Model type:[/green] {model_type}")
        console.print(f"[green]Framework:[/green] {framework}")
        console.print(f"[green]Tuning method:[/green] {method}")
        console.print(f"[green]Trials:[/green] {n_trials}")
        console.print(f"[green]CV Folds:[/green] {cv}")
        console.print(f"[green]Scoring:[/green] {scoring}")

        # Load dataset
        dataset_config = config_loader.get_dataset_config()
        console.print(f"\n[cyan]Loading dataset from:[/cyan] {dataset_config['path']}")

        X, y = load_data(
            data_path=dataset_config["path"],
            data_type=dataset_config.get("type", "csv"),
            target_column=dataset_config.get("target_column"),
            features=dataset_config.get("features"),
        )

        console.print(f"[green]Dataset shape:[/green] X={X.shape}, y={y.shape}")

        # Get parameter space from config
        param_space = config_loader.config.get("tuning", {}).get("param_space", {})

        if not param_space:
            console.print("[red]Error:[/red] No param_space defined in tuning configuration")
            console.print("[yellow]Add a 'tuning.param_space' section to your config file[/yellow]")
            raise typer.Exit(1)

        console.print("\n[cyan]Parameter Space:[/cyan]")
        for param, values in param_space.items():
            console.print(f"  {param}: {values}")

        # Get trainer class
        trainer_class = registry.get(model_type)
        base_config = config_loader.config.get("model", {})

        # Create tuner
        console.print(
            f"\n[bold cyan]Starting {method.upper()} hyperparameter tuning...[/bold cyan]\n"
        )

        tuner = get_tuner(
            method=method,
            param_space=param_space,
            n_trials=n_trials,
            scoring=scoring,
            cv=cv,
            verbose=2 if verbose else 0,
            random_state=config_loader.get_training_config().get("random_state", 42),
        )

        # Run tuning
        results = tuner.tune(trainer_class=trainer_class, X=X, y=y, trainer_config=base_config)

        # Display results
        console.print("\n")

        # Best parameters table
        best_params_table = Table(
            title="Best Hyperparameters", show_header=True, header_style="bold green"
        )
        best_params_table.add_column("Parameter", style="cyan")
        best_params_table.add_column("Value", style="green")

        for param, value in results["best_params"].items():
            if isinstance(value, float):
                best_params_table.add_row(param, f"{value:.6f}")
            else:
                best_params_table.add_row(param, str(value))

        console.print(best_params_table)

        # Summary panel
        console.print(
            Panel.fit(
                f"[bold green]Tuning Complete![/bold green]\n\n"
                f"Best Score ({scoring}): {results['best_score']:.4f}\n"
                f"Total Trials: {len(tuner.tuning_history_)}\n"
                f"Duration: {results['duration']:.1f}s",
                title="Summary",
                border_style="green",
            )
        )

        # Top 5 results
        top_results = tuner.get_top_n_params(5)
        if top_results:
            top_table = Table(title="Top 5 Parameter Combinations", show_header=True)
            top_table.add_column("Rank", style="yellow")
            top_table.add_column("Score", style="green")
            top_table.add_column("Parameters", style="cyan")

            for i, result in enumerate(top_results, 1):
                params_str = ", ".join(f"{k}={v}" for k, v in result["params"].items())
                top_table.add_row(
                    str(i),
                    f"{result['score']:.4f}",
                    params_str[:60] + "..." if len(params_str) > 60 else params_str,
                )

            console.print(top_table)

        # Save results
        if output:
            tuner.save_results(output)
            console.print(f"\n[green]Results saved to:[/green] {output}")
        else:
            # Default save location
            default_output = Path("runs") / f"tuning_{model_type}_{method}.json"
            tuner.save_results(default_output)
            console.print(f"\n[green]Results saved to:[/green] {default_output}")

        # Train with best params if requested
        if train_best:
            console.print("\n[bold cyan]Training model with best parameters...[/bold cyan]")

            # Update config with best params
            best_config = {**base_config, "params": results["best_params"]}

            # Create and train
            from sklearn.model_selection import train_test_split

            training_config = config_loader.get_training_config()
            test_size = training_config.get("test_size", 0.2)
            random_state = training_config.get("random_state", 42)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

            trainer = trainer_class(config=best_config)
            trainer.train(X_train, y_train, X_test, y_test)

            test_metrics = trainer.evaluate(X_test, y_test)

            console.print(
                f"\n[green]Final Model Test Accuracy:[/green] {test_metrics.get('accuracy', 0):.4f}"
            )

            # Save model
            output_config = config_loader.get_output_config()
            model_dir = Path(output_config.get("model_dir", "artifacts"))
            save_formats = output_config.get("save_formats", ["pickle"])

            saved_paths = trainer.save(model_dir, save_formats)
            for fmt, path in saved_paths.items():
                console.print(f"[green]Saved {fmt}:[/green] {path}")

    except Exception as e:
        console.print(f"\n[red]Error during tuning:[/red] {str(e)}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command("eval")
def evaluate(
    model_path: Path = typer.Option(
        ..., "--model", "-m", help="Path to saved model file", exists=True
    ),
    data_path: Path = typer.Option(
        ..., "--data", "-d", help="Path to evaluation data", exists=True
    ),
    model_type: str = typer.Option(
        ..., "--type", "-t", help="Model type (e.g., logistic_regression, tf_dnn)"
    ),
    model_format: str = typer.Option(
        "pickle", "--format", "-f", help="Model format (pickle, joblib, h5, savedmodel, onnx)"
    ),
    target_column: Optional[str] = typer.Option(
        None, "--target", help="Target column name in data"
    ),
    verbose: bool = typer.Option(True, "--verbose/--quiet", "-v/-q", help="Verbose output"),
):
    """
    Evaluate a saved model on a dataset.

    Example:
        mlcli eval --model models/logistic_model.pkl --data test.csv --type logistic_regression
        mlcli eval -m models/dnn_model.h5 -d test.csv -t tf_dnn -f h5
    """
    from mlcli.utils.io import load_data
    from mlcli.utils.logger import setup_logger

    log_level = "INFO" if verbose else "WARNING"
    setup_logger("mlcli", level=log_level)

    console.print(
        Panel.fit("[bold blue]MLCLI Evaluation Pipeline[/bold blue]", border_style="blue")
    )

    try:
        # Get registry and validate model type
        registry = get_registry()

        if model_type not in registry:
            console.print(f"[red]Error:[/red] Unknown model type '{model_type}'")
            console.print(f"Available models: {', '.join(registry.list_models())}")
            raise typer.Exit(1)

        console.print(f"[green]Model type:[/green] {model_type}")
        console.print(f"[green]Model path:[/green] {model_path}")
        console.print(f"[green]Model format:[/green] {model_format}")

        # Load data
        console.print(f"\n[cyan]Loading data from:[/cyan] {data_path}")

        X, y = load_data(data_path=data_path, data_type="csv", target_column=target_column)

        console.print(f"[green]Data shape:[/green] X={X.shape}")

        if y is None:
            console.print(
                "[yellow]Warning:[/yellow] No target column specified, cannot compute metrics"
            )
            raise typer.Exit(1)

        # Create and load trainer
        console.print("\n[cyan]Loading model...[/cyan]")
        trainer = registry.get_trainer(model_type, config={})
        trainer.load(model_path, model_format)

        # Evaluate
        console.print("\n[cyan]Evaluating model...[/cyan]")
        metrics = trainer.evaluate(X, y)

        # Display metrics
        metrics_table = Table(title="Evaluation Results", show_header=True)
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="green")

        for metric, value in metrics.items():
            if isinstance(value, float):
                metrics_table.add_row(metric, f"{value:.4f}")
            else:
                metrics_table.add_row(metric, str(value))

        console.print(metrics_table)

        console.print("\n[bold green]Evaluation complete![/bold green]")

    except Exception as e:
        console.print(f"\n[red]Error during evaluation:[/red] {str(e)}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command("list-models")
def list_models(
    framework: Optional[str] = typer.Option(
        None, "--framework", "-f", help="Filter by framework (sklearn, tensorflow, xgboost)"
    )
):
    """
    List all available model trainers in the registry.

    Example:
        mlcli list-models
        mlcli list-models --framework sklearn
    """
    registry = get_registry()

    console.print(Panel.fit("[bold blue]Available Model Trainers[/bold blue]", border_style="blue"))

    # Get models to display
    if framework:
        models = registry.get_models_by_framework(framework)
        if not models:
            console.print(f"[yellow]No models found for framework '{framework}'[/yellow]")
            return
    else:
        models = registry.list_models()

    # Create table
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Name", style="green")
    table.add_column("Framework", style="yellow")
    table.add_column("Type", style="magenta")
    table.add_column("Description")

    for model_name in models:
        metadata = registry.get_metadata(model_name)
        if metadata:
            table.add_row(
                model_name,
                metadata.get("framework", "unknown"),
                metadata.get("model_type", "unknown"),
                metadata.get("description", ""),
            )

    console.print(table)
    console.print(f"\n[dim]Total: {len(models)} models[/dim]")


@app.command("list-tuners")
def list_tuners():
    """
    List all available hyperparameter tuning methods.

    Example:
        mlcli list-tuners
    """
    from mlcli.tuner import TunerFactory

    console.print(
        Panel.fit("[bold magenta]Available Tuning Methods[/bold magenta]", border_style="magenta")
    )

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Method", style="green")
    table.add_column("Name", style="yellow")
    table.add_column("Best For", style="magenta")
    table.add_column("Description")

    for method in TunerFactory.list_methods():
        info = TunerFactory.get_method_info(method)
        table.add_row(
            method, info.get("name", method), info.get("best_for", ""), info.get("description", "")
        )

    console.print(table)

    console.print("\n[dim]Usage: mlcli tune --config <config.json> --method <method>[/dim]")


@app.command("list-runs")
def list_runs(
    n: int = typer.Option(10, "--last", "-n", help="Number of recent runs to show"),
    model_type: Optional[str] = typer.Option(None, "--model", "-m", help="Filter by model type"),
    framework: Optional[str] = typer.Option(None, "--framework", "-f", help="Filter by framework"),
):
    """
    List experiment runs from the tracker.

    Example:
        mlcli list-runs
        mlcli list-runs --last 20
        mlcli list-runs --model tf_dnn
    """
    tracker = get_tracker()

    console.print(Panel.fit("[bold blue]Experiment Runs[/bold blue]", border_style="blue"))

    # Get runs
    if model_type:
        runs = tracker.get_runs_by_model(model_type)
    elif framework:
        runs = tracker.get_runs_by_framework(framework)
    else:
        runs = tracker.get_recent_runs(n)

    if not runs:
        console.print("[yellow]No experiment runs found.[/yellow]")
        return

    # Create table
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Run ID", style="green")
    table.add_column("Name", style="white")
    table.add_column("Model", style="yellow")
    table.add_column("Status", style="magenta")
    table.add_column("Accuracy", style="cyan")
    table.add_column("Timestamp")

    for run in runs[:n]:
        accuracy = run.get("metrics", {}).get(
            "test_accuracy", run.get("metrics", {}).get("accuracy", "N/A")
        )

        if isinstance(accuracy, float):
            accuracy = f"{accuracy:.4f}"

        status_color = {"completed": "green", "failed": "red", "running": "yellow"}.get(
            run["status"], "white"
        )

        table.add_row(
            run["run_id"],
            run["run_name"][:20],
            run["model_type"],
            f"[{status_color}]{run['status']}[/{status_color}]",
            str(accuracy),
            run["timestamp"][:19],
        )

    console.print(table)
    console.print(f"\n[dim]Showing {min(len(runs), n)} of {len(tracker)} total runs[/dim]")


@app.command("show-run")
def show_run(run_id: str = typer.Argument(..., help="Run ID to display")):
    """
    Show detailed information about a specific run.

    Example:
        mlcli show-run abc123
    """
    tracker = get_tracker()

    run = tracker.get_run(run_id)

    if not run:
        console.print(f"[red]Run '{run_id}' not found.[/red]")
        raise typer.Exit(1)

    summary = tracker.get_run_summary(run_id)
    console.print(summary)


@app.command("delete-run")
def delete_run(
    run_id: str = typer.Argument(..., help="Run ID to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """
    Delete an experiment run.

    Example:
        mlcli delete-run abc123
        mlcli delete-run abc123 --force
    """
    tracker = get_tracker()

    run = tracker.get_run(run_id)

    if not run:
        console.print(f"[red]Run '{run_id}' not found.[/red]")
        raise typer.Exit(1)

    if not force:
        confirm = typer.confirm(f"Delete run '{run_id}' ({run['run_name']})?")
        if not confirm:
            console.print("[yellow]Cancelled.[/yellow]")
            return

    tracker.delete_run(run_id)
    console.print(f"[green]Deleted run '{run_id}'[/green]")


@app.command("export-runs")
def export_runs(
    output: Path = typer.Option("experiments.csv", "--output", "-o", help="Output CSV file path")
):
    """
    Export experiment runs to CSV file.

    Example:
        mlcli export-runs
        mlcli export-runs --output my_experiments.csv
    """
    tracker = get_tracker()

    if len(tracker) == 0:
        console.print("[yellow]No experiments to export.[/yellow]")
        return

    tracker.export_to_csv(str(output))
    console.print(f"[green]Exported {len(tracker)} runs to {output}[/green]")


@app.command("explain")
def explain(
    model_path: Path = typer.Option(
        ..., "--model", "-m", help="Path to saved model file", exists=True
    ),
    data_path: Path = typer.Option(
        ..., "--data", "-d", help="Path to data for explanation", exists=True
    ),
    model_type: str = typer.Option(
        ..., "--type", "-t", help="Model type (e.g., logistic_regression, random_forest, xgboost)"
    ),
    method: str = typer.Option("shap", "--method", "-e", help="Explanation method: shap or lime"),
    model_format: str = typer.Option(
        "pickle", "--format", "-f", help="Model format (pickle, joblib)"
    ),
    target_column: Optional[str] = typer.Option(
        None, "--target", help="Target column name in data"
    ),
    num_samples: int = typer.Option(
        100, "--num-samples", "-n", help="Number of samples to explain"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output path for explanation results (JSON)"
    ),
    plot: bool = typer.Option(True, "--plot/--no-plot", help="Generate explanation plot"),
    plot_output: Optional[Path] = typer.Option(
        None, "--plot-output", "-p", help="Output path for plot (PNG)"
    ),
    verbose: bool = typer.Option(True, "--verbose/--quiet", "-v/-q", help="Verbose output"),
):
    """
    Explain model predictions using SHAP or LIME.

    Example:
        mlcli explain --model models/rf_model.pkl --data data/train.csv --type random_forest
        mlcli explain -m models/xgb_model.pkl -d data/test.csv -t xgboost --method lime
        mlcli explain -m models/logistic_model.pkl -d data/train.csv -t logistic_regression -e shap --plot-output shap_plot.png
    """
    from mlcli.utils.io import load_data
    from mlcli.utils.logger import setup_logger
    from mlcli.explainer import ExplainerFactory

    log_level = "INFO" if verbose else "WARNING"
    setup_logger("mlcli", level=log_level)

    console.print(
        Panel.fit("[bold cyan]MLCLI Model Explainability[/bold cyan]", border_style="cyan")
    )

    try:
        # Validate model type
        registry = get_registry()

        if model_type not in registry:
            console.print(f"[red]Error:[/red] Unknown model type '{model_type}'")
            console.print(f"Available models: {', '.join(registry.list_models())}")
            raise typer.Exit(1)

        metadata = registry.get_metadata(model_type)
        framework = metadata["framework"]

        # Check for TensorFlow models - not fully supported yet
        if framework == "tensorflow":
            console.print(
                "[yellow]Warning:[/yellow] SHAP/LIME support for TensorFlow models is limited."
            )
            console.print(
                "[yellow]Results may vary. Consider using Tree-based models for best explanations.[/yellow]"
            )

        console.print(f"[green]Model type:[/green] {model_type}")
        console.print(f"[green]Framework:[/green] {framework}")
        console.print(f"[green]Explanation method:[/green] {method.upper()}")

        # Load data
        console.print(f"\n[cyan]Loading data from:[/cyan] {data_path}")

        X, y = load_data(data_path=data_path, data_type="csv", target_column=target_column)

        # Store feature names if available
        import pandas as pd

        df = pd.read_csv(data_path)
        if target_column and target_column in df.columns:
            feature_names = [col for col in df.columns if col != target_column]
        else:
            feature_names = df.columns.tolist()

        console.print(f"[green]Data shape:[/green] X={X.shape}")
        console.print(f"[green]Features:[/green] {len(feature_names)}")

        # Load model
        console.print(f"\n[cyan]Loading model from:[/cyan] {model_path}")
        trainer = registry.get_trainer(model_type, config={})
        trainer.load(model_path, model_format)

        # Get the underlying model
        model = trainer.model

        console.print("[green]Model loaded successfully[/green]")

        # Get class names if classification
        class_names = None
        if y is not None:
            unique_classes = np.unique(y)
            class_names = [str(c) for c in unique_classes]

        # Create explainer
        console.print(f"\n[bold cyan]Creating {method.upper()} explainer...[/bold cyan]")

        explainer = ExplainerFactory.create(
            method=method, model=model, feature_names=feature_names, class_names=class_names
        )

        # Generate explanations
        console.print(
            f"\n[cyan]Generating explanations for {min(num_samples, len(X))} samples...[/cyan]"
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task(f"Computing {method.upper()} values...", total=None)

            explanation = explainer.explain(
                X=X[:num_samples],
                X_background=X,  # Use all data as background
                max_samples=num_samples,
            )

        # Display results
        console.print("\n")

        # Feature importance table
        importance_table = Table(
            title=f"Feature Importance ({method.upper()})",
            show_header=True,
            header_style="bold cyan",
        )
        importance_table.add_column("Rank", style="yellow", width=6)
        importance_table.add_column("Feature", style="green")
        importance_table.add_column("Importance", style="cyan", justify="right")

        importance = explanation.get("feature_importance", {})
        for i, (feature, value) in enumerate(list(importance.items())[:15], 1):
            # Handle both scalar and array values
            if isinstance(value, (list, np.ndarray)):
                value = np.mean(value)
            importance_table.add_row(str(i), feature[:40], f"{value:.6f}")

        console.print(importance_table)

        # Summary panel
        console.print(
            Panel.fit(
                f"[bold green]Explanation Complete![/bold green]\n\n"
                f"Method: {method.upper()}\n"
                f"Samples Analyzed: {explanation.get('n_samples', num_samples)}\n"
                f"Features: {explanation.get('n_features', len(feature_names))}\n"
                f"Top Feature: {list(importance.keys())[0] if importance else 'N/A'}",
                title="Summary",
                border_style="green",
            )
        )

        # Save explanation results
        if output:
            explainer.save_explanation(output, format="json")
            console.print(f"\n[green]Explanation saved to:[/green] {output}")
        else:
            default_output = Path("runs") / f"explanation_{model_type}_{method}.json"
            explainer.save_explanation(default_output, format="json")
            console.print(f"\n[green]Explanation saved to:[/green] {default_output}")

        # Generate plot
        if plot:
            if plot_output:
                plot_path = plot_output
            else:
                plot_path = Path("runs") / f"explanation_{model_type}_{method}_plot.png"

            console.print("\n[cyan]Generating plot...[/cyan]")

            try:
                explainer.plot(
                    plot_type="bar", output_path=plot_path, max_display=15, X=X[:num_samples]
                )
                console.print(f"[green]Plot saved to:[/green] {plot_path}")
            except Exception as e:
                console.print(f"[yellow]Warning:[/yellow] Could not generate plot: {e}")

        # Print summary text
        console.print(f"\n{explainer.get_summary_text()}")

    except Exception as e:
        console.print(f"\n[red]Error during explanation:[/red] {str(e)}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command("explain-instance")
def explain_instance(
    model_path: Path = typer.Option(
        ..., "--model", "-m", help="Path to saved model file", exists=True
    ),
    data_path: Path = typer.Option(..., "--data", "-d", help="Path to data file", exists=True),
    model_type: str = typer.Option(
        ..., "--type", "-t", help="Model type (e.g., logistic_regression, random_forest)"
    ),
    instance_idx: int = typer.Option(0, "--instance", "-i", help="Index of instance to explain"),
    method: str = typer.Option("shap", "--method", "-e", help="Explanation method: shap or lime"),
    model_format: str = typer.Option(
        "pickle", "--format", "-f", help="Model format (pickle, joblib)"
    ),
    target_column: Optional[str] = typer.Option(
        None, "--target", help="Target column name in data"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output path for explanation (JSON)"
    ),
    verbose: bool = typer.Option(True, "--verbose/--quiet", "-v/-q", help="Verbose output"),
):
    """
    Explain a single prediction using SHAP or LIME.

    Example:
        mlcli explain-instance --model models/rf_model.pkl --data data/test.csv --type random_forest --instance 0
        mlcli explain-instance -m models/xgb_model.pkl -d data/test.csv -t xgboost -i 5 -e lime
    """
    from mlcli.utils.io import load_data
    from mlcli.utils.logger import setup_logger
    from mlcli.explainer import ExplainerFactory

    log_level = "INFO" if verbose else "WARNING"
    setup_logger("mlcli", level=log_level)

    console.print(
        Panel.fit("[bold cyan]MLCLI Instance Explanation[/bold cyan]", border_style="cyan")
    )

    try:
        # Validate model type
        registry = get_registry()

        if model_type not in registry:
            console.print(f"[red]Error:[/red] Unknown model type '{model_type}'")
            raise typer.Exit(1)

        console.print(f"[green]Model type:[/green] {model_type}")
        console.print(f"[green]Method:[/green] {method.upper()}")
        console.print(f"[green]Instance:[/green] {instance_idx}")

        # Load data
        console.print(f"\n[cyan]Loading data from:[/cyan] {data_path}")

        X, y = load_data(data_path=data_path, data_type="csv", target_column=target_column)

        import pandas as pd

        df = pd.read_csv(data_path)
        if target_column and target_column in df.columns:
            feature_names = [col for col in df.columns if col != target_column]
        else:
            feature_names = df.columns.tolist()

        if instance_idx >= len(X):
            console.print(
                f"[red]Error:[/red] Instance {instance_idx} out of range (max: {len(X)-1})"
            )
            raise typer.Exit(1)

        # Load model
        console.print("\n[cyan]Loading model...[/cyan]")
        trainer = registry.get_trainer(model_type, config={})
        trainer.load(model_path, model_format)
        model = trainer.model

        # Create explainer
        explainer = ExplainerFactory.create(method=method, model=model, feature_names=feature_names)

        # Get instance explanation
        instance = X[instance_idx]

        console.print(f"\n[cyan]Explaining instance {instance_idx}...[/cyan]")

        explanation = explainer.explain_instance(instance=instance, X_background=X)

        # Display instance values
        console.print("\n[bold]Instance Feature Values:[/bold]")
        instance_table = Table(show_header=True, header_style="bold cyan")
        instance_table.add_column("Feature", style="green")
        instance_table.add_column("Value", style="yellow", justify="right")

        for feat, val in zip(feature_names[:10], instance[:10]):
            instance_table.add_row(feat, f"{val:.4f}" if isinstance(val, float) else str(val))

        if len(feature_names) > 10:
            instance_table.add_row("...", "...")

        console.print(instance_table)

        # Display contributions
        console.print("\n[bold]Feature Contributions:[/bold]")
        contrib_table = Table(show_header=True, header_style="bold cyan")
        contrib_table.add_column("Feature", style="green")
        contrib_table.add_column("Contribution", style="cyan", justify="right")
        contrib_table.add_column("Direction", style="magenta")

        contributions = explanation.get("feature_contributions", {})
        for feat, contrib in list(contributions.items())[:10]:
            direction = "[green]↑ Positive[/green]" if contrib > 0 else "[red]↓ Negative[/red]"
            contrib_table.add_row(feat[:40], f"{contrib:.6f}", direction)

        console.print(contrib_table)

        # Prediction info
        if y is not None:
            actual = y[instance_idx]
            predicted = trainer.predict(instance.reshape(1, -1))[0]

            console.print(
                Panel.fit(
                    f"[bold]Prediction Info[/bold]\n\n"
                    f"Actual: {actual}\n"
                    f"Predicted: {predicted}",
                    border_style="blue",
                )
            )

        # Save explanation
        if output:
            with open(output, "w") as f:
                json.dump(explanation, f, indent=2, default=str)
            console.print(f"\n[green]Explanation saved to:[/green] {output}")

        console.print("\n[bold green]Instance explanation complete![/bold green]")

    except Exception as e:
        console.print(f"\n[red]Error:[/red] {str(e)}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command("list-explainers")
def list_explainers():
    """
    List all available explanation methods.

    Example:
        mlcli list-explainers
    """
    from mlcli.explainer import ExplainerFactory

    console.print(
        Panel.fit("[bold cyan]Available Explanation Methods[/bold cyan]", border_style="cyan")
    )

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Method", style="green")
    table.add_column("Full Name", style="yellow")
    table.add_column("Best For", style="magenta")
    table.add_column("Description")

    for method in ExplainerFactory.list_methods():
        info = ExplainerFactory.get_method_info(method)
        table.add_row(
            method,
            info.get("full_name", method),
            info.get("best_for", ""),
            info.get("description", ""),
        )

    console.print(table)

    console.print(
        "\n[dim]Usage: mlcli explain --model <model.pkl> --data <data.csv> --type <model_type> --method <method>[/dim]"
    )
    console.print(
        "[dim]       mlcli explain-instance --model <model.pkl> --data <data.csv> --type <model_type> --instance <idx>[/dim]"
    )


@app.command("preprocess")
def preprocess(
    data_path: Path = typer.Option(
        ..., "--data", "-d", help="Path to input data file (CSV)", exists=True
    ),
    output: Path = typer.Option(..., "--output", "-o", help="Path to save preprocessed data (CSV)"),
    method: str = typer.Option(
        "standard_scaler",
        "--method",
        "-m",
        help="Preprocessing method (standard_scaler, minmax_scaler, robust_scaler, normalizer, select_k_best, rfe, variance_threshold)",
    ),
    target_column: Optional[str] = typer.Option(
        None, "--target", "-t", help="Target column name (required for feature selection)"
    ),
    columns: Optional[str] = typer.Option(
        None,
        "--columns",
        "-c",
        help="Comma-separated list of columns to preprocess (default: all numeric)",
    ),
    k: int = typer.Option(10, "--k", help="Number of features for SelectKBest or RFE"),
    threshold: float = typer.Option(
        0.0, "--threshold", help="Variance threshold for VarianceThreshold"
    ),
    norm: str = typer.Option("l2", "--norm", help="Norm type for Normalizer (l1, l2, max)"),
    feature_range_min: float = typer.Option(
        0.0, "--range-min", help="Min value for MinMaxScaler range"
    ),
    feature_range_max: float = typer.Option(
        1.0, "--range-max", help="Max value for MinMaxScaler range"
    ),
    save_preprocessor: Optional[Path] = typer.Option(
        None, "--save-preprocessor", "-s", help="Path to save fitted preprocessor (pickle)"
    ),
    verbose: bool = typer.Option(True, "--verbose/--quiet", "-v/-q", help="Verbose output"),
):
    """
    Preprocess data using various scaling, normalization, or feature selection methods.

    Example:
        mlcli preprocess --data data/train.csv --output data/train_scaled.csv --method standard_scaler
        mlcli preprocess -d data/train.csv -o data/train_norm.csv -m minmax_scaler --range-min 0 --range-max 1
        mlcli preprocess -d data/train.csv -o data/train_selected.csv -m select_k_best --target label --k 10
        mlcli preprocess -d data/train.csv -o data/train_rfe.csv -m rfe --target label --k 15
    """
    from mlcli.utils.logger import setup_logger
    from mlcli.preprocessor import PreprocessorFactory

    log_level = "INFO" if verbose else "WARNING"
    setup_logger("mlcli", level=log_level)

    console.print(
        Panel.fit("[bold green]MLCLI Data Preprocessing[/bold green]", border_style="green")
    )

    try:
        # Load data
        console.print(f"\n[cyan]Loading data from:[/cyan] {data_path}")
        df = pd.read_csv(data_path)
        console.print(f"[green]Data shape:[/green] {df.shape}")
        console.print(f"[green]Columns:[/green] {list(df.columns)}")

        # Determine columns to preprocess
        if columns:
            preprocess_cols = [c.strip() for c in columns.split(",")]
        else:
            # Default to all numeric columns except target
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_column and target_column in numeric_cols:
                numeric_cols.remove(target_column)
            preprocess_cols = numeric_cols

        console.print(f"[green]Preprocessing columns:[/green] {preprocess_cols}")

        # Extract data
        X = df[preprocess_cols].values
        y = df[target_column].values if target_column else None

        # Feature names
        feature_names = preprocess_cols

        # Create preprocessor with appropriate params
        console.print(f"\n[cyan]Creating preprocessor:[/cyan] {method}")

        kwargs = {}
        if method == "normalizer" or method.endswith("_normalizer"):
            kwargs["norm"] = norm
        elif method == "minmax_scaler":
            kwargs["feature_range"] = (feature_range_min, feature_range_max)
        elif method == "select_k_best":
            kwargs["k"] = k
        elif method == "rfe":
            kwargs["n_features_to_select"] = k
        elif method == "variance_threshold":
            kwargs["threshold"] = threshold

        preprocessor = PreprocessorFactory.create(method, **kwargs)
        preprocessor.set_feature_names(feature_names)

        # Check if target is required
        if method in ["select_k_best", "rfe"] and y is None:
            console.print(f"[red]Error:[/red] {method} requires --target column")
            raise typer.Exit(1)

        # Fit and transform
        console.print("\n[bold cyan]Fitting and transforming data...[/bold cyan]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task(f"Applying {method}...", total=None)
            X_transformed = preprocessor.fit_transform(X, y)

        console.print(f"[green]Transformed shape:[/green] {X_transformed.shape}")

        # Get output feature names
        output_feature_names = preprocessor.get_feature_names_out() or feature_names
        if len(output_feature_names) != X_transformed.shape[1]:
            output_feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

        # Create output dataframe
        df_transformed = pd.DataFrame(X_transformed, columns=output_feature_names)

        # Add target column back if present
        if target_column and target_column in df.columns:
            df_transformed[target_column] = df[target_column].values

        # Add non-preprocessed columns back
        for col in df.columns:
            if (
                col not in preprocess_cols
                and col != target_column
                and col not in df_transformed.columns
            ):
                df_transformed[col] = df[col].values

        # Save preprocessed data
        output.parent.mkdir(parents=True, exist_ok=True)
        df_transformed.to_csv(output, index=False)
        console.print(f"\n[green]Preprocessed data saved to:[/green] {output}")

        # Display preprocessing info
        info_table = Table(
            title="Preprocessing Summary", show_header=True, header_style="bold green"
        )
        info_table.add_column("Property", style="cyan")
        info_table.add_column("Value", style="green")

        info_table.add_row("Method", method)
        info_table.add_row("Input Shape", f"{X.shape}")
        info_table.add_row("Output Shape", f"{X_transformed.shape}")
        info_table.add_row("Features In", str(len(feature_names)))
        info_table.add_row("Features Out", str(len(output_feature_names)))

        # Add method-specific info
        params = preprocessor.get_params()
        if "mean" in params:
            info_table.add_row("Mean (first 3)", str(params["mean"][:3]))
        if "scale" in params:
            info_table.add_row("Scale (first 3)", str(params["scale"][:3]))
        if "selected_features" in params:
            info_table.add_row("Selected Features", str(len(params["selected_features"])))

        console.print(info_table)

        # Save preprocessor if requested
        if save_preprocessor:
            preprocessor.save(save_preprocessor)
            console.print(f"[green]Preprocessor saved to:[/green] {save_preprocessor}")

        # Summary panel
        console.print(
            Panel.fit(
                f"[bold green]Preprocessing Complete![/bold green]\n\n"
                f"Method: {method}\n"
                f"Input: {X.shape} → Output: {X_transformed.shape}\n"
                f"Output: {output}",
                title="Summary",
                border_style="green",
            )
        )

    except Exception as e:
        console.print(f"\n[red]Error during preprocessing:[/red] {str(e)}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command("preprocess-pipeline")
def preprocess_pipeline(
    data_path: Path = typer.Option(
        ..., "--data", "-d", help="Path to input data file (CSV)", exists=True
    ),
    output: Path = typer.Option(..., "--output", "-o", help="Path to save preprocessed data (CSV)"),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to pipeline config file (JSON/YAML)"
    ),
    steps: Optional[str] = typer.Option(
        None,
        "--steps",
        "-s",
        help="Comma-separated preprocessing steps (e.g., 'standard_scaler,select_k_best')",
    ),
    target_column: Optional[str] = typer.Option(None, "--target", "-t", help="Target column name"),
    save_pipeline: Optional[Path] = typer.Option(
        None, "--save-pipeline", "-p", help="Path to save fitted pipeline (pickle)"
    ),
    verbose: bool = typer.Option(True, "--verbose/--quiet", "-v/-q", help="Verbose output"),
):
    """
    Apply a preprocessing pipeline with multiple steps.

    Example:
        mlcli preprocess-pipeline --data data/train.csv --output data/processed.csv --steps "standard_scaler,select_k_best" --target label
        mlcli preprocess-pipeline -d data/train.csv -o data/processed.csv -c pipeline_config.json
    """
    from mlcli.utils.logger import setup_logger
    from mlcli.preprocessor import PreprocessingPipeline

    log_level = "INFO" if verbose else "WARNING"
    setup_logger("mlcli", level=log_level)

    console.print(
        Panel.fit("[bold green]MLCLI Preprocessing Pipeline[/bold green]", border_style="green")
    )

    try:
        # Load data
        console.print(f"\n[cyan]Loading data from:[/cyan] {data_path}")
        df = pd.read_csv(data_path)
        console.print(f"[green]Data shape:[/green] {df.shape}")

        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_column and target_column in numeric_cols:
            numeric_cols.remove(target_column)

        X = df[numeric_cols].values
        y = df[target_column].values if target_column else None

        # Create pipeline
        pipeline = PreprocessingPipeline()

        if config:
            # Load from config file
            from mlcli.config.loader import ConfigLoader

            config_loader = ConfigLoader(config)
            pipeline_config = config_loader.config.get("preprocessing", {})
            pipeline = PreprocessingPipeline.from_config(pipeline_config)
            console.print(f"[green]Loaded pipeline from config:[/green] {config}")
        elif steps:
            # Create from steps string
            step_list = [s.strip() for s in steps.split(",")]
            for step in step_list:
                pipeline.add_preprocessor(step)
            console.print(f"[green]Created pipeline with steps:[/green] {step_list}")
        else:
            console.print("[red]Error:[/red] Either --config or --steps is required")
            raise typer.Exit(1)

        console.print(f"[green]Pipeline steps:[/green] {len(pipeline)}")

        # Fit and transform
        console.print("\n[bold cyan]Running preprocessing pipeline...[/bold cyan]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Processing...", total=None)
            X_transformed = pipeline.fit_transform(X, y, feature_names=numeric_cols)

        console.print(f"[green]Transformed shape:[/green] {X_transformed.shape}")

        # Get output feature names
        output_feature_names = pipeline.get_feature_names_out() or [
            f"feature_{i}" for i in range(X_transformed.shape[1])
        ]

        # Create output dataframe
        df_transformed = pd.DataFrame(X_transformed, columns=output_feature_names)

        # Add target column back
        if target_column and target_column in df.columns:
            df_transformed[target_column] = df[target_column].values

        # Save preprocessed data
        output.parent.mkdir(parents=True, exist_ok=True)
        df_transformed.to_csv(output, index=False)
        console.print(f"\n[green]Preprocessed data saved to:[/green] {output}")

        # Display pipeline info
        info_table = Table(title="Pipeline Summary", show_header=True, header_style="bold green")
        info_table.add_column("Step", style="cyan")
        info_table.add_column("Type", style="green")
        info_table.add_column("Status", style="yellow")

        for name, preprocessor in pipeline.steps:
            info_table.add_row(
                name,
                preprocessor.__class__.__name__,
                "[green]✓ Fitted[/green]" if preprocessor.is_fitted else "[red]✗ Not Fitted[/red]",
            )

        console.print(info_table)

        # Save pipeline if requested
        if save_pipeline:
            pipeline.save(save_pipeline)
            console.print(f"[green]Pipeline saved to:[/green] {save_pipeline}")

        console.print(
            Panel.fit(
                f"[bold green]Pipeline Complete![/bold green]\n\n"
                f"Steps: {len(pipeline)}\n"
                f"Input: {X.shape} → Output: {X_transformed.shape}\n"
                f"Output: {output}",
                title="Summary",
                border_style="green",
            )
        )

    except Exception as e:
        console.print(f"\n[red]Error during pipeline execution:[/red] {str(e)}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command("list-preprocessors")
def list_preprocessors(
    category: Optional[str] = typer.Option(
        None,
        "--category",
        "-c",
        help="Filter by category (Scaling, Normalization, Encoding, Feature Selection)",
    )
):
    """
    List all available preprocessing methods.

    Example:
        mlcli list-preprocessors
        mlcli list-preprocessors --category Scaling
    """
    from mlcli.preprocessor import PreprocessorFactory

    console.print(
        Panel.fit("[bold green]Available Preprocessing Methods[/bold green]", border_style="green")
    )

    # Get preprocessors by category
    categories = PreprocessorFactory.list_by_category()

    if category:
        # Filter to specific category
        if category not in categories:
            console.print(f"[red]Unknown category:[/red] {category}")
            console.print(f"[yellow]Available categories:[/yellow] {list(categories.keys())}")
            raise typer.Exit(1)
        categories = {category: categories[category]}

    for cat_name, methods in categories.items():
        table = Table(title=f"{cat_name}", show_header=True, header_style="bold green")
        table.add_column("Method", style="cyan")
        table.add_column("Name", style="yellow")
        table.add_column("Description")

        for method in methods:
            info = PreprocessorFactory.get_method_info(method)
            table.add_row(method, info.get("name", method), info.get("description", ""))

        console.print(table)
        console.print()

    console.print(
        "[dim]Usage: mlcli preprocess --data <data.csv> --output <output.csv> --method <method>[/dim]"
    )
    console.print(
        "[dim]       mlcli preprocess-pipeline --data <data.csv> --output <output.csv> --steps <step1,step2>[/dim]"
    )


@app.command("ui")
def launch_ui():
    """
    Launch interactive terminal UI.

    Example:
        mlcli ui
    """
    console.print(
        Panel.fit("[bold blue]Launching MLCLI Interactive UI...[/bold blue]", border_style="blue")
    )

    try:
        from mlcli.ui.app import MLCLIApp

        app_ui = MLCLIApp()
        app_ui.run()

    except ImportError as e:
        console.print(f"[red]Error:[/red] Could not import UI module: {e}")
        console.print("[yellow]Make sure textual is installed: pip install textual[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error launching UI:[/red] {e}")
        raise typer.Exit(1)


@app.command("version")
def version():
    """Show MLCLI version information."""
    from mlcli import __version__

    console.print(
        Panel.fit(
            f"[bold blue]MLCLI[/bold blue] v{__version__}\n\n"
            f"[dim]Production ML/DL CLI for training, evaluation,\n"
            f"and experiment tracking[/dim]",
            border_style="blue",
        )
    )


@app.callback()
def main():
    """
    MLCLI - Production ML/DL Command Line Interface

    Train, evaluate, and track machine learning models with ease.
    """


if __name__ == "__main__":
    app()
