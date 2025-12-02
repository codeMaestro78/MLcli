"""
MLCLI Command Line Interface

Main CLI application with commands for training, evaluation,
model management, and interactive UI.
"""

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint
from pathlib import Path
from typing import Optional, List
import json
import sys

# Initialize Typer app
app = typer.Typer(
    name="mlcli",
    help="Production ML/DL CLI for training, evaluation, and experiment tracking",
    add_completion=False
)

console = Console()


def get_registry():
    """Get the model registry with all trainers loaded."""
    from mlcli import registry
    # Import trainers to trigger registration
    from mlcli import trainers
    return registry


def get_tracker():
    """Get experiment tracker instance."""
    from mlcli.runner.experiment_tracker import ExperimentTracker
    return ExperimentTracker()


@app.command("train")
def train(
    config: Path = typer.Option(
        ...,
        "--config", "-c",
        help="Path to configuration file (JSON or YAML)",
        exists=True
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output directory for models (overrides config)"
    ),
    run_name: Optional[str] = typer.Option(
        None,
        "--name", "-n",
        help="Name for this training run"
    ),
    epochs: Optional[int] = typer.Option(
        None,
        "--epochs", "-e",
        help="Number of epochs (overrides config, for DL models)"
    ),
    batch_size: Optional[int] = typer.Option(
        None,
        "--batch-size", "-b",
        help="Batch size (overrides config, for DL models)"
    ),
    verbose: bool = typer.Option(
        True,
        "--verbose/--quiet", "-v/-q",
        help="Verbose output"
    )
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
    import numpy as np

    # Setup logger
    log_level = "INFO" if verbose else "WARNING"
    setup_logger("mlcli", level=log_level)

    console.print(Panel.fit(
        "[bold blue]MLCLI Training Pipeline[/bold blue]",
        border_style="blue"
    ))

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
            run_name=run_name
        )

        console.print(f"[green]Run ID:[/green] {run_id}")

        # Load dataset
        dataset_config = config_loader.get_dataset_config()
        console.print(f"\n[cyan]Loading dataset from:[/cyan] {dataset_config['path']}")

        X, y = load_data(
            data_path=dataset_config["path"],
            data_type=dataset_config.get("type", "csv"),
            target_column=dataset_config.get("target_column"),
            features=dataset_config.get("features")
        )

        console.print(f"[green]Dataset shape:[/green] X={X.shape}, y={y.shape if y is not None else 'None'}")

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
        console.print(f"\n[cyan]Initializing trainer...[/cyan]")
        trainer = registry.get_trainer(
            model_type,
            config=config_loader.config.get("model", {})
        )

        # Train model
        console.print(f"\n[bold cyan]Starting training...[/bold cyan]\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            progress.add_task("Training model...", total=None)

            training_history = trainer.train(
                X_train, y_train,
                X_val=X_test,
                y_val=y_test
            )

        # Log training history
        tracker.log_training_history(training_history)

        # Evaluate on test set
        console.print(f"\n[cyan]Evaluating on test set...[/cyan]")
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
        console.print(Panel.fit(
            f"[bold green]Training Complete![/bold green]\n\n"
            f"Run ID: {run_id}\n"
            f"Test Accuracy: {test_metrics.get('accuracy', 'N/A'):.4f}\n"
            f"Duration: {run_data.get('duration_seconds', 0):.1f}s",
            title="Summary",
            border_style="green"
        ))

    except Exception as e:
        console.print(f"\n[red]Error during training:[/red] {str(e)}")

        # End run with error
        try:
            tracker.end_run(status="failed", error=str(e))
        except:
            pass

        if verbose:
            console.print_exception()

        raise typer.Exit(1)


@app.command("eval")
def evaluate(
    model_path: Path = typer.Option(
        ...,
        "--model", "-m",
        help="Path to saved model file",
        exists=True
    ),
    data_path: Path = typer.Option(
        ...,
        "--data", "-d",
        help="Path to evaluation data",
        exists=True
    ),
    model_type: str = typer.Option(
        ...,
        "--type", "-t",
        help="Model type (e.g., logistic_regression, tf_dnn)"
    ),
    model_format: str = typer.Option(
        "pickle",
        "--format", "-f",
        help="Model format (pickle, joblib, h5, savedmodel, onnx)"
    ),
    target_column: Optional[str] = typer.Option(
        None,
        "--target",
        help="Target column name in data"
    ),
    verbose: bool = typer.Option(
        True,
        "--verbose/--quiet", "-v/-q",
        help="Verbose output"
    )
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

    console.print(Panel.fit(
        "[bold blue]MLCLI Evaluation Pipeline[/bold blue]",
        border_style="blue"
    ))

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

        X, y = load_data(
            data_path=data_path,
            data_type="csv",
            target_column=target_column
        )

        console.print(f"[green]Data shape:[/green] X={X.shape}")

        if y is None:
            console.print("[yellow]Warning:[/yellow] No target column specified, cannot compute metrics")
            raise typer.Exit(1)

        # Create and load trainer
        console.print(f"\n[cyan]Loading model...[/cyan]")
        trainer = registry.get_trainer(model_type, config={})
        trainer.load(model_path, model_format)

        # Evaluate
        console.print(f"\n[cyan]Evaluating model...[/cyan]")
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
        None,
        "--framework", "-f",
        help="Filter by framework (sklearn, tensorflow, xgboost)"
    )
):
    """
    List all available model trainers in the registry.

    Example:
        mlcli list-models
        mlcli list-models --framework sklearn
    """
    registry = get_registry()

    console.print(Panel.fit(
        "[bold blue]Available Model Trainers[/bold blue]",
        border_style="blue"
    ))

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
                metadata.get("description", "")
            )

    console.print(table)
    console.print(f"\n[dim]Total: {len(models)} models[/dim]")


@app.command("list-runs")
def list_runs(
    n: int = typer.Option(
        10,
        "--last", "-n",
        help="Number of recent runs to show"
    ),
    model_type: Optional[str] = typer.Option(
        None,
        "--model", "-m",
        help="Filter by model type"
    ),
    framework: Optional[str] = typer.Option(
        None,
        "--framework", "-f",
        help="Filter by framework"
    )
):
    """
    List experiment runs from the tracker.

    Example:
        mlcli list-runs
        mlcli list-runs --last 20
        mlcli list-runs --model tf_dnn
    """
    tracker = get_tracker()

    console.print(Panel.fit(
        "[bold blue]Experiment Runs[/bold blue]",
        border_style="blue"
    ))

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
        accuracy = run.get("metrics", {}).get("test_accuracy",
                   run.get("metrics", {}).get("accuracy", "N/A"))

        if isinstance(accuracy, float):
            accuracy = f"{accuracy:.4f}"

        status_color = {
            "completed": "green",
            "failed": "red",
            "running": "yellow"
        }.get(run["status"], "white")

        table.add_row(
            run["run_id"],
            run["run_name"][:20],
            run["model_type"],
            f"[{status_color}]{run['status']}[/{status_color}]",
            str(accuracy),
            run["timestamp"][:19]
        )

    console.print(table)
    console.print(f"\n[dim]Showing {min(len(runs), n)} of {len(tracker)} total runs[/dim]")


@app.command("show-run")
def show_run(
    run_id: str = typer.Argument(
        ...,
        help="Run ID to display"
    )
):
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
    run_id: str = typer.Argument(
        ...,
        help="Run ID to delete"
    ),
    force: bool = typer.Option(
        False,
        "--force", "-f",
        help="Skip confirmation"
    )
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
    output: Path = typer.Option(
        "experiments.csv",
        "--output", "-o",
        help="Output CSV file path"
    )
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


@app.command("ui")
def launch_ui():
    """
    Launch interactive terminal UI.

    Example:
        mlcli ui
    """
    console.print(Panel.fit(
        "[bold blue]Launching MLCLI Interactive UI...[/bold blue]",
        border_style="blue"
    ))

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

    console.print(Panel.fit(
        f"[bold blue]MLCLI[/bold blue] v{__version__}\n\n"
        f"[dim]Production ML/DL CLI for training, evaluation,\n"
        f"and experiment tracking[/dim]",
        border_style="blue"
    ))


@app.callback()
def main():
    """
    MLCLI - Production ML/DL Command Line Interface

    Train, evaluate, and track machine learning models with ease.
    """
    pass


if __name__ == "__main__":
    app()
