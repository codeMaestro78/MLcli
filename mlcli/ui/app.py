"""
MLCLI Interactive Terminal UI

Main Textual application providing interactive menus for
training, evaluation, and experiment management.
"""

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Header,
    Footer,
    Button,
    Static,
    Label,
    DataTable,
    Input,
    Select,
    ListView,
    ListItem,
    Rule,
)
from textual.screen import Screen
from textual.binding import Binding
from pathlib import Path
from typing import Optional, Dict, Any


class WelcomeScreen(Screen):
    """Welcome/Home screen with main menu."""

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("1", "train", "Train"),
        Binding("2", "evaluate", "Evaluate"),
        Binding("3", "experiments", "Experiments"),
        Binding("4", "models", "Models"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Container(
            Static(
                """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   ███╗   ███╗██╗      ██████╗██╗     ██╗                        ║
║   ████╗ ████║██║     ██╔════╝██║     ██║                        ║
║   ██╔████╔██║██║     ██║     ██║     ██║                        ║
║   ██║╚██╔╝██║██║     ██║     ██║     ██║                        ║
║   ██║ ╚═╝ ██║███████╗╚██████╗███████╗██║                        ║
║   ╚═╝     ╚═╝╚══════╝ ╚═════╝╚══════╝╚═╝                        ║
║                                                                  ║
║           Production ML/DL Training & Tracking                   ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
                """,
                id="banner",
                classes="banner",
            ),
            Horizontal(
                Button("🎯 Train Model", id="btn-train", variant="primary"),
                Button("📊 Evaluate Model", id="btn-eval", variant="success"),
                Button("📈 View Experiments", id="btn-experiments", variant="warning"),
                Button("🔧 List Models", id="btn-models", variant="default"),
                id="menu-buttons",
                classes="menu-buttons",
            ),
            Static(
                "\n[dim]Press number keys (1-4) for quick access, or 'q' to quit[/dim]", id="hint"
            ),
            id="welcome-container",
            classes="welcome-container",
        )
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id

        if button_id == "btn-train":
            self.app.push_screen(TrainScreen())
        elif button_id == "btn-eval":
            self.app.push_screen(EvaluateScreen())
        elif button_id == "btn-experiments":
            self.app.push_screen(ExperimentsScreen())
        elif button_id == "btn-models":
            self.app.push_screen(ModelsScreen())

    def action_train(self) -> None:
        self.app.push_screen(TrainScreen())

    def action_evaluate(self) -> None:
        self.app.push_screen(EvaluateScreen())

    def action_experiments(self) -> None:
        self.app.push_screen(ExperimentsScreen())

    def action_models(self) -> None:
        self.app.push_screen(ModelsScreen())

    def action_quit(self) -> None:
        self.app.exit()


class TrainScreen(Screen):
    """Training configuration and execution screen."""

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("enter", "start_training", "Start Training"),
    ]

    def __init__(self):
        super().__init__()
        self.selected_config: Optional[Path] = None
        self.selected_model: Optional[str] = None
        self.config_data: Dict[str, Any] = {}

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield ScrollableContainer(
            Static("🎯 [bold]Train a Model[/bold]", classes="screen-title"),
            Rule(),
            Static("[bold cyan]1. Select Configuration[/bold cyan]"),
            ListView(id="config-list"),
            Rule(),
            Static("[bold cyan]2. Select Model Type (optional)[/bold cyan]"),
            ListView(id="model-list"),
            Rule(),
            Static("[bold cyan]3. Override Parameters (optional)[/bold cyan]"),
            Horizontal(
                Vertical(
                    Label("Epochs:"),
                    Input(placeholder="Leave empty", id="input-epochs"),
                    classes="param-input",
                ),
                Vertical(
                    Label("Batch Size:"),
                    Input(placeholder="Leave empty", id="input-batch"),
                    classes="param-input",
                ),
                id="params-row",
            ),
            Horizontal(
                Vertical(
                    Label("Learning Rate:"),
                    Input(placeholder="Leave empty", id="input-lr"),
                    classes="param-input",
                ),
                Vertical(
                    Label("Run Name:"),
                    Input(placeholder="Optional", id="input-name"),
                    classes="param-input",
                ),
                id="params-row2",
            ),
            Rule(),
            Horizontal(
                Button("▶ Start Training", id="btn-start", variant="success"),
                Button("← Back", id="btn-back", variant="default"),
                id="action-buttons",
            ),
            Static(
                "[dim]Select a config above, then click Start Training[/dim]", id="status-message"
            ),
            id="train-container",
        )
        yield Footer()

    def on_mount(self) -> None:
        """Load configs and models on mount."""
        self._load_configs()
        self._load_models()
        # Auto-select first config for convenience
        self._auto_select_first_config()

    def _auto_select_first_config(self) -> None:
        """Auto-select the first available config."""
        config_dir = Path("configs")
        if config_dir.exists():
            for ext in [".json", ".yaml", ".yml"]:
                configs = sorted(config_dir.glob(f"*{ext}"))
                if configs:
                    self.selected_config = configs[0]
                    self._update_status(
                        f"✓ Auto-selected: [green]{configs[0].name}[/green] | Use ↑↓ to change, then Start Training"
                    )
                    break

    def _load_configs(self) -> None:
        """Load available config files."""
        config_list = self.query_one("#config-list", ListView)
        config_list.clear()

        config_dir = Path("configs")
        if config_dir.exists():
            for config_file in sorted(config_dir.glob("*.json")):
                config_list.append(ListItem(Label(config_file.name), id=f"cfg-{config_file.stem}"))
            for config_file in sorted(config_dir.glob("*.yaml")):
                config_list.append(ListItem(Label(config_file.name), id=f"cfg-{config_file.stem}"))
            for config_file in sorted(config_dir.glob("*.yml")):
                config_list.append(ListItem(Label(config_file.name), id=f"cfg-{config_file.stem}"))

        if config_list.children == []:
            config_list.append(ListItem(Label("[dim]No configs found[/dim]")))

    def _load_models(self) -> None:
        """Load available model types from registry."""
        model_list = self.query_one("#model-list", ListView)
        model_list.clear()

        try:
            from mlcli import registry
            from mlcli import trainers  # noqa: F401

            for model_name in registry.list_models():
                metadata = registry.get_metadata(model_name)
                framework = metadata.get("framework", "unknown") if metadata else "unknown"
                model_list.append(
                    ListItem(Label(f"{model_name} [{framework}]"), id=f"model-{model_name}")
                )
        except Exception as e:
            model_list.append(ListItem(Label(f"[red]Error loading models: {e}[/red]")))

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        """Handle list highlight (cursor movement) - auto-select on highlight."""
        if event.item is None:
            return
        item_id = event.item.id or ""
        self._handle_item_selection(item_id)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle list selection (Enter key)."""
        item_id = event.item.id or ""
        self._handle_item_selection(item_id)

    def _handle_item_selection(self, item_id: str) -> None:
        """Process item selection from either highlight or select event."""
        if item_id.startswith("cfg-"):
            config_name = item_id[4:]
            config_dir = Path("configs")

            for ext in [".json", ".yaml", ".yml"]:
                config_path = config_dir / f"{config_name}{ext}"
                if config_path.exists():
                    self.selected_config = config_path
                    self._update_status(
                        f"✓ Config: [green]{config_path.name}[/green] | Press 'Start Training' or Enter"
                    )
                    break

        elif item_id.startswith("model-"):
            self.selected_model = item_id[6:]
            self._update_status(f"✓ Model: [green]{self.selected_model}[/green]")

    def _update_status(self, message: str) -> None:
        """Update status message."""
        status = self.query_one("#status-message", Static)
        status.update(f"[cyan]{message}[/cyan]")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "btn-back":
            self.app.pop_screen()
        elif event.button.id == "btn-start":
            self._start_training()

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_start_training(self) -> None:
        self._start_training()

    def _start_training(self) -> None:
        """Start the training process."""
        # If no config selected, try to auto-select first available
        if not self.selected_config:
            config_dir = Path("configs")
            if config_dir.exists():
                for ext in [".json", ".yaml", ".yml"]:
                    configs = list(config_dir.glob(f"*{ext}"))
                    if configs:
                        self.selected_config = configs[0]
                        break

        if not self.selected_config:
            self._update_status(
                "[red]❌ No config selected! Use ↑↓ to navigate and highlight a config file.[/red]"
            )
            return

        # Check if config file exists
        if not self.selected_config.exists():
            self._update_status(f"[red]❌ Config file not found: {self.selected_config}[/red]")
            return

        # Get parameter overrides
        epochs_input = self.query_one("#input-epochs", Input)
        batch_input = self.query_one("#input-batch", Input)
        lr_input = self.query_one("#input-lr", Input)
        name_input = self.query_one("#input-name", Input)

        # Build command
        self._update_status(
            f"[yellow]🚀 Starting training with {self.selected_config.name}...[/yellow]"
        )

        # Push to training progress screen
        self.app.push_screen(
            TrainingProgressScreen(
                config_path=self.selected_config,
                epochs=epochs_input.value if epochs_input.value else None,
                batch_size=batch_input.value if batch_input.value else None,
                learning_rate=lr_input.value if lr_input.value else None,
                run_name=name_input.value if name_input.value else None,
            )
        )


class TrainingProgressScreen(Screen):
    """Screen showing training progress and results."""

    BINDINGS = [
        Binding("escape", "back", "Back"),
    ]

    def __init__(
        self,
        config_path: Path,
        epochs: Optional[str] = None,
        batch_size: Optional[str] = None,
        learning_rate: Optional[str] = None,
        run_name: Optional[str] = None,
    ):
        super().__init__()
        self.config_path = config_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.run_name = run_name
        self.training_complete = False
        self.results: Dict[str, Any] = {}

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Container(
            Static("🚀 [bold]Training in Progress[/bold]", id="progress-title"),
            Rule(),
            Static(f"Config: {self.config_path}", id="config-info"),
            Static("", id="progress-status"),
            Rule(),
            Static("[bold cyan]Training Output:[/bold cyan]"),
            ScrollableContainer(Static("", id="training-log"), id="log-container"),
            Rule(),
            Static("[bold cyan]Results:[/bold cyan]"),
            DataTable(id="results-table"),
            Horizontal(
                Button("← Back to Menu", id="btn-back", variant="default"),
                Button(
                    "📈 View in Experiments", id="btn-view-exp", variant="primary", disabled=True
                ),
                id="result-buttons",
            ),
            id="progress-container",
        )
        yield Footer()

    def on_mount(self) -> None:
        """Start training when screen mounts."""
        # Initialize log
        self._log_lines: list = []
        # Start training after a short delay to let UI render
        self.set_timer(0.5, self._run_training)

    def _add_log(self, msg: str) -> None:
        """Add a message to the log."""
        self._log_lines.append(msg)
        try:
            log_widget = self.query_one("#training-log", Static)
            log_widget.update("\n".join(self._log_lines[-30:]))
        except Exception:
            pass

    def _set_status(self, msg: str) -> None:
        """Set status message."""
        try:
            status_widget = self.query_one("#progress-status", Static)
            status_widget.update(msg)
        except Exception:
            pass

    def _run_training(self) -> None:
        """Run training synchronously (called by timer)."""
        tracker = None
        try:
            self._set_status("[yellow]⏳ Initializing...[/yellow]")
            self._add_log("Loading configuration...")
            self.refresh()

            from mlcli.config.loader import ConfigLoader
            from mlcli.utils.io import load_data
            from mlcli import registry
            from mlcli import trainers  # noqa: F401
            from mlcli.runner.experiment_tracker import ExperimentTracker
            from sklearn.model_selection import train_test_split

            # Load config
            config_loader = ConfigLoader(self.config_path)
            self._add_log(f"✓ Loaded config: {self.config_path.name}")
            self.refresh()

            # Apply overrides
            if self.epochs:
                config_loader.set("model.params.epochs", int(self.epochs))
                self._add_log(f"  Override: epochs = {self.epochs}")

            if self.batch_size:
                config_loader.set("model.params.batch_size", int(self.batch_size))
                self._add_log(f"  Override: batch_size = {self.batch_size}")

            if self.learning_rate:
                config_loader.set("model.params.learning_rate", float(self.learning_rate))
                self._add_log(f"  Override: learning_rate = {self.learning_rate}")

            # Get model info
            model_type = config_loader.get_model_type()
            metadata = registry.get_metadata(model_type)
            framework = metadata["framework"] if metadata else "unknown"

            self._add_log(f"✓ Model: {model_type} ({framework})")
            self.refresh()

            # Start tracking
            tracker = ExperimentTracker()
            run_id = tracker.start_run(
                model_type=model_type,
                framework=framework,
                config=config_loader.to_dict(),
                run_name=self.run_name,
            )
            self._add_log(f"✓ Started run: {run_id}")
            self.refresh()

            # Load data
            self._set_status("[yellow]⏳ Loading data...[/yellow]")
            dataset_config = config_loader.get_dataset_config()
            self._add_log(f"Loading data from: {dataset_config['path']}")
            self.refresh()

            X, y = load_data(
                data_path=dataset_config["path"],
                data_type=dataset_config.get("type", "csv"),
                target_column=dataset_config.get("target_column"),
                features=dataset_config.get("features"),
            )
            self._add_log(f"✓ Data loaded: X={X.shape}, y={y.shape}")
            self.refresh()

            # Split data
            training_config = config_loader.get_training_config()
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=training_config.get("test_size", 0.2),
                random_state=training_config.get("random_state", 42),
            )
            self._add_log(f"✓ Split: train={len(X_train)}, test={len(X_test)}")
            self.refresh()

            # Create trainer
            self._set_status("[yellow]⏳ Training model...[/yellow]")
            self._add_log("Initializing trainer...")
            self.refresh()

            trainer_instance = registry.get_trainer(
                model_type, config=config_loader.config.get("model", {})
            )
            self._add_log("✓ Trainer initialized")
            self._add_log("Starting training...")
            self.refresh()

            # Train
            training_history = trainer_instance.train(X_train, y_train, X_val=X_test, y_val=y_test)
            self._add_log("✓ Training complete!")
            self.refresh()

            # Evaluate
            self._set_status("[yellow]⏳ Evaluating...[/yellow]")
            test_metrics = trainer_instance.evaluate(X_test, y_test)
            self._add_log("✓ Evaluation complete")
            self.refresh()

            # Log metrics
            tracker.log_metrics(training_history.get("train_metrics", {}), prefix="train_")
            tracker.log_metrics(test_metrics, prefix="test_")
            tracker.log_training_history(training_history)

            # Save model
            self._set_status("[yellow]⏳ Saving model...[/yellow]")
            output_config = config_loader.get_output_config()
            model_dir = Path(output_config.get("model_dir", "mlcli/models"))

            if framework == "tensorflow":
                save_formats = ["h5", "savedmodel"]
            else:
                save_formats = ["pickle"]

            saved_paths = trainer_instance.save(model_dir, save_formats)

            for fmt, path in saved_paths.items():
                tracker.log_model_path(fmt, path)
                self._add_log(f"✓ Saved {fmt}: {path}")
            self.refresh()

            # End run
            run_data = tracker.end_run(status="completed")

            # Display results
            self._set_status("[green]✅ Training Complete![/green]")
            self._add_log(f"✓ Run completed in {run_data.get('duration_seconds', 0):.1f}s")

            # Show results in table
            results_table = self.query_one("#results-table", DataTable)
            results_table.add_columns("Metric", "Value")
            for metric, value in test_metrics.items():
                if isinstance(value, float):
                    results_table.add_row(metric, f"{value:.4f}")
                else:
                    results_table.add_row(metric, str(value))
            results_table.add_row("Run ID", run_id)
            results_table.add_row("Duration", f"{run_data.get('duration_seconds', 0):.1f}s")

            self.results = {"run_id": run_id, "metrics": test_metrics}
            self.training_complete = True

            # Enable view experiments button
            view_btn = self.query_one("#btn-view-exp", Button)
            view_btn.disabled = False
            self.refresh()

        except Exception as e:
            error_msg = str(e)
            self._set_status(f"[red]❌ Error: {error_msg}[/red]")
            self._add_log(f"[red]ERROR: {error_msg}[/red]")
            self.refresh()

            if tracker:
                try:
                    tracker.end_run(status="failed", error=error_msg)
                except Exception:
                    pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "btn-back":
            # Go back to welcome screen
            self.app.pop_screen()
            self.app.pop_screen()
        elif event.button.id == "btn-view-exp":
            self.app.pop_screen()
            self.app.pop_screen()
            self.app.push_screen(ExperimentsScreen())

    def action_back(self) -> None:
        self.app.pop_screen()


class EvaluateScreen(Screen):
    """Model evaluation screen."""

    BINDINGS = [
        Binding("escape", "back", "Back"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield ScrollableContainer(
            Static("📊 [bold]Evaluate a Model[/bold]", classes="screen-title"),
            Rule(),
            Label("Model Path:"),
            Input(
                placeholder="e.g., mlcli/models/rf_model.pkl",
                id="input-model-path",
            ),
            Label("Data Path:"),
            Input(placeholder="e.g., data/sample_data.csv", id="input-data-path"),
            Label("Target Column:"),
            Input(placeholder="e.g., target", id="input-target"),
            Label("Model Type:"),
            Select(options=[], id="select-model-type", prompt="Select model type"),
            Label("Model Format:"),
            Select(
                options=[
                    ("pickle", "pickle"),
                    ("joblib", "joblib"),
                    ("h5", "h5"),
                    ("savedmodel", "savedmodel"),
                    ("onnx", "onnx"),
                ],
                id="select-format",
                prompt="Select format",
            ),
            Rule(),
            Horizontal(
                Button("▶ Evaluate", id="btn-evaluate", variant="success"),
                Button("← Back", id="btn-back", variant="default"),
                id="action-buttons",
            ),
            Static("", id="eval-status"),
            Rule(),
            Static("[bold cyan]Results:[/bold cyan]"),
            DataTable(id="eval-results"),
            id="eval-container",
        )
        yield Footer()

    def on_mount(self) -> None:
        """Load model types on mount."""
        select = self.query_one("#select-model-type", Select)

        try:
            from mlcli import registry
            from mlcli import trainers  # noqa: F401

            options = [(name, name) for name in registry.list_models()]
            select.set_options(options)
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "btn-back":
            self.app.pop_screen()
        elif event.button.id == "btn-evaluate":
            self._run_evaluation()

    def action_back(self) -> None:
        self.app.pop_screen()

    def _run_evaluation(self) -> None:
        """Run model evaluation."""
        model_path = self.query_one("#input-model-path", Input).value
        data_path = self.query_one("#input-data-path", Input).value
        model_type = self.query_one("#select-model-type", Select).value
        model_format = self.query_one("#select-format", Select).value
        target_col = self.query_one("#input-target", Input).value

        status = self.query_one("#eval-status", Static)
        results_table = self.query_one("#eval-results", DataTable)

        # Clear previous results
        results_table.clear(columns=True)

        # Validate inputs
        if not model_path or not data_path:
            status.update("[red]Please provide model path and data path[/red]")
            return

        if not model_type or model_type == Select.BLANK:
            status.update("[red]Please select a model type[/red]")
            return

        if not model_format or model_format == Select.BLANK:
            status.update("[red]Please select a model format[/red]")
            return

        try:
            status.update("[yellow]⏳ Loading model and data...[/yellow]")

            from mlcli import registry
            from mlcli import trainers  # noqa: F401
            from mlcli.utils.io import load_data
            from pathlib import Path

            # Load data
            X, y = load_data(
                data_path=data_path,
                data_type="csv",
                target_column=target_col if target_col else None,
            )

            if y is None:
                status.update("[red]Target column required for evaluation[/red]")
                return

            # Load model
            trainer = registry.get_trainer(model_type, config={})
            trainer.load(Path(model_path), model_format)

            status.update("[yellow]⏳ Evaluating...[/yellow]")

            # Evaluate
            metrics = trainer.evaluate(X, y)

            # Display results
            results_table.add_columns("Metric", "Value")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    results_table.add_row(metric, f"{value:.4f}")
                else:
                    results_table.add_row(metric, str(value))

            status.update("[green]✅ Evaluation complete![/green]")

        except Exception as e:
            status.update(f"[red]Error: {str(e)}[/red]")


class ExperimentsScreen(Screen):
    """Experiment history viewer screen."""

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("r", "refresh", "Refresh"),
        Binding("d", "delete", "Delete"),
    ]

    def __init__(self):
        super().__init__()
        self.selected_run_id: Optional[str] = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Container(
            Static("📈 [bold]Experiment History[/bold]", classes="screen-title"),
            Rule(),
            DataTable(id="experiments-table", cursor_type="row"),
            Rule(),
            Static("[bold cyan]Run Details:[/bold cyan]"),
            ScrollableContainer(
                Static("Select a run to view details", id="run-details"), id="details-container"
            ),
            Horizontal(
                Button("🔄 Refresh", id="btn-refresh", variant="default"),
                Button("🗑️ Delete Selected", id="btn-delete", variant="error"),
                Button("📥 Export CSV", id="btn-export", variant="primary"),
                Button("← Back", id="btn-back", variant="default"),
                id="action-buttons",
            ),
            id="experiments-container",
        )
        yield Footer()

    def on_mount(self) -> None:
        """Load experiments on mount."""
        self._load_experiments()

    def _load_experiments(self) -> None:
        """Load and display experiments."""
        table = self.query_one("#experiments-table", DataTable)
        table.clear(columns=True)

        table.add_columns("Run ID", "Name", "Model", "Framework", "Accuracy", "Status", "Timestamp")

        try:
            from mlcli.runner.experiment_tracker import ExperimentTracker

            tracker = ExperimentTracker()

            runs = tracker.get_recent_runs(50)

            for run in runs:
                accuracy = run.get("metrics", {}).get(
                    "test_accuracy", run.get("metrics", {}).get("accuracy", "N/A")
                )

                if isinstance(accuracy, float):
                    accuracy = f"{accuracy:.4f}"

                table.add_row(
                    run["run_id"],
                    run["run_name"][:25],
                    run["model_type"],
                    run["framework"],
                    str(accuracy),
                    run["status"],
                    run["timestamp"][:19],
                )
        except Exception as e:
            table.add_row("Error", str(e), "", "", "", "", "")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection."""
        table = self.query_one("#experiments-table", DataTable)
        row_key = event.row_key

        if row_key:
            row_data = table.get_row(row_key)
            if row_data:
                self.selected_run_id = row_data[0]
                self._show_run_details(self.selected_run_id)

    def _show_run_details(self, run_id: str) -> None:
        """Display details for selected run."""
        details = self.query_one("#run-details", Static)

        try:
            from mlcli.runner.experiment_tracker import ExperimentTracker

            tracker = ExperimentTracker()

            summary = tracker.get_run_summary(run_id)
            details.update(summary)
        except Exception as e:
            details.update(f"[red]Error loading details: {e}[/red]")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "btn-back":
            self.app.pop_screen()
        elif event.button.id == "btn-refresh":
            self._load_experiments()
        elif event.button.id == "btn-delete":
            self._delete_selected()
        elif event.button.id == "btn-export":
            self._export_csv()

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_refresh(self) -> None:
        self._load_experiments()

    def action_delete(self) -> None:
        self._delete_selected()

    def _delete_selected(self) -> None:
        """Delete selected run."""
        if not self.selected_run_id:
            return

        try:
            from mlcli.runner.experiment_tracker import ExperimentTracker

            tracker = ExperimentTracker()
            tracker.delete_run(self.selected_run_id)
            self._load_experiments()

            details = self.query_one("#run-details", Static)
            details.update("[green]Run deleted[/green]")
            self.selected_run_id = None
        except Exception as e:
            details = self.query_one("#run-details", Static)
            details.update(f"[red]Error deleting: {e}[/red]")

    def _export_csv(self) -> None:
        """Export experiments to CSV."""
        try:
            from mlcli.runner.experiment_tracker import ExperimentTracker

            tracker = ExperimentTracker()
            tracker.export_to_csv("experiments_export.csv")

            details = self.query_one("#run-details", Static)
            details.update("[green]Exported to experiments_export.csv[/green]")
        except Exception as e:
            details = self.query_one("#run-details", Static)
            details.update(f"[red]Error exporting: {e}[/red]")


class ModelsScreen(Screen):
    """Available models viewer screen."""

    BINDINGS = [
        Binding("escape", "back", "Back"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Container(
            Static("🔧 [bold]Available Model Trainers[/bold]", classes="screen-title"),
            Rule(),
            DataTable(id="models-table", cursor_type="row"),
            Rule(),
            Static("[bold cyan]Model Details:[/bold cyan]"),
            ScrollableContainer(
                Static("Select a model to view details", id="model-details"), id="details-container"
            ),
            Horizontal(Button("← Back", id="btn-back", variant="default"), id="action-buttons"),
            id="models-container",
        )
        yield Footer()

    def on_mount(self) -> None:
        """Load models on mount."""
        self._load_models()

    def _load_models(self) -> None:
        """Load and display available models."""
        table = self.query_one("#models-table", DataTable)
        table.clear(columns=True)

        table.add_columns("Name", "Framework", "Type", "Description")

        try:
            from mlcli import registry
            from mlcli import trainers  # noqa: F401

            for model_name in registry.list_models():
                metadata = registry.get_metadata(model_name)
                if metadata:
                    table.add_row(
                        model_name,
                        metadata.get("framework", "unknown"),
                        metadata.get("model_type", "unknown"),
                        metadata.get("description", "")[:50],
                    )
        except Exception as e:
            table.add_row("Error", str(e), "", "")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection."""
        table = self.query_one("#models-table", DataTable)
        row_key = event.row_key

        if row_key:
            row_data = table.get_row(row_key)
            if row_data:
                model_name = row_data[0]
                self._show_model_details(model_name)

    def _show_model_details(self, model_name: str) -> None:
        """Display details for selected model."""
        details = self.query_one("#model-details", Static)

        try:
            from mlcli import registry
            from mlcli import trainers  # noqa: F401

            metadata = registry.get_metadata(model_name)
            trainer_class = registry.get(model_name)

            if trainer_class:
                default_params = trainer_class.get_default_params()

                lines = [
                    f"[bold]{model_name}[/bold]",
                    "",
                    f"Framework: {metadata.get('framework', 'unknown')}",
                    f"Type: {metadata.get('model_type', 'unknown')}",
                    f"Class: {metadata.get('class_name', 'unknown')}",
                    "",
                    "[bold]Description:[/bold]",
                    metadata.get("description", "No description"),
                    "",
                    "[bold]Default Parameters:[/bold]",
                ]

                for param, value in default_params.items():
                    lines.append(f"  {param}: {value}")

                details.update("\n".join(lines))
            else:
                details.update(f"[red]Model {model_name} not found[/red]")

        except Exception as e:
            details.update(f"[red]Error loading details: {e}[/red]")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "btn-back":
            self.app.pop_screen()

    def action_back(self) -> None:
        self.app.pop_screen()


class MLCLIApp(App):
    """Main MLCLI Textual Application."""

    TITLE = "MLCLI - ML/DL Training & Tracking"
    SUB_TITLE = "Interactive Terminal UI"

    CSS = """
    Screen {
        overflow: auto;
    }

    .banner {
        text-align: center;
        color: $primary;
        padding: 1;
    }

    .welcome-container {
        align: center middle;
        padding: 1;
        height: auto;
    }

    .menu-buttons {
        align: center middle;
        padding: 1;
        width: 100%;
        height: auto;
    }

    .menu-buttons Button {
        margin: 1 2;
        min-width: 16;
    }

    .screen-title {
        text-align: center;
        padding: 1;
        text-style: bold;
    }

    .section {
        width: 1fr;
        height: 12;
        border: solid $primary;
        padding: 1;
        margin: 0 1;
    }

    #selection-row {
        height: 14;
        width: 100%;
    }

    #params-row {
        height: auto;
        padding: 1;
    }

    .param-input {
        width: 1fr;
        padding: 0 1;
    }

    .input-group {
        padding: 1;
    }

    .select-group {
        width: 50%;
        padding: 1;
    }

    #selects-row {
        height: auto;
    }

    #action-buttons {
        align: center middle;
        padding: 1;
        height: auto;
    }

    #action-buttons Button {
        margin: 0 1;
    }

    #train-container {
        padding: 1;
    }

    #train-container ListView {
        height: 8;
        border: solid $primary;
        margin: 0 0 1 0;
    }

    #eval-container {
        padding: 1;
    }

    #eval-container Input {
        margin: 0 0 1 0;
    }

    #eval-container Select {
        margin: 0 0 1 0;
    }

    #log-container {
        height: 12;
        border: solid $secondary;
        padding: 1;
    }

    #details-container {
        height: 15;
        border: solid $secondary;
        padding: 1;
    }

    DataTable {
        height: 10;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("h", "home", "Home", show=True),
    ]

    def on_mount(self) -> None:
        """Push the welcome screen on startup."""
        self.push_screen(WelcomeScreen())

    def action_home(self) -> None:
        """Go to home screen."""
        # Pop all screens except the base, then push welcome
        while len(self.screen_stack) > 1:
            self.pop_screen()
        # Push a fresh welcome screen
        self.push_screen(WelcomeScreen())

    def action_quit(self) -> None:
        """Quit the application."""
        self.exit()


def main():
    """Entry point for the TUI."""
    app = MLCLIApp()
    app.run()


if __name__ == "__main__":
    main()
