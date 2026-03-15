"""CLI interface for Local LLM Manager."""

import click
import sys
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from .hardware import get_hardware_profile, format_hardware_profile
from .recommendations import get_recommendations, get_all_models
from .ollama_client import OllamaClient
from .benchmark import benchmark_model, run_full_benchmark, format_benchmark_stats
from .database import BenchmarkDB
from .quantize import auto_quantize, quantize_model, get_quantize_options

console = Console()


def check_ollama():
    """Check if Ollama is running."""
    client = OllamaClient()
    if not client.is_running():
        console.print("[red]Error: Ollama is not running![/red]")
        console.print("[yellow]Please start Ollama first:[/yellow]")
        console.print("  ollama serve")
        sys.exit(1)


@click.group()
@click.version_option(version="0.2.0", prog_name="llm-manager")
def cli():
    """Local LLM Manager - Hardware-aware LLM management & evaluation harness."""
    pass


# ──────────────────────────────────────────────────────────────────────────────
#  Original commands
# ──────────────────────────────────────────────────────────────────────────────

@cli.command()
def detect():
    """Detect and display hardware specifications."""
    console.print(Panel.fit("[bold blue]Hardware Detection[/bold blue]"))

    with console.status("[bold green]Detecting hardware..."):
        profile = get_hardware_profile()
        data = format_hardware_profile(profile)

    # Create table
    table = Table(title="System Hardware")
    table.add_column("Component", style="cyan")
    table.add_column("Property", style="magenta")
    table.add_column("Value", style="green")

    for component, props in data.items():
        for prop, value in props.items():
            table.add_row(component, prop, str(value))

    console.print(table)

    # Summary
    if profile.gpu:
        vram_gb = profile.gpu.vram_total_mb / 1024
        console.print(f"\n[green]✓[/green] GPU detected with {vram_gb:.1f}GB VRAM")
        console.print(f"[green]✓[/green] System is ready for local LLM inference")
    else:
        console.print("\n[yellow]⚠[/yellow] No GPU detected - models will run on CPU (slower)")


@cli.command()
@click.option("--all", "show_all", is_flag=True, help="Show all models, not just recommended")
def recommend(show_all):
    """Get model recommendations based on hardware."""
    console.print(Panel.fit("[bold blue]Model Recommendations[/bold blue]"))

    with console.status("[bold green]Analyzing hardware..."):
        profile = get_hardware_profile()

    if show_all:
        models = get_all_models()
        title = "All Available Models"
    else:
        models = get_recommendations(profile)
        title = "Recommended Models for Your Hardware"

        # Show hardware summary
        if profile.gpu:
            console.print(f"[dim]GPU: {profile.gpu.name} ({profile.gpu.vram_total_mb}MB VRAM)[/dim]")
        console.print(f"[dim]RAM: {profile.ram.total_gb:.1f}GB[/dim]\n")

    table = Table(title=title)
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")
    table.add_column("Size", style="blue")
    table.add_column("Requirements", style="magenta")

    if not show_all:
        table.add_column("Fit", style="green")

    for model in models:
        row = [
            model["name"],
            model["description"],
            model["size"],
            model["requirements"],
        ]
        if not show_all:
            row.append(model["fit_score"])
        table.add_row(*row)

    console.print(table)

    if not show_all:
        console.print("\n[dim]Use --all to see all available models[/dim]")


@cli.command()
@click.argument("model")
def install(model):
    """Install a model via Ollama."""
    check_ollama()

    console.print(Panel.fit(f"[bold blue]Installing Model: {model}[/bold blue]"))

    client = OllamaClient()

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"Pulling {model}...", total=100)

            for update in client.pull_model(model):
                if update.percent >= 0:
                    progress.update(task, completed=update.percent, description=update.status)
                else:
                    progress.update(task, description=update.status)

        console.print(f"[green]✓[/green] Model [bold]{model}[/bold] installed successfully!")

    except Exception as e:
        console.print(f"[red]✗[/red] Failed to install {model}: {e}")
        sys.exit(1)


@cli.command(name="list")
def list_models():
    """List installed models."""
    check_ollama()

    console.print(Panel.fit("[bold blue]Installed Models[/bold blue]"))

    client = OllamaClient()
    models = client.list_models()

    if not models:
        console.print("[yellow]No models installed.[/yellow]")
        console.print("[dim]Use 'llm-manager install <model>' to install one.[/dim]")
        return

    table = Table()
    table.add_column("Model", style="cyan")
    table.add_column("Size", style="blue")
    table.add_column("Modified", style="magenta")
    table.add_column("Digest", style="dim")

    for model in models:
        size_gb = model.size / (1024**3)
        table.add_row(
            model.name,
            f"{size_gb:.2f} GB",
            model.modified_at[:19] if model.modified_at else "Unknown",
            model.digest[:12] + "..." if model.digest else ""
        )

    console.print(table)
    console.print(f"\n[dim]Total: {len(models)} model(s)[/dim]")


@cli.command()
@click.argument("model")
@click.option("--full", is_flag=True, help="Run full benchmark with multiple prompts")
def benchmark(model, full):
    """Benchmark a model's performance."""
    check_ollama()

    console.print(Panel.fit(f"[bold blue]Benchmarking: {model}[/bold blue]"))

    client = OllamaClient()

    # Check if model exists
    installed = [m.name for m in client.list_models()]
    if model not in installed:
        console.print(f"[red]Model {model} is not installed.[/red]")
        console.print(f"[dim]Run: llm-manager install {model}[/dim]")
        sys.exit(1)

    try:
        if full:
            results = run_full_benchmark(model, client)

            table = Table(title="Benchmark Results")
            table.add_column("Prompt Type", style="cyan")
            table.add_column("Tokens/sec", style="green")
            table.add_column("Total Time", style="blue")
            table.add_column("Tokens", style="magenta")

            for prompt_type, stats in results.items():
                table.add_row(
                    prompt_type.capitalize(),
                    f"{stats.tokens_per_second:.2f}",
                    f"{stats.total_duration_ms}ms",
                    str(stats.eval_count)
                )

            console.print(table)
        else:
            stats = benchmark_model(model, client, save=True)

            table = Table(title="Benchmark Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            formatted = format_benchmark_stats(stats)
            for key, value in formatted.items():
                table.add_row(key, value)

            console.print(table)

            # Show historical comparison
            db = BenchmarkDB()
            hist_stats = db.get_model_stats(model)
            if hist_stats and hist_stats["benchmark_count"] > 1:
                console.print(f"\n[dim]History: {hist_stats['benchmark_count']} runs, "
                            f"avg: {hist_stats['avg_tokens_per_second']} tok/s, "
                            f"max: {hist_stats['max_tokens_per_second']} tok/s[/dim]")

    except Exception as e:
        console.print(f"[red]✗[/red] Benchmark failed: {e}")
        sys.exit(1)


@cli.command()
@click.argument("model")
@click.option("--level", default="auto", help="Quantization level (auto, q8_0, q4_k_m, q4_0, q3_k_m, q2_k)")
def quantize(model, level):
    """Quantize a model for optimal hardware usage."""
    check_ollama()

    console.print(Panel.fit(f"[bold blue]Quantizing: {model}[/bold blue]"))

    client = OllamaClient()

    # Check if model exists
    installed = [m.name for m in client.list_models()]
    if model not in installed:
        console.print(f"[red]Model {model} is not installed.[/red]")
        sys.exit(1)

    try:
        if level == "auto":
            new_name = auto_quantize(model)
        else:
            # Validate level
            options = get_quantize_options()
            valid_levels = [opt["level"] for opt in options]
            if level not in valid_levels:
                console.print(f"[red]Invalid quantization level: {level}[/red]")
                console.print(f"[dim]Valid options: {', '.join(valid_levels)}[/dim]")
                sys.exit(1)
            new_name = quantize_model(model, level)

        console.print(f"[green]✓[/green] Quantized model created: [bold]{new_name}[/bold]")
        console.print(f"[dim]Use 'llm-manager benchmark {new_name}' to test performance[/dim]")

    except Exception as e:
        console.print(f"[red]✗[/red] Quantization failed: {e}")
        sys.exit(1)


@cli.command(name="history")
@click.option("--model", help="Filter by model name")
@click.option("--limit", default=10, help="Number of results to show")
def show_history(model, limit):
    """Show benchmark history."""
    console.print(Panel.fit("[bold blue]Benchmark History[/bold blue]"))

    db = BenchmarkDB()
    results = db.get_benchmarks(model, limit)

    if not results:
        console.print("[yellow]No benchmark history found.[/yellow]")
        return

    table = Table()
    table.add_column("ID", style="dim")
    table.add_column("Model", style="cyan")
    table.add_column("Date", style="blue")
    table.add_column("Tokens/sec", style="green")
    table.add_column("Duration", style="magenta")
    table.add_column("GPU", style="dim")

    for r in results:
        gpu_name = r.hardware_info.get("gpu", "CPU")
        if gpu_name and len(gpu_name) > 20:
            gpu_name = gpu_name[:17] + "..."

        table.add_row(
            str(r.id),
            r.model_name,
            r.timestamp.strftime("%Y-%m-%d %H:%M"),
            f"{r.tokens_per_second:.2f}",
            f"{r.total_duration_ms}ms",
            gpu_name or "CPU"
        )

    console.print(table)


# ──────────────────────────────────────────────────────────────────────────────
#  Eval Harness commands
# ──────────────────────────────────────────────────────────────────────────────

@cli.command(name="eval")
@click.argument("model")
@click.option(
    "--dataset", "-d",
    type=click.Choice(["truthfulqa", "mmlu", "all"], case_sensitive=False),
    default="all",
    help="Dataset to evaluate on",
)
@click.option("--samples", "-n", type=int, default=None, help="Number of questions (default: all)")
@click.option("--seed", type=int, default=42, help="Random seed for sampling")
def eval_model(model, dataset, samples, seed):
    """Evaluate a model on TruthfulQA / MMLU benchmarks."""
    check_ollama()

    from .datasets import DatasetManager
    from .eval_harness import EvalRunner, EvalConfig

    console.print(Panel.fit(f"[bold blue]Evaluating: {model}[/bold blue]"))

    client = OllamaClient()

    # Check if model exists
    installed = [m.name for m in client.list_models()]
    if model not in installed:
        console.print(f"[red]Model {model} is not installed.[/red]")
        console.print(f"[dim]Install it first: llm-manager install {model}[/dim]")
        sys.exit(1)

    # Determine datasets
    if dataset == "all":
        dataset_names = ["truthfulqa", "mmlu"]
    else:
        dataset_names = [dataset]

    config = EvalConfig(models=[model], datasets=dataset_names, num_samples=samples, seed=seed)
    runner = EvalRunner(client=client, config=config)
    db = BenchmarkDB()

    for ds_name in dataset_names:
        ds = DatasetManager.load(ds_name)
        n = samples or len(ds)
        console.print(f"\n[bold]Dataset:[/bold] {ds.name} ({n} questions)")

        # Progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"Evaluating {ds.name}...", total=n)

            def on_progress(current, total, qr):
                status = "✓" if qr.is_correct else "✗"
                progress.update(
                    task,
                    completed=current,
                    description=f"[{current}/{total}] {status} Q{qr.question_id}",
                )

            result = runner.evaluate(model, ds, num_samples=samples, progress_callback=on_progress)

        # Save to DB
        db.save_eval_result(result)

        # Display results
        console.print(f"\n[green]✓[/green] Evaluation saved to database")

        # Summary metrics
        table = Table(title=f"Results: {model} on {ds.name}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Accuracy", f"{result.accuracy * 100:.1f}% ({result.correct_count}/{result.total_questions})")
        table.add_row("Avg Latency", f"{result.avg_latency_ms:.0f}ms")
        table.add_row("P50 Latency", f"{result.p50_latency_ms:.0f}ms")
        table.add_row("P95 Latency", f"{result.p95_latency_ms:.0f}ms")
        table.add_row("Total Time", f"{result.total_eval_time_s:.1f}s")
        console.print(table)

        # Per-group breakdown (if MMLU)
        if result.per_group_accuracy:
            grp_table = Table(title="Per-Group Breakdown")
            grp_table.add_column("Group", style="cyan")
            grp_table.add_column("Accuracy", style="green")
            grp_table.add_column("Score", style="blue")
            for grp, data in sorted(result.per_group_accuracy.items()):
                acc = data.get("accuracy", 0)
                cnt = f"{data.get('correct', 0)}/{data.get('total', 0)}"
                grp_table.add_row(grp, f"{acc * 100:.1f}%", cnt)
            console.print(grp_table)

        # Per-subject breakdown
        if result.per_subject_accuracy:
            subj_table = Table(title="Per-Subject Breakdown")
            subj_table.add_column("Subject", style="cyan")
            subj_table.add_column("Accuracy", style="green")
            subj_table.add_column("Score", style="blue")
            for subj, data in sorted(result.per_subject_accuracy.items()):
                acc = data.get("accuracy", 0)
                cnt = f"{data.get('correct', 0)}/{data.get('total', 0)}"
                subj_table.add_row(subj, f"{acc * 100:.1f}%", cnt)
            console.print(subj_table)


@cli.command()
@click.argument("model_a")
@click.argument("model_b")
@click.option(
    "--dataset", "-d",
    type=click.Choice(["truthfulqa", "mmlu", "all"], case_sensitive=False),
    default="all",
    help="Dataset to compare on",
)
@click.option("--samples", "-n", type=int, default=None, help="Questions per dataset")
@click.option("--seed", type=int, default=42, help="Random seed for sampling")
def compare(model_a, model_b, dataset, samples, seed):
    """Compare two models (e.g. full vs quantized)."""
    check_ollama()

    from .datasets import DatasetManager
    from .eval_harness import EvalRunner, EvalConfig
    from .comparison import compare_models

    console.print(Panel.fit(f"[bold blue]Comparing: {model_a} vs {model_b}[/bold blue]"))

    client = OllamaClient()

    # Verify both models installed
    installed = [m.name for m in client.list_models()]
    for m in [model_a, model_b]:
        if m not in installed:
            console.print(f"[red]Model {m} is not installed.[/red]")
            sys.exit(1)

    if dataset == "all":
        dataset_names = ["truthfulqa", "mmlu"]
    else:
        dataset_names = [dataset]

    config = EvalConfig(models=[model_a, model_b], datasets=dataset_names, num_samples=samples, seed=seed)
    runner = EvalRunner(client=client, config=config)
    db = BenchmarkDB()

    for ds_name in dataset_names:
        ds = DatasetManager.load(ds_name)
        n = samples or len(ds)
        console.print(f"\n[bold]Dataset:[/bold] {ds.name} ({n} questions)")

        # Eval model A
        console.print(f"\n[dim]Evaluating {model_a}...[/dim]")
        with console.status(f"[bold green]Running {model_a} on {ds.name}..."):
            result_a = runner.evaluate(model_a, ds, num_samples=samples)
        db.save_eval_result(result_a)

        # Eval model B
        console.print(f"[dim]Evaluating {model_b}...[/dim]")
        with console.status(f"[bold green]Running {model_b} on {ds.name}..."):
            result_b = runner.evaluate(model_b, ds, num_samples=samples)
        db.save_eval_result(result_b)

        # Compare
        comparison = compare_models(result_a, result_b)

        # Display comparison
        table = Table(title=f"Comparison: {model_a} vs {model_b} on {ds.name}")
        table.add_column("Metric", style="cyan")
        table.add_column(model_a, style="blue")
        table.add_column(model_b, style="magenta")
        table.add_column("Delta", style="green")

        delta_sign = "+" if comparison.accuracy_delta >= 0 else ""
        table.add_row(
            "Accuracy",
            f"{comparison.accuracy_a * 100:.1f}%",
            f"{comparison.accuracy_b * 100:.1f}%",
            f"{delta_sign}{comparison.accuracy_delta * 100:.1f}%",
        )
        table.add_row(
            "Avg Latency",
            f"{comparison.avg_latency_a_ms:.0f}ms",
            f"{comparison.avg_latency_b_ms:.0f}ms",
            f"{comparison.latency_speedup:.2f}x",
        )
        table.add_row(
            "P95 Latency",
            f"{comparison.p95_latency_a_ms:.0f}ms",
            f"{comparison.p95_latency_b_ms:.0f}ms",
            "—",
        )
        table.add_row(
            "Quality-Efficiency",
            "—",
            "—",
            f"{comparison.quality_efficiency_score:.3f}",
        )
        console.print(table)
        console.print(f"\n[dim]{comparison.summary}[/dim]")

        # Per-group delta (if applicable)
        if comparison.per_group_delta:
            grp_table = Table(title="Per-Group Accuracy Delta")
            grp_table.add_column("Group", style="cyan")
            grp_table.add_column("Delta", style="green")
            for grp, delta in sorted(comparison.per_group_delta.items()):
                sign = "+" if delta >= 0 else ""
                grp_table.add_row(grp, f"{sign}{delta * 100:.1f}%")
            console.print(grp_table)


@cli.command()
@click.option("--format", "-f", "fmt", type=click.Choice(["md", "html", "json"]), default="md", help="Report format")
@click.option("--output", "-o", type=click.Path(), default=None, help="Output file path")
@click.option("--model", help="Filter by model name")
@click.option("--dataset", help="Filter by dataset name")
def report(fmt, output, model, dataset):
    """Generate a benchmark report from stored eval results."""
    from .reports import ReportGenerator
    from .eval_harness import EvalResult
    from .comparison import build_leaderboard

    console.print(Panel.fit("[bold blue]Generating Report[/bold blue]"))

    db = BenchmarkDB()
    raw_results = db.get_eval_results(model_name=model, dataset_name=dataset)

    if not raw_results:
        console.print("[yellow]No eval results found.[/yellow]")
        console.print("[dim]Run 'llm-manager eval <model>' first.[/dim]")
        return

    # Reconstruct EvalResult objects
    eval_results = []
    for r in raw_results:
        eval_results.append(EvalResult(
            model_name=r["model_name"],
            dataset_name=r["dataset_name"],
            timestamp=datetime.fromisoformat(r["timestamp"]),
            total_questions=r["total_questions"],
            correct_count=r["correct_count"],
            accuracy=r["accuracy"],
            per_subject_accuracy=r.get("per_subject_scores", {}),
            per_group_accuracy=r.get("per_group_scores", {}),
            avg_latency_ms=r["avg_latency_ms"],
            p50_latency_ms=r.get("p50_latency_ms", 0),
            p95_latency_ms=r.get("p95_latency_ms", 0),
            min_latency_ms=r.get("min_latency_ms", 0),
            max_latency_ms=r.get("max_latency_ms", 0),
            total_eval_time_s=r.get("total_eval_time_s", 0),
            hardware_info=r.get("hardware_info", {}),
        ))

    # Build leaderboard
    lb = build_leaderboard(eval_results)

    gen = ReportGenerator(results=eval_results, leaderboard=lb)

    if output:
        gen.save(output, fmt)
        console.print(f"[green]✓[/green] Report saved to [bold]{output}[/bold]")
    else:
        content = gen.generate(fmt)
        console.print(content)


@cli.command(name="leaderboard")
@click.option("--sort", type=click.Choice(["accuracy", "latency", "efficiency"]), default="accuracy")
@click.option("--limit", default=20, help="Max entries")
def show_leaderboard(sort, limit):
    """Show evaluation leaderboard."""
    from .eval_harness import EvalResult
    from .comparison import build_leaderboard

    console.print(Panel.fit("[bold blue]🏆 Evaluation Leaderboard[/bold blue]"))

    db = BenchmarkDB()
    raw_results = db.get_eval_results(limit=limit)

    if not raw_results:
        console.print("[yellow]No eval results found.[/yellow]")
        console.print("[dim]Run 'llm-manager eval <model>' first.[/dim]")
        return

    # Reconstruct
    eval_results = []
    for r in raw_results:
        eval_results.append(EvalResult(
            model_name=r["model_name"],
            dataset_name=r["dataset_name"],
            timestamp=datetime.fromisoformat(r["timestamp"]),
            total_questions=r["total_questions"],
            correct_count=r["correct_count"],
            accuracy=r["accuracy"],
            per_subject_accuracy=r.get("per_subject_scores", {}),
            per_group_accuracy=r.get("per_group_scores", {}),
            avg_latency_ms=r["avg_latency_ms"],
            p50_latency_ms=r.get("p50_latency_ms", 0),
            p95_latency_ms=r.get("p95_latency_ms", 0),
            min_latency_ms=r.get("min_latency_ms", 0),
            max_latency_ms=r.get("max_latency_ms", 0),
            total_eval_time_s=r.get("total_eval_time_s", 0),
            hardware_info=r.get("hardware_info", {}),
        ))

    lb = build_leaderboard(eval_results, sort_by=sort)

    table = Table(title=f"Leaderboard (sorted by {sort})")
    table.add_column("Rank", style="bold")
    table.add_column("Model", style="cyan")
    table.add_column("Dataset", style="blue")
    table.add_column("Accuracy", style="green")
    table.add_column("Avg Latency", style="magenta")
    table.add_column("Date", style="dim")

    for entry in lb[:limit]:
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(entry.rank, str(entry.rank))
        table.add_row(
            medal,
            entry.model_name,
            entry.dataset_name,
            f"{entry.accuracy * 100:.1f}%",
            f"{entry.avg_latency_ms:.0f}ms",
            entry.timestamp,
        )

    console.print(table)


@cli.command(name="datasets")
def list_datasets():
    """List available evaluation datasets."""
    from .datasets import DatasetManager

    console.print(Panel.fit("[bold blue]Available Datasets[/bold blue]"))

    for info in DatasetManager.list_datasets():
        console.print(f"\n[bold cyan]{info['display_name']}[/bold cyan] ({info['name']})")
        console.print(f"  [dim]{info['description']}[/dim]")
        console.print(f"  Questions: [green]{info['total_questions']}[/green]")
        if info["groups"]:
            console.print(f"  Groups: {', '.join(info['groups'])}")
        if info["subjects"]:
            subjects_str = ", ".join(info["subjects"][:8])
            if len(info["subjects"]) > 8:
                subjects_str += f" (+{len(info['subjects']) - 8} more)"
            console.print(f"  Subjects: {subjects_str}")


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
