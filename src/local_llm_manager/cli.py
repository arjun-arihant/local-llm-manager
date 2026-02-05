"""CLI interface for Local LLM Manager."""

import click
import sys
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
@click.version_option(version="0.1.0", prog_name="llm-manager")
def cli():
    """Local LLM Manager - Intelligent hardware-aware tool for managing local LLMs."""
    pass


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


@cli.command()
def list():
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


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
