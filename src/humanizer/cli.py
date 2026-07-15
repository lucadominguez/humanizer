"""
Humanizer CLI - Benchmark and humanize text from the command line.
"""

import sys
import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

from .features import FeatureExtractor
from .benchmark import BenchmarkHarness, DetectionResult
from .perturb import Humanizer

console = Console()


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="humanize",
        description="AI text humanization with statistical proof",
        epilog="Benchmarked against heuristic and perplexity-based detection",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # detect command
    detect_parser = subparsers.add_parser("detect", help="Detect if text is AI-generated")
    detect_parser.add_argument("text", nargs="?", help="Text to analyze (or pipe via stdin)")
    detect_parser.add_argument("--file", "-f", help="Read text from file")
    detect_parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    detect_parser.add_argument("--features", action="store_true", help="Show extracted features")
    
    # humanize command
    humanize_parser = subparsers.add_parser("humanize", help="Humanize AI-generated text")
    humanize_parser.add_argument("text", nargs="?", help="Text to humanize (or pipe via stdin)")
    humanize_parser.add_argument("--file", "-f", help="Read text from file")
    humanize_parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    humanize_parser.add_argument("--iterations", "-i", type=int, default=3, help="Max iterations (default: 3)")
    
    # benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run benchmark suite on text samples")
    benchmark_parser.add_argument("--human", "-H", help="File containing human-written text samples (one per line)")
    benchmark_parser.add_argument("--ai", "-A", help="File containing AI-generated text samples (one per line)")
    benchmark_parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    
    return parser


def get_input_text(args) -> str:
    """Get text from args, file, or stdin."""
    if args.file:
        return Path(args.file).read_text()
    elif args.text:
        return args.text
    elif not sys.stdin.isatty():
        return sys.stdin.read().strip()
    return ""


def display_detection(result):
    """Display detection results."""
    color = {
        DetectionResult.HUMAN: "green",
        DetectionResult.AI: "red",
        DetectionResult.MIXED: "yellow",
        DetectionResult.UNCERTAIN: "dim",
    }.get(result.majority_vote, "white")
    
    verdict = {
        DetectionResult.HUMAN: "HUMAN-WRITTEN",
        DetectionResult.AI: "AI-GENERATED",
        DetectionResult.MIXED: "MIXED / UNCERTAIN",
        DetectionResult.UNCERTAIN: "UNCERTAIN",
    }.get(result.majority_vote, "?")
    
    console.print(Panel.fit(
        f"[bold {color}]{verdict}[/bold {color}]\n"
        f"Human probability: [bold]{result.overall_human_probability:.1%}[/bold]\n"
        f"Text length: {result.text_length} chars\n"
        f"Detectors agree: {'[green]Yes[/green]' if result.detectors_agree else '[yellow]No[/yellow]'}",
        border_style=color
    ))
    
    # Per-detector scores
    table = Table(title="Detector Results")
    table.add_column("Detector", style="bold")
    table.add_column("Verdict")
    table.add_column("Human Prob", justify="right")
    table.add_column("Confidence", justify="right")
    
    for score in result.scores:
        v_color = {
            DetectionResult.HUMAN: "green",
            DetectionResult.AI: "red",
            DetectionResult.MIXED: "yellow",
            DetectionResult.UNCERTAIN: "dim",
        }.get(score.result, "white")
        
        table.add_row(
            score.detector_name,
            f"[{v_color}]{score.result.value.upper()}[/{v_color}]",
            f"{score.human_probability:.1%}",
            f"{score.confidence:.1%}",
        )
    
    console.print(table)


def display_features(features):
    """Display extracted text features."""
    vec = __import__('humanizer.features', fromlist=['feature_vector']).feature_vector(features)
    
    table = Table(title="Text Features")
    table.add_column("Feature", style="bold")
    table.add_column("Value", justify="right")
    
    for key, value in sorted(vec.items()):
        if isinstance(value, float):
            table.add_row(key, f"{value:.4f}")
    
    console.print(table)


def main():
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == "detect":
        text = get_input_text(args)
        if not text:
            console.print("[red]No text provided. Pipe text, use --file, or pass as argument.[/red]")
            sys.exit(1)
        
        harness = BenchmarkHarness()
        result = harness.benchmark(text)
        
        if args.json:
            output = {
                "verdict": result.majority_vote.value,
                "human_probability": result.overall_human_probability,
                "text_length": result.text_length,
                "detectors_agree": result.detectors_agree,
                "scores": [
                    {
                        "detector": s.detector_name,
                        "verdict": s.result.value,
                        "human_probability": s.human_probability,
                        "ai_probability": s.ai_probability,
                        "confidence": s.confidence,
                    }
                    for s in result.scores
                ],
            }
            print(json.dumps(output, indent=2))
        else:
            display_detection(result)
            if args.features and result.features:
                from humanizer.features import feature_vector as fv
                display_features(result.features)
    
    elif args.command == "humanize":
        text = get_input_text(args)
        if not text:
            console.print("[red]No text provided.[/red]")
            sys.exit(1)
        
        humanizer = Humanizer()
        
        with console.status("[bold green]Humanizing...[/bold green]"):
            report = humanizer.humanize_with_report(text)
        
        if args.json:
            print(json.dumps({
                "before_ai_probability": report["before_ai_probability"],
                "after_ai_probability": report["after_ai_probability"],
                "improvement": report["improvement"],
                "humanized_text": report["humanized_text"],
                "strategies_applied": report["strategies_applied"],
                "characters_changed": report["characters_changed"],
            }, indent=2))
        else:
            console.print(Panel.fit(
                f"[bold]Before:[/bold] AI probability [red]{report['before_ai_probability']:.1%}[/red]\n"
                f"[bold]After:[/bold]  AI probability [green]{report['after_ai_probability']:.1%}[/green]\n"
                f"[bold]Improvement:[/bold] [green]{report['improvement']:.1%}[/green]",
                border_style="green"
            ))
            
            console.print("\n[bold]Humanized text:[/bold]")
            console.print(report["humanized_text"])
    
    elif args.command == "benchmark":
        if not args.human and not args.ai:
            console.print("[red]Provide --human and/or --ai files with text samples.[/red]")
            sys.exit(1)
        
        harness = BenchmarkHarness()
        
        human_texts = []
        ai_texts = []
        
        if args.human:
            with open(args.human) as f:
                human_texts = [line.strip() for line in f if line.strip()]
        
        if args.ai:
            with open(args.ai) as f:
                ai_texts = [line.strip() for line in f if line.strip()]
        
        all_texts = human_texts + ai_texts
        all_labels = ["human"] * len(human_texts) + ["ai"] * len(ai_texts)
        
        results = harness.benchmark_batch(list(zip(all_texts, all_labels)))
        report = harness.accuracy_report(results, all_labels)
        
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            console.print(f"\n[bold]Benchmark Results:[/bold]")
            console.print(f"  Human samples: {report['human_samples']}")
            console.print(f"  AI samples: {report['ai_samples']}")
            
            for name, stats in report["detectors"].items():
                console.print(f"\n  [bold cyan]{name}[/bold cyan]")
                console.print(f"    Accuracy: {stats['accuracy']:.1%}")
                console.print(f"    Human recall: {stats['human_recall']:.1%}")
                console.print(f"    AI recall: {stats['ai_recall']:.1%}")


if __name__ == "__main__":
    main()