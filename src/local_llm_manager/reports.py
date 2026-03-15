"""Report generation for evaluation results — Markdown, HTML, and JSON."""

import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from .eval_harness import EvalResult
from .comparison import ModelComparison, LeaderboardEntry


def _bar(value: float, max_val: float = 1.0, width: int = 20) -> str:
    """Create an ASCII bar chart segment."""
    filled = int((value / max_val) * width) if max_val > 0 else 0
    return "█" * filled + "░" * (width - filled)


def _pct(value: float) -> str:
    """Format a 0-1 float as percentage string."""
    return f"{value * 100:.1f}%"


# ---------------------------------------------------------------------------
# Markdown Report
# ---------------------------------------------------------------------------

def generate_markdown_report(
    results: List[EvalResult],
    comparisons: Optional[List[ModelComparison]] = None,
    leaderboard: Optional[List[LeaderboardEntry]] = None,
) -> str:
    """Generate a Markdown benchmark report."""
    lines: List[str] = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines.append("# 📊 LLM Evaluation Report")
    lines.append(f"\n*Generated: {now}*\n")

    # Hardware info from first result
    if results and results[0].hardware_info:
        hw = results[0].hardware_info
        lines.append("## 🖥️ Hardware")
        lines.append("")
        lines.append(f"| Component | Value |")
        lines.append(f"|-----------|-------|")
        if hw.get("gpu"):
            lines.append(f"| GPU | {hw['gpu']} ({hw.get('vram_mb', 0)}MB VRAM) |")
        lines.append(f"| CPU | {hw.get('cpu', 'N/A')} ({hw.get('cores', 'N/A')} cores) |")
        lines.append(f"| RAM | {hw.get('ram_gb', 'N/A')}GB |")
        lines.append("")

    # Per-model results
    lines.append("## 📈 Evaluation Results\n")
    for r in results:
        lines.append(f"### {r.model_name} — {r.dataset_name}\n")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Accuracy | **{_pct(r.accuracy)}** ({r.correct_count}/{r.total_questions}) |")
        lines.append(f"| Avg Latency | {r.avg_latency_ms:.0f}ms |")
        lines.append(f"| P50 Latency | {r.p50_latency_ms:.0f}ms |")
        lines.append(f"| P95 Latency | {r.p95_latency_ms:.0f}ms |")
        lines.append(f"| Total Eval Time | {r.total_eval_time_s:.1f}s |")
        lines.append("")

        # Per-subject breakdown
        if r.per_subject_accuracy:
            lines.append(f"**Per-Subject Breakdown:**\n")
            lines.append("| Subject | Accuracy | Bar |")
            lines.append("|---------|----------|-----|")
            for subj, data in sorted(r.per_subject_accuracy.items()):
                acc = data.get("accuracy", 0)
                lines.append(f"| {subj} | {_pct(acc)} | `{_bar(acc)}` |")
            lines.append("")

        # Per-group breakdown
        if r.per_group_accuracy:
            lines.append(f"**Per-Group Breakdown:**\n")
            lines.append("| Group | Accuracy | Bar |")
            lines.append("|-------|----------|-----|")
            for grp, data in sorted(r.per_group_accuracy.items()):
                acc = data.get("accuracy", 0)
                lines.append(f"| {grp} | {_pct(acc)} | `{_bar(acc)}` |")
            lines.append("")

    # Comparisons
    if comparisons:
        lines.append("## ⚖️ Model Comparisons\n")
        for c in comparisons:
            lines.append(f"### {c.model_a} vs {c.model_b} ({c.dataset_name})\n")
            lines.append(f"| Metric | {c.model_a} | {c.model_b} | Delta |")
            lines.append(f"|--------|------------|------------|-------|")
            lines.append(
                f"| Accuracy | {_pct(c.accuracy_a)} | {_pct(c.accuracy_b)} | "
                f"{'+' if c.accuracy_delta >= 0 else ''}{_pct(c.accuracy_delta)} |"
            )
            lines.append(
                f"| Avg Latency | {c.avg_latency_a_ms:.0f}ms | "
                f"{c.avg_latency_b_ms:.0f}ms | {c.latency_speedup:.2f}x |"
            )
            lines.append(
                f"| P95 Latency | {c.p95_latency_a_ms:.0f}ms | "
                f"{c.p95_latency_b_ms:.0f}ms | — |"
            )
            lines.append(f"| Quality-Efficiency | — | — | {c.quality_efficiency_score:.3f} |")
            lines.append("")
            lines.append(f"> {c.summary}")
            lines.append("")

    # Leaderboard
    if leaderboard:
        lines.append("## 🏆 Leaderboard\n")
        lines.append("| Rank | Model | Dataset | Accuracy | Avg Latency | Date |")
        lines.append("|------|-------|---------|----------|-------------|------|")
        for e in leaderboard:
            medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(e.rank, str(e.rank))
            lines.append(
                f"| {medal} | {e.model_name} | {e.dataset_name} | "
                f"{_pct(e.accuracy)} | {e.avg_latency_ms:.0f}ms | {e.timestamp} |"
            )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML Report
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LLM Evaluation Report</title>
<style>
  :root {
    --bg: #0f1117; --surface: #1a1d27; --border: #2a2d3a;
    --text: #e4e4e7; --muted: #9ca3af; --accent: #818cf8;
    --green: #34d399; --red: #f87171; --yellow: #fbbf24;
    --gradient-start: #6366f1; --gradient-end: #8b5cf6;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: var(--bg); color: var(--text);
    line-height: 1.6; padding: 2rem;
  }
  .container { max-width: 1100px; margin: 0 auto; }
  h1 {
    font-size: 2rem; margin-bottom: .5rem;
    background: linear-gradient(135deg, var(--gradient-start), var(--gradient-end));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  h2 { font-size: 1.4rem; margin: 2rem 0 1rem; color: var(--accent); }
  h3 { font-size: 1.1rem; margin: 1.5rem 0 .75rem; color: var(--text); }
  .subtitle { color: var(--muted); font-size: .9rem; margin-bottom: 2rem; }
  .card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem;
  }
  table { width: 100%%; border-collapse: collapse; margin: .75rem 0; }
  th, td {
    text-align: left; padding: .6rem .8rem;
    border-bottom: 1px solid var(--border); font-size: .9rem;
  }
  th { color: var(--muted); font-weight: 600; text-transform: uppercase; font-size: .75rem; letter-spacing: .05em; }
  td { color: var(--text); }
  .bar-container {
    background: var(--border); border-radius: 4px; height: 20px;
    overflow: hidden; width: 100%%;
  }
  .bar-fill {
    height: 100%%; border-radius: 4px;
    background: linear-gradient(90deg, var(--gradient-start), var(--gradient-end));
    transition: width .4s ease;
  }
  .bar-fill.green { background: linear-gradient(90deg, #059669, var(--green)); }
  .bar-fill.red { background: linear-gradient(90deg, var(--red), #ef4444); }
  .metric-value { font-size: 1.6rem; font-weight: 700; color: var(--accent); }
  .metric-label { font-size: .8rem; color: var(--muted); text-transform: uppercase; }
  .metrics-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 1rem; margin: 1rem 0;
  }
  .metric-card {
    background: var(--bg); border: 1px solid var(--border);
    border-radius: 8px; padding: 1rem; text-align: center;
  }
  .badge {
    display: inline-block; padding: .2rem .6rem; border-radius: 9999px;
    font-size: .75rem; font-weight: 600;
  }
  .badge-green { background: rgba(52,211,153,.15); color: var(--green); }
  .badge-red { background: rgba(248,113,113,.15); color: var(--red); }
  .badge-yellow { background: rgba(251,191,36,.15); color: var(--yellow); }
  .medal { font-size: 1.2rem; }
  .comparison-arrow { color: var(--accent); font-weight: 700; }
  .delta-positive { color: var(--green); }
  .delta-negative { color: var(--red); }
  footer { text-align: center; color: var(--muted); font-size: .8rem; margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--border); }
</style>
</head>
<body>
<div class="container">
  <h1>📊 LLM Evaluation Report</h1>
  <p class="subtitle">Generated: %(timestamp)s</p>

  %(hardware_section)s
  %(results_section)s
  %(comparison_section)s
  %(leaderboard_section)s

  <footer>
    Generated by <strong>Local LLM Manager</strong> v0.2.0 — Eval Harness
  </footer>
</div>
</body>
</html>"""


def _html_hardware(hw: Dict[str, Any]) -> str:
    if not hw:
        return ""
    rows = ""
    if hw.get("gpu"):
        rows += f'<tr><td>GPU</td><td>{hw["gpu"]} ({hw.get("vram_mb", 0)}MB VRAM)</td></tr>'
    rows += f'<tr><td>CPU</td><td>{hw.get("cpu", "N/A")} ({hw.get("cores", "N/A")} cores)</td></tr>'
    rows += f'<tr><td>RAM</td><td>{hw.get("ram_gb", "N/A")}GB</td></tr>'
    return f"""
  <h2>🖥️ Hardware</h2>
  <div class="card">
    <table><thead><tr><th>Component</th><th>Value</th></tr></thead>
    <tbody>{rows}</tbody></table>
  </div>"""


def _html_bar(value: float, css_class: str = "") -> str:
    pct = min(value * 100, 100)
    cls = f"bar-fill {css_class}" if css_class else "bar-fill"
    return f'<div class="bar-container"><div class="{cls}" style="width:{pct:.1f}%%"></div></div>'


def _html_results(results: List[EvalResult]) -> str:
    sections = []
    for r in results:
        # Metrics grid
        metrics = f"""
    <div class="metrics-grid">
      <div class="metric-card">
        <div class="metric-value">{r.accuracy*100:.1f}%%</div>
        <div class="metric-label">Accuracy</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">{r.correct_count}/{r.total_questions}</div>
        <div class="metric-label">Correct</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">{r.avg_latency_ms:.0f}ms</div>
        <div class="metric-label">Avg Latency</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">{r.p95_latency_ms:.0f}ms</div>
        <div class="metric-label">P95 Latency</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">{r.total_eval_time_s:.1f}s</div>
        <div class="metric-label">Total Time</div>
      </div>
    </div>"""

        # Subject breakdown
        subject_rows = ""
        if r.per_subject_accuracy:
            for subj, data in sorted(r.per_subject_accuracy.items()):
                acc = data.get("accuracy", 0)
                cnt = f'{data.get("correct", 0)}/{data.get("total", 0)}'
                subject_rows += f'<tr><td>{subj}</td><td>{acc*100:.1f}%%</td><td>{cnt}</td><td>{_html_bar(acc)}</td></tr>'

        subject_table = ""
        if subject_rows:
            subject_table = f"""
    <h3>Per-Subject Breakdown</h3>
    <table>
      <thead><tr><th>Subject</th><th>Accuracy</th><th>Score</th><th>Distribution</th></tr></thead>
      <tbody>{subject_rows}</tbody>
    </table>"""

        # Group breakdown
        group_rows = ""
        if r.per_group_accuracy:
            for grp, data in sorted(r.per_group_accuracy.items()):
                acc = data.get("accuracy", 0)
                cnt = f'{data.get("correct", 0)}/{data.get("total", 0)}'
                css = "green" if acc >= 0.7 else ("" if acc >= 0.4 else "red")
                group_rows += f'<tr><td>{grp}</td><td>{acc*100:.1f}%%</td><td>{cnt}</td><td>{_html_bar(acc, css)}</td></tr>'

        group_table = ""
        if group_rows:
            group_table = f"""
    <h3>Per-Group Breakdown</h3>
    <table>
      <thead><tr><th>Group</th><th>Accuracy</th><th>Score</th><th>Distribution</th></tr></thead>
      <tbody>{group_rows}</tbody>
    </table>"""

        sections.append(f"""
  <h3>{r.model_name} — {r.dataset_name}</h3>
  <div class="card">
    {metrics}
    {subject_table}
    {group_table}
  </div>""")

    return f'<h2>📈 Evaluation Results</h2>{"".join(sections)}'


def _html_comparisons(comparisons: List[ModelComparison]) -> str:
    if not comparisons:
        return ""
    sections = []
    for c in comparisons:
        delta_cls = "delta-positive" if c.accuracy_delta >= 0 else "delta-negative"
        delta_sign = "+" if c.accuracy_delta >= 0 else ""
        sections.append(f"""
    <div class="card">
      <h3>{c.model_a} <span class="comparison-arrow">→</span> {c.model_b} ({c.dataset_name})</h3>
      <table>
        <thead><tr><th>Metric</th><th>{c.model_a}</th><th>{c.model_b}</th><th>Delta</th></tr></thead>
        <tbody>
          <tr><td>Accuracy</td><td>{c.accuracy_a*100:.1f}%%</td><td>{c.accuracy_b*100:.1f}%%</td>
              <td class="{delta_cls}">{delta_sign}{c.accuracy_delta*100:.1f}%%</td></tr>
          <tr><td>Avg Latency</td><td>{c.avg_latency_a_ms:.0f}ms</td><td>{c.avg_latency_b_ms:.0f}ms</td>
              <td>{c.latency_speedup:.2f}x</td></tr>
          <tr><td>P95 Latency</td><td>{c.p95_latency_a_ms:.0f}ms</td><td>{c.p95_latency_b_ms:.0f}ms</td>
              <td>—</td></tr>
          <tr><td>Quality-Efficiency</td><td>—</td><td>—</td>
              <td>{c.quality_efficiency_score:.3f}</td></tr>
        </tbody>
      </table>
    </div>""")
    return f'<h2>⚖️ Model Comparisons</h2>{"".join(sections)}'


def _html_leaderboard(leaderboard: List[LeaderboardEntry]) -> str:
    if not leaderboard:
        return ""
    rows = ""
    for e in leaderboard:
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(e.rank, str(e.rank))
        rows += (
            f'<tr><td><span class="medal">{medal}</span></td>'
            f"<td>{e.model_name}</td><td>{e.dataset_name}</td>"
            f"<td>{e.accuracy*100:.1f}%%</td><td>{e.avg_latency_ms:.0f}ms</td>"
            f"<td>{e.timestamp}</td></tr>"
        )
    return f"""
  <h2>🏆 Leaderboard</h2>
  <div class="card">
    <table>
      <thead><tr><th>Rank</th><th>Model</th><th>Dataset</th><th>Accuracy</th><th>Avg Latency</th><th>Date</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>"""


def generate_html_report(
    results: List[EvalResult],
    comparisons: Optional[List[ModelComparison]] = None,
    leaderboard: Optional[List[LeaderboardEntry]] = None,
) -> str:
    """Generate a standalone dark-themed HTML report."""
    hw = results[0].hardware_info if results else {}
    return _HTML_TEMPLATE % {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "hardware_section": _html_hardware(hw),
        "results_section": _html_results(results),
        "comparison_section": _html_comparisons(comparisons or []),
        "leaderboard_section": _html_leaderboard(leaderboard or []),
    }


# ---------------------------------------------------------------------------
# JSON Export
# ---------------------------------------------------------------------------

def generate_json_report(
    results: List[EvalResult],
    comparisons: Optional[List[ModelComparison]] = None,
    leaderboard: Optional[List[LeaderboardEntry]] = None,
) -> str:
    """Generate a JSON export of evaluation results."""
    def _result_to_dict(r: EvalResult) -> Dict[str, Any]:
        return {
            "model_name": r.model_name,
            "dataset_name": r.dataset_name,
            "timestamp": r.timestamp.isoformat(),
            "accuracy": r.accuracy,
            "correct_count": r.correct_count,
            "total_questions": r.total_questions,
            "per_subject_accuracy": r.per_subject_accuracy,
            "per_group_accuracy": r.per_group_accuracy,
            "avg_latency_ms": r.avg_latency_ms,
            "p50_latency_ms": r.p50_latency_ms,
            "p95_latency_ms": r.p95_latency_ms,
            "min_latency_ms": r.min_latency_ms,
            "max_latency_ms": r.max_latency_ms,
            "total_eval_time_s": r.total_eval_time_s,
            "hardware_info": r.hardware_info,
        }

    def _comparison_to_dict(c: ModelComparison) -> Dict[str, Any]:
        return {
            "model_a": c.model_a,
            "model_b": c.model_b,
            "dataset_name": c.dataset_name,
            "accuracy_a": c.accuracy_a,
            "accuracy_b": c.accuracy_b,
            "accuracy_delta": c.accuracy_delta,
            "accuracy_pct_change": c.accuracy_pct_change,
            "avg_latency_a_ms": c.avg_latency_a_ms,
            "avg_latency_b_ms": c.avg_latency_b_ms,
            "latency_speedup": c.latency_speedup,
            "quality_efficiency_score": c.quality_efficiency_score,
            "per_subject_delta": c.per_subject_delta,
            "per_group_delta": c.per_group_delta,
        }

    payload = {
        "generated_at": datetime.now().isoformat(),
        "version": "0.2.0",
        "results": [_result_to_dict(r) for r in results],
        "comparisons": [_comparison_to_dict(c) for c in (comparisons or [])],
        "leaderboard": [
            {
                "rank": e.rank,
                "model_name": e.model_name,
                "dataset_name": e.dataset_name,
                "accuracy": e.accuracy,
                "avg_latency_ms": e.avg_latency_ms,
                "timestamp": e.timestamp,
            }
            for e in (leaderboard or [])
        ],
    }
    return json.dumps(payload, indent=2)


class ReportGenerator:
    """Unified report generator."""

    def __init__(
        self,
        results: List[EvalResult],
        comparisons: Optional[List[ModelComparison]] = None,
        leaderboard: Optional[List[LeaderboardEntry]] = None,
    ):
        self.results = results
        self.comparisons = comparisons
        self.leaderboard = leaderboard

    def generate(self, fmt: str = "md") -> str:
        """Generate report in specified format (md, html, json)."""
        if fmt == "html":
            return generate_html_report(self.results, self.comparisons, self.leaderboard)
        elif fmt == "json":
            return generate_json_report(self.results, self.comparisons, self.leaderboard)
        else:
            return generate_markdown_report(self.results, self.comparisons, self.leaderboard)

    def save(self, path: str, fmt: Optional[str] = None):
        """Generate and save report to file."""
        if fmt is None:
            # Infer from extension
            ext = Path(path).suffix.lower()
            fmt = {"html": "html", ".json": "json"}.get(ext, "md")
        content = self.generate(fmt)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
