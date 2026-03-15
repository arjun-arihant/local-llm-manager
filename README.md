# 🧠 Local LLM Manager & Eval Harness

> **Hardware-aware LLM management + evaluation harness for local models via Ollama.**
> Run TruthfulQA & MMLU benchmarks, compare quantized vs full-precision models, and generate rich benchmark reports — all from a single CLI.

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Ollama](https://img.shields.io/badge/backend-Ollama-orange)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🖥️ **Hardware Detection** | Auto-detect GPU, CPU, RAM specs and determine what models your machine can run |
| 🎯 **Smart Recommendations** | Get model suggestions ranked by hardware fit score |
| 📊 **Eval Harness** | Run TruthfulQA (50 questions) & MMLU (80 questions across 4 domains) benchmarks locally |
| ⚖️ **Quantized vs Full Comparison** | Compare accuracy, latency, and quality-efficiency tradeoffs between model variants |
| 📈 **Rich Reports** | Generate dark-themed HTML, Markdown, and JSON benchmark reports |
| 🏆 **Leaderboard** | Track and rank models across evaluations with sortable leaderboards |
| ⚡ **Performance Benchmarks** | Measure tokens/sec, prompt eval rate, and load times |
| 🔧 **Quantization Helper** | Auto-quantize models to optimal precision for your hardware |

---

## 🏗️ Architecture

```
local-llm-manager/
├── src/local_llm_manager/
│   ├── cli.py              # Click CLI — all commands
│   ├── hardware.py         # GPU/CPU/RAM detection (NVIDIA + AMD)
│   ├── recommendations.py  # Hardware-aware model recommendations
│   ├── ollama_client.py    # Ollama REST API client
│   ├── benchmark.py        # Performance benchmarking (tokens/sec)
│   ├── quantize.py         # Quantization helper (q2_k → q8_0)
│   ├── datasets.py         # TruthfulQA & MMLU dataset loaders
│   ├── eval_harness.py     # Evaluation engine with MC answer extraction
│   ├── comparison.py       # Model comparison & leaderboard
│   ├── reports.py          # Markdown / HTML / JSON report generation
│   ├── database.py         # SQLite persistence (benchmarks + eval results)
│   └── data/
│       ├── truthfulqa_subset.json   # 50 curated TruthfulQA questions
│       └── mmlu_subset.json         # 80 curated MMLU questions
└── pyproject.toml
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **[Ollama](https://ollama.com)** installed and running (`ollama serve`)

### Install

```bash
# From source
git clone https://github.com/arjun-arihant/local-llm-manager.git
cd local-llm-manager
pip install -e .

# Or from PyPI (when published)
pip install local-llm-manager
```

### First Run

```bash
# 1. Detect hardware
llm-manager detect

# 2. Get model recommendations
llm-manager recommend

# 3. Install a model
llm-manager install llama3.2

# 4. Run evaluation
llm-manager eval llama3.2 --dataset truthfulqa
```

---

## 📖 Usage Guide

### 🖥️ Detect Hardware

```bash
llm-manager detect
```

```
╭────────────────────╮
│ Hardware Detection │
╰────────────────────╯
                            System Hardware
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Component ┃ Property      ┃ Value                                   ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ GPU       │ Model         │ NVIDIA GeForce RTX 3060                 │
│ GPU       │ VRAM          │ 12288 MB                                │
│ CPU       │ Cores/Threads │ 8/16                                    │
│ RAM       │ Total         │ 32.0 GB                                 │
└───────────┴───────────────┴─────────────────────────────────────────┘
```

### 📊 Evaluate a Model

```bash
# Run on all datasets
llm-manager eval llama3.2

# Run on specific dataset with sample limit
llm-manager eval llama3.2 --dataset mmlu --samples 20

# Fine-grained control
llm-manager eval mistral --dataset truthfulqa --samples 30 --seed 123
```

```
╭────────────────────────────╮
│ Evaluating: llama3.2       │
╰────────────────────────────╯
Dataset: TruthfulQA (50 questions)
[50/50] ✓ Qtqa_050  ━━━━━━━━━━━━━━━━━━━━━━━━  100%
✓ Evaluation saved to database

         Results: llama3.2 on TruthfulQA
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Metric      ┃ Value           ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ Accuracy    │ 78.0% (39/50)   │
│ Avg Latency │ 1234ms          │
│ P50 Latency │ 1150ms          │
│ P95 Latency │ 2100ms          │
│ Total Time  │ 61.7s           │
└─────────────┴─────────────────┘
```

### ⚖️ Compare Models (Quantized vs Full)

```bash
llm-manager compare llama3.2 llama3.2-q4_k_m --dataset mmlu --samples 20
```

```
       Comparison: llama3.2 vs llama3.2-q4_k_m on MMLU
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Metric             ┃ llama3.2   ┃ llama3.2-q4_k_m┃ Delta   ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ Accuracy           │ 72.5%      │ 68.0%          │ -4.5%   │
│ Avg Latency        │ 1500ms     │ 950ms          │ 1.58x   │
│ P95 Latency        │ 2200ms     │ 1400ms         │ —       │
│ Quality-Efficiency  │ —          │ —              │ 0.712   │
└────────────────────┴────────────┴────────────────┴─────────┘
```

### 📈 Generate Reports

```bash
# Markdown report
llm-manager report --format md --output eval_report.md

# Dark-themed HTML report
llm-manager report --format html --output eval_report.html

# JSON export (for CI/CD)
llm-manager report --format json --output results.json

# Filter by model
llm-manager report --model llama3.2 --format html --output llama_report.html
```

### 🏆 View Leaderboard

```bash
# Sort by accuracy (default)
llm-manager leaderboard

# Sort by latency or efficiency
llm-manager leaderboard --sort latency
llm-manager leaderboard --sort efficiency
```

```
         🏆 Leaderboard (sorted by accuracy)
┏━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Rank ┃ Model           ┃ Dataset    ┃ Accuracy ┃ Avg Latency  ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ 🥇   │ mistral         │ TruthfulQA │ 82.0%    │ 1800ms       │
│ 🥈   │ llama3.2        │ TruthfulQA │ 78.0%    │ 1234ms       │
│ 🥉   │ llama3.2-q4_k_m │ TruthfulQA │ 74.0%    │ 850ms        │
└──────┴─────────────────┴────────────┴──────────┴──────────────┘
```

### 📋 List Datasets

```bash
llm-manager datasets
```

### 🎯 Other Commands

```bash
# Get model recommendations for your hardware
llm-manager recommend

# Install a model
llm-manager install <model>

# List installed models
llm-manager list

# Benchmark raw performance (tokens/sec)
llm-manager benchmark <model>
llm-manager benchmark <model> --full

# Quantize a model
llm-manager quantize <model>
llm-manager quantize <model> --level q4_k_m

# View benchmark history
llm-manager history
```

---

## 📐 Evaluation Benchmarks

### TruthfulQA Subset (50 questions)

Tests model truthfulness — can it avoid common misconceptions, debunked health claims, and popular falsehoods?

**Categories:** Misconceptions · Health · Science · History

### MMLU Subset (80 questions)

Tests broad knowledge across academic disciplines via multiple-choice questions.

**Groups:**

| Group | Subjects | Questions |
|-------|----------|-----------|
| **STEM** | Computer Science, Physics, Chemistry, Biology, Mathematics, ML, Statistics | 30 |
| **Humanities** | Philosophy, Logic, History, Law, Religion | 15 |
| **Social Sciences** | Psychology, Sociology, Economics, Geography, Politics | 15 |
| **Other** | Medicine, Nutrition, Business, Management, Global Facts | 20 |

---

## ⚙️ How It Works

1. **Hardware Detection** — Queries `nvidia-smi`/`rocm-smi` for GPU info, `psutil`/`cpuinfo` for CPU/RAM
2. **Evaluation** — Builds structured multiple-choice prompts, sends to Ollama, extracts answer letters via regex
3. **Scoring** — Computes per-subject accuracy, latency percentiles (P50/P95), and throughput metrics
4. **Comparison** — Calculates accuracy delta, latency speedup, and a quality-efficiency tradeoff score
5. **Reports** — Generates standalone HTML with embedded CSS (dark theme, gradient bars, metrics cards)
6. **Storage** — All results persist in a local SQLite database for trend tracking

---

## 🗃️ Data Storage

Results are stored in a local SQLite database:

```
~/.local/share/local-llm-manager/benchmarks.db
```

Tables:
- `benchmarks` — Raw performance benchmarks (tokens/sec, load time)
- `eval_results` — Evaluation results (accuracy, per-subject scores, latency stats)

---

## 🛠️ Development

```bash
git clone https://github.com/arjun-arihant/local-llm-manager.git
cd local-llm-manager
pip install -e .

# Verify installation
llm-manager --help
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [Ollama](https://ollama.com) — making local LLMs accessible
- [Rich](https://rich.readthedocs.io) — beautiful terminal UI
- [TruthfulQA](https://github.com/sylinrl/TruthfulQA) — truthfulness benchmark
- [MMLU](https://github.com/hendrycks/test) — massive multitask language understanding
