# Local LLM Manager

An intelligent hardware-aware tool for managing local LLMs via Ollama.

## Features

- **Hardware Detection** - Automatically detect GPU, CPU, and RAM specs
- **Smart Model Recommendations** - Get model suggestions based on your hardware
- **Benchmarking** - Test and track model performance on your machine
- **One-click Install** - Pull models directly via Ollama
- **Quantization Helper** - Auto-quantize models for optimal hardware usage

## Installation

```bash
pip install local-llm-manager
```

Or install from source:

```bash
git clone https://github.com/arjun-arihant/local-llm-manager.git
cd local-llm-manager
pip install -e .
```

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running

## Usage

### Detect Hardware

```bash
llm-manager detect
```

Shows your GPU, CPU, and RAM specifications.

```
╭────────────────────╮
│ Hardware Detection │
╰────────────────────╯
                            System Hardware                             
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Component ┃ Property      ┃ Value                                    ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ CPU       │ Model         │ Intel(R) Core(TM) i5-8300H CPU @ 2.30GHz │
│ CPU       │ Cores/Threads │ 4/8                                      │
│ CPU       │ Architecture  │ X86_64                                   │
│ RAM       │ Total         │ 15.5 GB                                  │
│ RAM       │ Available     │ 12.1 GB                                  │
│ RAM       │ Used          │ 21.6%                                    │
│ GPU       │ Model         │ NVIDIA GeForce GTX 1050 Ti               │
│ GPU       │ VRAM          │ 6/4096 MB                                │
│ GPU       │ Driver        │ 535.288.01                               │
│ GPU       │ CUDA          │ Not available                            │
└───────────┴───────────────┴──────────────────────────────────────────┘
```

### Get Model Recommendations

```bash
llm-manager recommend
```

Suggests models based on your detected hardware capabilities.

```
╭───────────────────────╮
│ Model Recommendations │
╰───────────────────────╯
GPU: NVIDIA GeForce GTX 1050 Ti (4096MB VRAM)
RAM: 15.5GB

                      Recommended Models for Your Hardware                      
┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━┓
┃ Model      ┃ Description             ┃ Size   ┃ Requirements           ┃ Fit ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━┩
│ qwen2.5:3b │ Qwen2.5 3B - Alibaba's  │ 1.9 GB │ 2500MB VRAM, 4.0GB RAM │ 96% │
│ gemma:2b   │ Gemma 2B - Google's     │ 1.6 GB │ 2500MB VRAM, 4.0GB RAM │ 96% │
│ phi3:mini  │ Phi-3 Mini 3.8B -       │ 2.3 GB │ 3000MB VRAM, 4.0GB RAM │ 92% │
│ llama3.2   │ Llama 3.2 3B - Meta's   │ 2.0 GB │ 3500MB VRAM, 6.0GB RAM │ 80% │
│ tinyllama  │ TinyLlama 1.1B - Fast   │ 0.6 GB │ 1500MB VRAM, 2.0GB RAM │ 72% │
└────────────┴─────────────────────────┴────────┴────────────────────────┴─────┘
```

### Install a Model

```bash
llm-manager install llama3.2
```

Pulls the specified model via Ollama with progress tracking.

### List Installed Models

```bash
llm-manager list
```

Shows all locally available models with size and details.

### Benchmark a Model

```bash
llm-manager benchmark llama3.2
```

Tests model performance and stores results in local database.

```
╭──────────────────────────╮
│ Benchmarking: llama3.2   │
╰──────────────────────────╯
Running benchmark for llama3.2...
✓ Benchmark saved to database

           Benchmark Results           
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ Metric           ┃ Value             ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ Model            │ llama3.2          │
│ Tokens/sec       │ 12.45             │
│ Prompt eval rate │ 145.32 tok/s      │
│ Total time       │ 5234ms            │
│ Load time        │ 234ms             │
│ Tokens generated │ 45                │
│ Prompt tokens    │ 12                │
└──────────────────┴───────────────────┘
```

### Quantize a Model

```bash
llm-manager quantize llama3.2
```

Creates a quantized version optimized for your hardware.

```
╭──────────────────────────╮
│ Quantizing: llama3.2     │
╰──────────────────────────╯
Hardware: NVIDIA GeForce GTX 1050 Ti with 4096MB VRAM
Recommended quantization: 4-bit (K-means medium) - best balance, 75% smaller
✓ Quantized model created: llama3.2-q4_k_m
```

### View Benchmark History

```bash
llm-manager history
```

Shows your benchmark history with performance trends.

## Hardware Requirements

The tool automatically detects your hardware and recommends suitable models:

| Hardware | Recommended Models |
|----------|-------------------|
| Low-end (4GB VRAM) | TinyLlama, Phi-3 Mini, Qwen2.5-3B |
| Mid-range (8GB VRAM) | Llama 3.2, Mistral 7B, Gemma 2B |
| High-end (16GB+ VRAM) | Llama 3.1 70B, Mixtral, CodeLlama |

## Configuration

The tool stores benchmark history in a local SQLite database at:
```
~/.local/share/local-llm-manager/benchmarks.db
```

## Development

```bash
# Clone the repository
git clone https://github.com/arjun-arihant/local-llm-manager.git
cd local-llm-manager

# Install in development mode
pip install -e .

# Run tests
pytest tests/
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Ollama](https://ollama.com) for making local LLMs accessible
- [Rich](https://rich.readthedocs.io) for beautiful terminal UI
