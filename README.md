<div align="center">

# 🧠 Reasoning-Qwen

### Clean Architecture Implementation of Reasoning Models Based on Qwen3

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green?style=for-the-badge&logo=apache&logoColor=white)](LICENSE)
[![Version](https://img.shields.io/badge/Version-1.0.0-blueviolet?style=for-the-badge)](setup.py)
[![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)](https://github.com/simonpierreboucher02/reasonning-qwen)

[![CUDA](https://img.shields.io/badge/CUDA-Supported-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![MPS](https://img.shields.io/badge/Apple%20Silicon-MPS%20Supported-999999?style=flat-square&logo=apple&logoColor=white)](https://developer.apple.com/metal/)
[![Intel XPU](https://img.shields.io/badge/Intel-XPU%20Supported-0071C5?style=flat-square&logo=intel&logoColor=white)](https://www.intel.com/)
[![CPU](https://img.shields.io/badge/CPU-Fallback-lightgrey?style=flat-square)](https://github.com/simonpierreboucher02/reasonning-qwen)

[![SymPy](https://img.shields.io/badge/SymPy-Math%20Verification-3B5526?style=flat-square&logo=sympy&logoColor=white)](https://www.sympy.org/)
[![Tokenizers](https://img.shields.io/badge/Tokenizers-0.21.2%2B-FFD700?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/docs/tokenizers/)
[![Code Style](https://img.shields.io/badge/Code%20Style-Clean%20Architecture-informational?style=flat-square)](https://github.com/simonpierreboucher02/reasonning-qwen)
[![Type Hints](https://img.shields.io/badge/Type%20Hints-Full%20Annotations-blueviolet?style=flat-square)](https://docs.python.org/3/library/typing.html)

[![GitHub stars](https://img.shields.io/github/stars/simonpierreboucher02/reasonning-qwen?style=social)](https://github.com/simonpierreboucher02/reasonning-qwen/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/simonpierreboucher02/reasonning-qwen?style=social)](https://github.com/simonpierreboucher02/reasonning-qwen/network/members)
[![GitHub issues](https://img.shields.io/github/issues/simonpierreboucher02/reasonning-qwen?style=social)](https://github.com/simonpierreboucher02/reasonning-qwen/issues)

</div>

---

## 👤 Authors

<table>
  <tr>
    <td align="center">
      <strong>Simon-Pierre Boucher</strong><br/>
      <a href="mailto:spbou4@protonmail.com">📧 spbou4@protonmail.com</a><br/>
      <a href="https://www.spboucher.ai">🌐 www.spboucher.ai</a><br/>
      <a href="https://github.com/simonpierreboucher02">GitHub @simonpierreboucher02</a>
    </td>
    <td align="center">
      <strong>Claude (Anthropic)</strong><br/>
      AI Assistant &amp; Co-Author<br/>
      <a href="https://www.anthropic.com">🌐 www.anthropic.com</a><br/>
      <em>Architecture &amp; Documentation</em>
    </td>
  </tr>
</table>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Metrics](#-key-metrics)
- [Features](#-features)
- [Architecture](#️-architecture)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [CLI Usage](#️-cli-usage)
- [Sampling Strategies](#-sampling-strategies)
- [Evaluation Framework](#-evaluation-framework)
- [Device Support](#-device-support)
- [Model Variants](#-model-variants)
- [Performance Tips](#-performance-tips)
- [Examples](#-examples)
- [Credits](#-credits)
- [License](#-license)

---

## 🔍 Overview

**Reasoning-Qwen** is a clean, production-ready implementation of reasoning models based on the **Qwen3 architecture**. Built from scratch with a focus on code clarity, modularity, and extensibility, this framework enables you to run, evaluate, and extend Qwen3 reasoning models with minimal boilerplate.

This project is a refactored and extended version of Sebastian Raschka's work from *"Build a Reasoning Model (From Scratch)"*, enhanced with:

- Full **Clean Architecture** separation of concerns
- **Multi-device** support (CUDA / Apple Silicon MPS / Intel XPU / CPU)
- **Mathematical reasoning** evaluation with SymPy verification
- **Self-consistency** sampling for improved accuracy
- Complete **type annotations** throughout the codebase
- Ready-to-run **CLI** and **examples**

---

## 📊 Key Metrics

<div align="center">

| Metric | Value |
|---|---|
| 🐍 Python Files | 25 |
| 📦 Modules | 4 core modules |
| 🔢 Model Parameters | 0.6B → 32B (configurable) |
| 🎯 Sampling Strategies | 4 (greedy, temperature, top-p, self-consistency) |
| 🧮 Math Verifier | SymPy-based symbolic verification |
| 💾 KV Cache | Full KV caching support |
| 🖥️ Devices Supported | CUDA, MPS, XPU, CPU |
| 🏗️ Architecture | Grouped Query Attention (GQA) + RoPE |
| 📐 Context Length | Up to 40,960 tokens |
| 🔤 Vocabulary Size | 151,936 tokens |
| ⚡ Streaming | Token-by-token streaming supported |
| 🧪 Test Coverage | Comprehensive test suite included |
| 📄 License | Apache 2.0 |
| 🏷️ Version | 1.0.0 |

</div>

---

## ✨ Features

### Core Capabilities

| Feature | Description |
|---|---|
| 🏗️ **Clean Architecture** | Modular design with clear separation of concerns |
| 📚 **Full Documentation** | Docstrings for every module, class, and function |
| 🔧 **Modular Design** | Easy to extend with new components |
| 🎯 **Multiple Samplers** | Greedy, temperature, top-p, and self-consistency |
| 🧮 **Math Reasoning** | Built-in evaluation framework for math problems |
| ⚡ **KV Caching** | Efficient key-value caching for fast generation |
| 🖥️ **Multi-Device** | CUDA, MPS (Apple Silicon), XPU (Intel), CPU |
| 🏷️ **Type Hints** | Full type annotations for excellent IDE support |
| 🌊 **Streaming** | Real-time token-by-token output |
| 📦 **Batch Generation** | Process multiple prompts efficiently |

### Model Architecture Highlights

| Component | Details |
|---|---|
| **Attention** | Grouped Query Attention (GQA) |
| **Positional Encoding** | Rotary Position Embedding (RoPE) |
| **Normalization** | RMS Normalization |
| **Activation** | SiLU (Swish) |
| **Precision** | TF32 optimization on CUDA |

---

## 🏛️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Reasoning-Qwen                       │
├───────────────┬───────────────┬───────────┬─────────────┤
│     core/     │  inference/   │evaluation/│    utils/   │
├───────────────┼───────────────┼───────────┼─────────────┤
│ config.py     │ generator.py  │evaluator.py│ device.py  │
│ model.py      │ sampler.py    │math_verifier│download.py│
│ attention.py  │               │           │ logger.py   │
│ tokenizer.py  │               │           │             │
└───────────────┴───────────────┴───────────┴─────────────┘
                         │
              ┌──────────┴──────────┐
              │      main.py        │
              │   (CLI Entry Point) │
              └──────────┬──────────┘
                         │
              ┌──────────┴──────────┐
              │     examples/       │
              │ basic_generation.py │
              │ math_evaluation.py  │
              │ self_consistency.py │
              └─────────────────────┘
```

### Data Flow

```
User Input (prompt)
       │
       ▼
  Tokenizer ──────► Token IDs
       │
       ▼
  Qwen3 Model (GQA + RoPE + RMSNorm)
       │
       ▼
  Sampler (Greedy / Temperature / Top-p / Self-Consistency)
       │
       ▼
  Detokenizer ──────► Generated Text
       │
       ▼
  [Optional] Math Evaluator (SymPy verification)
```

---

## 📁 Project Structure

```
reasonning-qwen/
│
├── core/                          # 🧠 Core model components
│   ├── __init__.py
│   ├── config.py                  # Model configurations (0.6B → 32B)
│   ├── model.py                   # Qwen3 transformer implementation
│   ├── attention.py               # GQA + RoPE attention mechanisms
│   └── tokenizer.py               # Tokenizer with chat templates
│
├── inference/                     # ⚡ Text generation engine
│   ├── __init__.py
│   ├── generator.py               # Main text generator (stream + batch)
│   └── sampler.py                 # All sampling strategies
│
├── evaluation/                    # 📊 Evaluation framework
│   ├── __init__.py
│   ├── evaluator.py               # Dataset evaluation + result tracking
│   └── math_verifier.py           # SymPy-based mathematical verification
│
├── utils/                         # 🔧 Utility helpers
│   ├── __init__.py
│   ├── device.py                  # CUDA/MPS/XPU/CPU device management
│   ├── download.py                # Hugging Face model downloading
│   └── logger.py                  # Structured logging utilities
│
├── examples/                      # 📖 Ready-to-run examples
│   ├── basic_generation.py        # Generation with multiple samplers
│   ├── math_evaluation.py         # Evaluate on math datasets
│   └── self_consistency.py        # Self-consistency voting
│
├── main.py                        # 🖥️  CLI entry point
├── setup.py                       # Package setup
├── requirements.txt               # Dependencies
├── comprehensive_tests.py         # Full test suite
├── test_imports.py                # Import validation tests
├── FILES_INDEX.md                 # Complete file index
├── MODEL_EVALUATION_REPORT.md     # Evaluation report
├── PROJECT_COMPLETE.md            # Project summary
└── README.md                      # This file
```

---

## 📦 Installation

### Prerequisites

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://www.python.org/downloads/)
[![PyTorch 2.7.1+](https://img.shields.io/badge/PyTorch-2.7.1%2B-red?style=flat-square&logo=pytorch)](https://pytorch.org/get-started/locally/)

### Step 1 — Clone the Repository

```bash
git clone https://github.com/simonpierreboucher02/reasonning-qwen.git
cd reasonning-qwen
```

### Step 2 — Create Virtual Environment (Recommended)

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate         # Windows
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e .
```

### Step 4 — Verify Installation

```bash
python test_imports.py
```

Expected output:
```
✅ core imports OK
✅ inference imports OK
✅ evaluation imports OK
✅ utils imports OK
```

---

## 🚀 Quick Start

### 1. Basic Text Generation

```python
import torch
from core import ModelConfig, Qwen3Model, Qwen3Tokenizer
from inference import TextGenerator, GreedySampler
from utils import setup_device, download_qwen3_model

# Setup device (auto-detects CUDA / MPS / XPU / CPU)
device = setup_device()

# Download and load model
paths = download_qwen3_model(model_type="base", size="0.6B")
tokenizer = Qwen3Tokenizer(tokenizer_file_path=str(paths["tokenizer"]))

config = ModelConfig.qwen3_0_6b()
model = Qwen3Model(config)
model.load_state_dict(torch.load(paths["model"], map_location=device))
model.to(device).eval()

# Generate text
generator = TextGenerator(model, tokenizer, device, sampler=GreedySampler())
result = generator.generate("What is the capital of France?", max_new_tokens=50)
print(result)
```

### 2. Mathematical Reasoning

```python
from evaluation import MathEvaluator

def generate_fn(prompt: str) -> str:
    return generator.generate(prompt, max_new_tokens=512, use_cache=True)

evaluator = MathEvaluator(generate_fn=generate_fn)
results = evaluator.load_and_evaluate("path/to/dataset.json")
evaluator.print_summary(results)
```

### 3. Streaming Output

```python
print("Generating: ", end="", flush=True)
for token in generator.generate_stream("Explain quantum entanglement", max_new_tokens=200):
    print(token, end="", flush=True)
print()
```

### 4. Batch Generation

```python
prompts = [
    "What is 2 + 2?",
    "Explain the Pythagorean theorem.",
    "Solve: x^2 - 5x + 6 = 0",
]
results = generator.batch_generate(prompts, max_new_tokens=100)
for prompt, result in zip(prompts, results):
    print(f"Q: {prompt}\nA: {result}\n")
```

---

## 🖥️ CLI Usage

The project ships with a full-featured CLI via `main.py`.

### Generate Text

```bash
# Greedy decoding
python main.py generate "What is 2+2?" --model-type reasoning

# Temperature sampling with streaming
python main.py generate "Tell me a story" \
    --model-type base \
    --sampler temperature \
    --temperature 0.8 \
    --stream

# Top-p nucleus sampling
python main.py generate "Explain GQA attention" \
    --sampler top_p \
    --top-p 0.95 \
    --temperature 0.7 \
    --max-tokens 512

# With chat template
python main.py generate "Solve: x^2 = 16" \
    --model-type reasoning \
    --chat-template \
    --stream
```

### CLI Options

| Flag | Type | Default | Description |
|---|---|---|---|
| `--model-type` | `str` | `base` | `base` or `reasoning` |
| `--model-size` | `str` | `0.6B` | Model size |
| `--max-tokens` | `int` | `256` | Max tokens to generate |
| `--sampler` | `str` | `greedy` | `greedy`, `temperature`, `top_p` |
| `--temperature` | `float` | `0.7` | Sampling temperature |
| `--top-p` | `float` | `0.9` | Nucleus sampling threshold |
| `--stream` | `flag` | `False` | Stream token by token |
| `--chat-template` | `flag` | `False` | Use chat template |

### Evaluate on Dataset

```bash
python main.py evaluate dataset.json \
    --output results.json \
    --model-type reasoning \
    --max-tokens 512 \
    --max-problems 100
```

---

## 🎯 Sampling Strategies

### Comparison Table

| Strategy | Deterministic | Best For | Speed |
|---|---|---|---|
| **Greedy** | ✅ Yes | Factual Q&A, math | ⚡ Fastest |
| **Temperature** | ❌ No | Creative text | ⚡ Fast |
| **Top-p (Nucleus)** | ❌ No | Balanced quality | ⚡ Fast |
| **Self-Consistency** | ❌ No | Complex reasoning | 🐢 Slower (N×) |

### Code Examples

```python
from inference import GreedySampler, TemperatureSampler, TopPSampler, SelfConsistencySampler
from evaluation import extract_final_answer

# Greedy — always most likely token
sampler = GreedySampler()

# Temperature — control creativity
sampler = TemperatureSampler(temperature=0.7)

# Top-p — nucleus sampling
sampler = TopPSampler(top_p=0.95, temperature=0.7)

# Self-Consistency — majority vote across N samples
sampler = SelfConsistencySampler(
    num_samples=5,
    sampler=TopPSampler(top_p=0.95, temperature=0.7),
    answer_extractor=extract_final_answer,
)

def generate_once():
    return generator.generate(prompt, max_new_tokens=256)

answer, all_generations = sampler.generate_and_vote(generate_once)
print(f"Consensus answer: {answer}")
```

---

## 📊 Evaluation Framework

The evaluation module provides end-to-end math reasoning assessment.

### Features

| Feature | Description |
|---|---|
| 📝 **Answer Extraction** | Automatic extraction from chain-of-thought |
| 🔢 **LaTeX Normalization** | Convert LaTeX expressions to evaluable form |
| 🧮 **SymPy Verification** | Symbolic math equality checking |
| 📋 **Tuple/List Support** | Multi-answer problem support |
| 📈 **Result Tracking** | Full history with per-problem breakdown |
| 💾 **JSON Export** | Save results for later analysis |

### Usage

```python
from evaluation import MathEvaluator, extract_final_answer

evaluator = MathEvaluator(generate_fn=generate_fn)

# Evaluate single problem
is_correct = evaluator.evaluate_single(
    problem="What is the integral of x^2?",
    ground_truth="x^3/3 + C"
)

# Evaluate full dataset
results = evaluator.load_and_evaluate("gsm8k_test.json")
evaluator.print_summary(results)
# Output:
# Accuracy: 78.4% (784/1000)
# Avg tokens/problem: 312
# Total time: 142.3s
```

---

## 🖥️ Device Support

```python
from utils import setup_device, print_device_info

device = setup_device(optimize_for_device=True)
print_device_info(device)
```

### Priority Order

```
1. CUDA   (NVIDIA GPU)    → TF32 optimization enabled
2. MPS    (Apple Silicon) → Native Metal GPU acceleration
3. XPU    (Intel GPU)     → Intel GPU support
4. CPU    (fallback)      → Works on any machine
```

### Device-Specific Optimizations

| Device | Optimization | Notes |
|---|---|---|
| **CUDA** | TF32 enabled | Best performance |
| **MPS** | Metal backend | Great on M1/M2/M3/M4 |
| **XPU** | Intel native | Intel Arc/Xe GPUs |
| **CPU** | Multi-thread | Universal fallback |

---

## 🔧 Model Variants

### Supported Configurations

| Model | Parameters | Layers | Heads | KV Groups | Context |
|---|---|---|---|---|---|
| **Qwen3-0.6B** | ~600M | 28 | 16 | 8 | 40,960 |
| **Qwen3-1.7B** | ~1.7B | 28 | 16 | 8 | 40,960 |
| **Qwen3-4B** | ~4B | 36 | 32 | 8 | 40,960 |
| **Qwen3-8B** | ~8B | 36 | 32 | 8 | 40,960 |
| **Qwen3-14B** | ~14B | 40 | 40 | 8 | 40,960 |
| **Qwen3-32B** | ~32B | 64 | 64 | 8 | 40,960 |

### Custom Configuration

```python
from core import ModelConfig

# Predefined config
config = ModelConfig.qwen3_0_6b()

# Custom config
config = ModelConfig(
    vocab_size=151_936,
    context_length=40_960,
    emb_dim=1024,
    n_heads=16,
    n_layers=28,
    n_kv_groups=8,
    rope_base=1_000_000.0,
)
```

---

## ⚡ Performance Tips

### 1. Enable KV Caching

```python
# Always use use_cache=True for long generations
result = generator.generate(prompt, max_new_tokens=512, use_cache=True)
```

### 2. Use Streaming for Long Outputs

```python
# Avoid holding large tensors in memory
for token in generator.generate_stream(prompt, max_new_tokens=1000):
    print(token, end="", flush=True)
```

### 3. Batch Processing

```python
# More efficient than sequential calls
results = generator.batch_generate(prompts, max_new_tokens=100)
```

### 4. Device Optimization

```python
# Auto-detect and optimize for your hardware
device = setup_device(optimize_for_device=True)
```

### 5. Precision

```python
# For CUDA: TF32 is enabled automatically
# For CPU: consider float16 for memory savings
model = model.half()  # float16
```

---

## 📖 Examples

Run the included examples:

```bash
# Basic generation with all sampling strategies
python examples/basic_generation.py

# Evaluate on mathematical reasoning benchmarks
python examples/math_evaluation.py

# Self-consistency for improved accuracy
python examples/self_consistency.py

# Demo the project structure
python demo_structure.py

# Run full test suite
python comprehensive_tests.py
```

---

## 🏗️ Extending the Framework

### Add a New Sampling Strategy

```python
from inference.sampler import BaseSampler
import torch

class MyCustomSampler(BaseSampler):
    def __init__(self, my_param: float = 1.0):
        self.my_param = my_param

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        # Your custom sampling logic here
        probs = torch.softmax(logits / self.my_param, dim=-1)
        return torch.multinomial(probs, num_samples=1)
```

### Add a New Evaluation Metric

```python
from evaluation.evaluator import BaseEvaluator

class MyEvaluator(BaseEvaluator):
    def evaluate_single(self, prediction: str, ground_truth: str) -> bool:
        # Your custom evaluation logic
        return prediction.strip() == ground_truth.strip()
```

---

## 🧪 Testing

```bash
# Quick import test
python test_imports.py

# Full test suite
python comprehensive_tests.py

# If pytest is installed
pytest comprehensive_tests.py -v
```

---

## 📜 Credits

This repository is a refactored and extended version of work from:

> **"Build a Reasoning Model (From Scratch)"** — Sebastian Raschka
> - Original Repo: [rasbt/reasoning-from-scratch](https://github.com/rasbt/reasoning-from-scratch)
> - Book: [https://mng.bz/lZ5B](https://mng.bz/lZ5B)
> - License: Apache 2.0

### Improvements Over Original

| Area | Enhancement |
|---|---|
| 🗂️ **Organization** | Clear 4-module structure (core / inference / evaluation / utils) |
| 📚 **Documentation** | Docstrings on every function, class, and module |
| 🏷️ **Type Safety** | Full type annotations throughout |
| 🎯 **API Design** | More intuitive and consistent interfaces |
| 📖 **Examples** | 3 ready-to-run example scripts |
| 🖥️ **CLI** | Full command-line interface |
| 🛡️ **Error Handling** | Better validation and error messages |
| 🔌 **Extensibility** | Easy to add samplers, evaluators, model configs |

---

## 📄 License

[![Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green?style=for-the-badge&logo=apache)](LICENSE)

This project is licensed under the **Apache 2.0 License** — the same as the original repository.

You are free to use, modify, and distribute this code for both personal and commercial purposes, provided you include the original license and attribution.

---

## 🤝 Contributing

Contributions are welcome! Feel free to extend this framework with:

- Additional model size configurations
- New sampling strategies
- Custom evaluation metrics
- Performance optimizations (quantization, flash attention)
- Better visualization tools
- Additional model architectures

---

## 📬 Contact & Support

| Resource | Link |
|---|---|
| 🌐 Website | [www.spboucher.ai](https://www.spboucher.ai) |
| 📧 Email | [spbou4@protonmail.com](mailto:spbou4@protonmail.com) |
| 🐙 GitHub | [@simonpierreboucher02](https://github.com/simonpierreboucher02) |
| 🤖 Original Repo | [rasbt/reasoning-from-scratch](https://github.com/rasbt/reasoning-from-scratch) |
| 🐛 Issues | [Open an Issue](https://github.com/simonpierreboucher02/reasonning-qwen/issues) |

---

<div align="center">

Made with ❤️ by **Simon-Pierre Boucher** & **Claude (Anthropic)**

[![Website](https://img.shields.io/badge/🌐-www.spboucher.ai-blue?style=for-the-badge)](https://www.spboucher.ai)

</div>
