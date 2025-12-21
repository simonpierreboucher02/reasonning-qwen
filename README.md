# Reasoning Model - Clean Architecture

A clean, well-structured implementation of reasoning models based on Qwen3 architecture. This is a refactored version with improved code organization, comprehensive documentation, and modular design.

## Features

- **Clean Architecture**: Well-organized modules with clear separation of concerns
- **Comprehensive Documentation**: Every module, class, and function is thoroughly documented
- **Modular Design**: Easy to extend and customize
- **Multiple Sampling Strategies**: Greedy, temperature, top-p, and self-consistency
- **Mathematical Reasoning**: Built-in evaluation framework for math problems
- **Efficient Generation**: KV caching for fast autoregressive generation
- **Multi-Device Support**: CUDA, MPS (Apple Silicon), XPU (Intel), and CPU
- **Type Hints**: Full type annotations for better IDE support

## Project Structure

```
main/
├── core/                      # Core model components
│   ├── __init__.py
│   ├── config.py             # Model configurations
│   ├── model.py              # Qwen3 model implementation
│   ├── attention.py          # Attention mechanisms (GQA, RoPE)
│   └── tokenizer.py          # Tokenizer with chat templates
│
├── inference/                 # Text generation
│   ├── __init__.py
│   ├── generator.py          # Text generator
│   └── sampler.py            # Sampling strategies
│
├── evaluation/                # Model evaluation
│   ├── __init__.py
│   ├── evaluator.py          # Evaluation framework
│   └── math_verifier.py      # Mathematical verification
│
├── utils/                     # Utilities
│   ├── __init__.py
│   ├── device.py             # Device management
│   ├── download.py           # Model downloading
│   └── logger.py             # Logging utilities
│
├── examples/                  # Usage examples
│   ├── basic_generation.py   # Text generation example
│   ├── math_evaluation.py    # Math evaluation example
│   └── self_consistency.py   # Self-consistency example
│
├── main.py                   # CLI entry point
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## Installation

### Requirements

- Python 3.10 or higher
- PyTorch 2.7.1 or higher

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Basic Text Generation

```python
import torch
from core import ModelConfig, Qwen3Model, Qwen3Tokenizer
from inference import TextGenerator, GreedySampler
from utils import setup_device, download_qwen3_model

# Setup device
device = setup_device()

# Download and load model
paths = download_qwen3_model(model_type="base", size="0.6B")
tokenizer = Qwen3Tokenizer(tokenizer_file_path=str(paths["tokenizer"]))
config = ModelConfig.qwen3_0_6b()
model = Qwen3Model(config)
model.load_state_dict(torch.load(paths["model"], map_location=device))
model.to(device)
model.eval()

# Generate text
generator = TextGenerator(model, tokenizer, device, sampler=GreedySampler())
result = generator.generate("What is the capital of France?", max_new_tokens=50)
print(result)
```

### 2. Mathematical Reasoning

```python
from evaluation import MathEvaluator

# Create generator function
def generate_fn(prompt: str) -> str:
    return generator.generate(prompt, max_new_tokens=512, use_cache=True)

# Create evaluator
evaluator = MathEvaluator(generate_fn=generate_fn)

# Evaluate on dataset
results = evaluator.load_and_evaluate("path/to/dataset.json")
evaluator.print_summary(results)
```

### 3. Self-Consistency Sampling

```python
from inference import TopPSampler, SelfConsistencySampler
from evaluation import extract_final_answer

# Create self-consistency sampler
sc_sampler = SelfConsistencySampler(
    num_samples=5,
    sampler=TopPSampler(top_p=0.95, temperature=0.7),
    answer_extractor=extract_final_answer,
)

# Generate and vote
def generate_once():
    return generator.generate(prompt, max_new_tokens=256)

answer, generations = sc_sampler.generate_and_vote(generate_once)
print(f"Most common answer: {answer}")
```

## CLI Usage

The project includes a command-line interface for common tasks.

### Generate Text

```bash
python main.py generate "What is 2+2?" --model-type reasoning --stream
```

Options:
- `--model-type`: Model type (`base` or `reasoning`)
- `--model-size`: Model size (default: `0.6B`)
- `--max-tokens`: Maximum tokens to generate
- `--sampler`: Sampling strategy (`greedy`, `temperature`, `top_p`)
- `--temperature`: Temperature for sampling
- `--top-p`: Top-p value for nucleus sampling
- `--stream`: Stream output token by token
- `--chat-template`: Use chat template formatting

### Evaluate on Dataset

```bash
python main.py evaluate dataset.json --output results.json
```

Options:
- `--model-type`: Model type (`base` or `reasoning`)
- `--max-tokens`: Maximum tokens per problem
- `--max-problems`: Limit number of problems
- `--output`: Save results to file
- `--greedy`: Use greedy decoding (faster)

## Examples

Run the included examples to see the framework in action:

```bash
# Basic generation with different sampling strategies
python examples/basic_generation.py

# Evaluate on mathematical problems
python examples/math_evaluation.py

# Self-consistency for improved accuracy
python examples/self_consistency.py
```

## Architecture Details

### Core Components

#### Model Configuration
```python
from core import ModelConfig

# Load predefined configuration
config = ModelConfig.qwen3_0_6b()

# Or create custom configuration
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

#### Sampling Strategies

**Greedy Sampling**: Always select most likely token
```python
from inference import GreedySampler
sampler = GreedySampler()
```

**Temperature Sampling**: Control randomness
```python
from inference import TemperatureSampler
sampler = TemperatureSampler(temperature=0.7)
```

**Top-p Sampling**: Nucleus sampling
```python
from inference import TopPSampler
sampler = TopPSampler(top_p=0.9, temperature=0.7)
```

**Self-Consistency**: Multiple samples with voting
```python
from inference import SelfConsistencySampler
sampler = SelfConsistencySampler(num_samples=5)
```

### Evaluation Framework

The evaluation framework provides:
- Answer extraction from generated text
- LaTeX normalization
- Symbolic mathematical verification using SymPy
- Support for tuples and lists
- Comprehensive result tracking

## Device Support

The framework automatically detects and uses the best available device:

1. **CUDA** (NVIDIA GPUs): Full support with TF32 optimization
2. **MPS** (Apple Silicon): Native macOS GPU acceleration
3. **XPU** (Intel GPUs): Intel GPU support
4. **CPU**: Fallback for systems without GPU

```python
from utils import setup_device, print_device_info

device = setup_device(optimize_for_device=True)
print_device_info(device)
```

## Model Variants

Currently supported:
- **Qwen3-0.6B-base**: Pre-trained base model
- **Qwen3-0.6B-reasoning**: Fine-tuned for reasoning tasks

Configurations available for larger models (1.7B, 4B, 8B, 14B, 32B).

## Performance Tips

### Use KV Caching
Enable KV caching for faster generation:
```python
result = generator.generate(prompt, max_new_tokens=100, use_cache=True)
```

### Batch Processing
For multiple prompts, use batch generation:
```python
results = generator.batch_generate(prompts, max_new_tokens=50)
```

### Streaming
For real-time display, use streaming:
```python
for token in generator.generate_stream(prompt, max_new_tokens=100):
    print(token, end="", flush=True)
```

## Testing

The original repository includes comprehensive tests. To run tests:

```bash
pytest ../tests/
```

## Credits

This is a refactored version of the code from the book:
**"Build a Reasoning Model (From Scratch)"** by Sebastian Raschka

- Original Repository: https://github.com/rasbt/reasoning-from-scratch
- Book: https://mng.bz/lZ5B
- License: Apache 2.0

## Improvements Over Original

This refactored version provides:

1. **Better Organization**: Clear module structure with separation of concerns
2. **Comprehensive Documentation**: Docstrings for all components
3. **Type Hints**: Full type annotations throughout
4. **Improved API**: More intuitive and consistent interfaces
5. **Examples**: Ready-to-run examples for common use cases
6. **CLI Tool**: Command-line interface for quick tasks
7. **Error Handling**: Better error messages and validation
8. **Extensibility**: Easy to add new sampling strategies or evaluation metrics

## License

Apache 2.0 License (same as original repository)

## Contributing

Feel free to extend this framework with:
- Additional model sizes
- New sampling strategies
- Custom evaluation metrics
- Performance optimizations
- Better visualization tools

## Support

For issues related to:
- **Original implementation**: See https://github.com/rasbt/reasoning-from-scratch
- **This refactored version**: Create an issue in this repository
