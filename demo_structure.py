"""
Demo script to showcase the code structure and functionality.

This demonstrates the clean architecture without requiring model downloads.
"""

from core import ModelConfig
from inference import GreedySampler, TemperatureSampler, TopPSampler
from evaluation import extract_boxed_answer, check_mathematical_equality, grade_answer
from utils import get_device, print_device_info

def main():
    print("=" * 70)
    print("REASONING MODEL - DEMO")
    print("=" * 70)

    # 1. Device Detection
    print("\n1. Device Detection:")
    print("-" * 70)
    device = get_device()
    print_device_info(device)

    # 2. Model Configurations
    print("\n2. Available Model Configurations:")
    print("-" * 70)
    configs = {
        "Qwen3-0.6B": ModelConfig.qwen3_0_6b(),
        "Qwen3-1.7B": ModelConfig.qwen3_1_7b(),
        "Qwen3-4B": ModelConfig.qwen3_4b(),
        "Qwen3-8B": ModelConfig.qwen3_8b(),
    }

    for name, config in configs.items():
        params = (config.emb_dim * config.vocab_size +
                  config.n_layers * (config.emb_dim ** 2 * 3 + config.emb_dim * config.emb_dim * 3))
        print(f"{name:15} | Layers: {config.n_layers:2} | Dim: {config.emb_dim:4} | ~{params/1e9:.2f}B params")

    # 3. Sampling Strategies
    print("\n3. Available Sampling Strategies:")
    print("-" * 70)
    samplers = [
        ("Greedy", GreedySampler(), "Always select most likely token"),
        ("Temperature", TemperatureSampler(temperature=0.7), "Control randomness (temp=0.7)"),
        ("Top-p", TopPSampler(top_p=0.9, temperature=0.7), "Nucleus sampling (p=0.9)"),
    ]

    for name, sampler, desc in samplers:
        print(f"{name:15} | {desc}")

    # 4. Mathematical Verification
    print("\n4. Mathematical Answer Verification:")
    print("-" * 70)
    test_cases = [
        ("1/2", "0.5", "Fraction to decimal"),
        ("2**3", "8", "Exponentiation"),
        ("sqrt(16)", "4", "Square root"),
        ("(1, 2, 3)", "(1, 2, 3)", "Tuple matching"),
        ("x+1", "1+x", "Commutative equality"),
    ]

    for expr1, expr2, description in test_cases:
        result = check_mathematical_equality(expr1, expr2)
        status = "✓" if result else "✗"
        print(f"{status} {description:25} | {expr1:12} == {expr2:12}")

    # 5. Answer Extraction
    print("\n5. Answer Extraction from Generated Text:")
    print("-" * 70)
    test_texts = [
        "The answer is \\boxed{42}",
        "After calculation, we get \\boxed{\\frac{1}{2}}",
        "The solution is \\boxed{(2, 3, 5)}",
    ]

    for text in test_texts:
        answer = extract_boxed_answer(text)
        print(f"Text: {text[:50]}")
        print(f"  → Extracted: {answer}\n")

    # 6. Full Grading Example
    print("\n6. Complete Answer Grading:")
    print("-" * 70)
    problems = [
        ("\\boxed{1/2}", "0.5", "Fraction answer"),
        ("The answer is \\boxed{42}", "42", "Simple number"),
        ("\\boxed{\\frac{3}{4}}", "0.75", "LaTeX fraction"),
    ]

    for predicted, ground_truth, description in problems:
        is_correct = grade_answer(predicted, ground_truth)
        status = "✓ CORRECT" if is_correct else "✗ INCORRECT"
        print(f"{status:12} | {description:20} | Predicted: {predicted[:20]:20} | Truth: {ground_truth}")

    # 7. Architecture Overview
    print("\n7. Project Structure:")
    print("-" * 70)
    structure = """
    core/                  # Model architecture
      ├── config.py        # Model configurations
      ├── model.py         # Qwen3 implementation
      ├── attention.py     # GQA + RoPE
      └── tokenizer.py     # Tokenization

    inference/             # Text generation
      ├── generator.py     # TextGenerator class
      └── sampler.py       # Sampling strategies

    evaluation/            # Model evaluation
      ├── evaluator.py     # MathEvaluator
      └── math_verifier.py # Answer verification

    utils/                 # Utilities
      ├── device.py        # Device management
      ├── download.py      # Model downloading
      └── logger.py        # Logging
    """
    print(structure)

    print("\n8. Key Features:")
    print("-" * 70)
    features = [
        "✓ Clean, modular architecture",
        "✓ Comprehensive documentation",
        "✓ Type hints throughout",
        "✓ Multiple sampling strategies",
        "✓ Symbolic math verification",
        "✓ Multi-device support (CUDA/MPS/XPU/CPU)",
        "✓ KV caching for efficiency",
        "✓ Streaming generation",
        "✓ Self-consistency sampling",
        "✓ CLI interface",
    ]
    for feature in features:
        print(f"  {feature}")

    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("To generate text, download a model and run:")
    print("  python main.py generate 'What is 2+2?' --model-type base")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
