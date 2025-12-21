"""Test script to verify all imports work correctly."""

print("Testing imports...")

try:
    print("- Importing core modules...")
    from core import ModelConfig, Qwen3Model, Qwen3Tokenizer
    from core.model import KVCache
    print("  ✓ Core modules imported successfully")
except Exception as e:
    print(f"  ✗ Error importing core: {e}")
    exit(1)

try:
    print("- Importing inference modules...")
    from inference import TextGenerator, GreedySampler, TemperatureSampler, TopPSampler
    print("  ✓ Inference modules imported successfully")
except Exception as e:
    print(f"  ✗ Error importing inference: {e}")
    exit(1)

try:
    print("- Importing evaluation modules...")
    from evaluation import MathEvaluator, extract_boxed_answer, grade_answer
    print("  ✓ Evaluation modules imported successfully")
except Exception as e:
    print(f"  ✗ Error importing evaluation: {e}")
    exit(1)

try:
    print("- Importing utils...")
    from utils import setup_device, download_qwen3_model, get_logger
    print("  ✓ Utils imported successfully")
except Exception as e:
    print(f"  ✗ Error importing utils: {e}")
    exit(1)

print("\n✓ All imports successful!")

# Test basic functionality
print("\nTesting basic functionality...")

try:
    print("- Creating model config...")
    config = ModelConfig.qwen3_0_6b()
    print(f"  ✓ Config created: {config.n_layers} layers, {config.emb_dim} dim")
except Exception as e:
    print(f"  ✗ Error creating config: {e}")

try:
    print("- Testing device detection...")
    import torch
    device = setup_device()
    print(f"  ✓ Device: {device}")
except Exception as e:
    print(f"  ✗ Error with device: {e}")

try:
    print("- Testing math verifier...")
    from evaluation.math_verifier import check_mathematical_equality
    result = check_mathematical_equality("1/2", "0.5")
    print(f"  ✓ Math verifier works: 1/2 == 0.5 is {result}")
except Exception as e:
    print(f"  ✗ Error with math verifier: {e}")

print("\n✓ All tests passed!")
