"""
Basic text generation example.

This example demonstrates how to load a model and generate text with
different sampling strategies.
"""

import torch
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import ModelConfig, Qwen3Model, Qwen3Tokenizer
from core.model import KVCache
from inference import TextGenerator, GreedySampler, TemperatureSampler, TopPSampler
from utils import setup_device, download_qwen3_model, get_logger


def main():
    # Setup logger
    logger = get_logger()
    logger.info("Starting basic text generation example")

    # Setup device
    device = setup_device(optimize_for_device=True)
    logger.info(f"Using device: {device}")

    # Download model and tokenizer
    logger.info("Downloading model files...")
    paths = download_qwen3_model(model_type="base", size="0.6B", out_dir="../models")

    # Load tokenizer
    tokenizer = Qwen3Tokenizer(tokenizer_file_path=str(paths["tokenizer"]))

    # Create and load model
    logger.info("Loading model...")
    config = ModelConfig.qwen3_0_6b()
    model = Qwen3Model(config)
    model.load_state_dict(torch.load(paths["model"], map_location=device))
    model.to(device)
    model.eval()

    # Create generator with greedy sampling
    logger.info("Generating text with greedy sampling...")
    generator = TextGenerator(model, tokenizer, device, sampler=GreedySampler())

    prompt = "What is the capital of France?"
    result = generator.generate(prompt, max_new_tokens=50, use_cache=True)
    print(f"\nPrompt: {prompt}")
    print(f"Greedy: {result}\n")

    # Try with temperature sampling
    logger.info("Generating text with temperature sampling...")
    generator.set_sampler(TemperatureSampler(temperature=0.7))
    result = generator.generate(prompt, max_new_tokens=50, use_cache=True)
    print(f"Temperature (0.7): {result}\n")

    # Try with top-p sampling
    logger.info("Generating text with top-p sampling...")
    generator.set_sampler(TopPSampler(top_p=0.9, temperature=0.7))
    result = generator.generate(prompt, max_new_tokens=50, use_cache=True)
    print(f"Top-p (0.9): {result}\n")

    # Try streaming generation
    logger.info("Generating text with streaming...")
    generator.set_sampler(GreedySampler())
    print("Streaming: ", end="", flush=True)
    for token in generator.generate_stream(prompt, max_new_tokens=50, use_cache=True):
        print(token, end="", flush=True)
    print("\n")

    logger.info("Example completed!")


if __name__ == "__main__":
    main()
