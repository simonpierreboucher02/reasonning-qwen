"""
Self-consistency sampling example.

This example demonstrates how to use self-consistency to improve
accuracy on reasoning tasks by generating multiple solutions and voting.
"""

import torch
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import ModelConfig, Qwen3Model, Qwen3Tokenizer
from inference import TextGenerator, TopPSampler, SelfConsistencySampler
from evaluation import extract_final_answer
from utils import setup_device, download_qwen3_model, get_logger


def main():
    # Setup logger
    logger = get_logger()
    logger.info("Starting self-consistency example")

    # Setup device
    device = setup_device(optimize_for_device=True)
    logger.info(f"Using device: {device}")

    # Download model and tokenizer
    logger.info("Downloading model files...")
    paths = download_qwen3_model(
        model_type="reasoning", size="0.6B", out_dir="../models"
    )

    # Load tokenizer with chat template
    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path=str(paths["tokenizer"]),
        apply_chat_template=True,
        add_generation_prompt=True,
        add_thinking=True,
    )

    # Create and load model
    logger.info("Loading model...")
    config = ModelConfig.qwen3_0_6b()
    model = Qwen3Model(config)
    model.load_state_dict(torch.load(paths["model"], map_location=device))
    model.to(device)
    model.eval()

    # Create generator with sampling
    generator = TextGenerator(
        model, tokenizer, device, sampler=TopPSampler(top_p=0.95, temperature=0.7)
    )

    # Math problem
    problem = """
    A store sells apples at $2 each and oranges at $3 each.
    If John buys 5 apples and 3 oranges, how much does he spend in total?
    """

    prompt = (
        f"Solve the following math problem step by step. "
        f"Put your final answer in \\boxed{{}}.\n\n{problem}"
    )

    # Create self-consistency sampler
    sc_sampler = SelfConsistencySampler(
        num_samples=5,
        sampler=TopPSampler(top_p=0.95, temperature=0.7),
        answer_extractor=extract_final_answer,
    )

    # Define generation function
    def generate_once():
        return generator.generate(prompt, max_new_tokens=256, use_cache=True)

    # Generate multiple solutions and vote
    logger.info("Generating multiple solutions with self-consistency...")
    print(f"\nProblem: {problem.strip()}\n")

    most_common_answer, all_generations = sc_sampler.generate_and_vote(generate_once)

    # Display results
    print("\nGenerated Solutions:")
    print("=" * 60)
    for i, gen in enumerate(all_generations, 1):
        answer = extract_final_answer(gen)
        print(f"\nSolution {i}:")
        print(f"Answer: {answer}")
        print(f"Full text: {gen[:200]}...")

    print("\n" + "=" * 60)
    print(f"\nMost Common Answer (by voting): {most_common_answer}")
    print("=" * 60)

    logger.info("Self-consistency example completed!")


if __name__ == "__main__":
    main()
