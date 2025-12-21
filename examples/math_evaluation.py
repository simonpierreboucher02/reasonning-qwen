"""
Mathematical reasoning evaluation example.

This example demonstrates how to evaluate a model on mathematical
reasoning problems using the MATH dataset.
"""

import torch
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import ModelConfig, Qwen3Model, Qwen3Tokenizer
from inference import TextGenerator, GreedySampler
from evaluation import MathEvaluator
from utils import setup_device, download_qwen3_model, get_logger


def main():
    # Setup logger
    logger = get_logger()
    logger.info("Starting math evaluation example")

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

    # Create generator
    generator = TextGenerator(model, tokenizer, device, sampler=GreedySampler())

    # Create evaluator
    def generate_fn(prompt: str) -> str:
        return generator.generate(
            prompt, max_new_tokens=512, use_cache=True, chat_wrapped=False
        )

    prompt_template = (
        "Solve the following math problem step by step. "
        "Show your reasoning and put your final answer in \\boxed{{}}.\n\n"
        "{problem}"
    )

    evaluator = MathEvaluator(
        generate_fn=generate_fn, prompt_template=prompt_template
    )

    # Example problems
    test_problems = [
        {
            "id": "1",
            "problem": "What is 2 + 2?",
            "answer": "4",
        },
        {
            "id": "2",
            "problem": "If a triangle has angles of 60 degrees each, what type of triangle is it?",
            "answer": "equilateral",
        },
        {
            "id": "3",
            "problem": "Solve for x: 2x + 5 = 13",
            "answer": "4",
        },
    ]

    logger.info("Evaluating on example problems...")
    results = evaluator.evaluate_dataset(
        test_problems, problem_key="problem", answer_key="answer", verbose=True
    )

    # Print summary
    evaluator.print_summary(results)

    # Save results
    output_path = Path("../results/example_evaluation.json")
    evaluator.save_results(results, str(output_path))

    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
