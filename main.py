"""
Main entry point for the reasoning model.

This script provides a simple CLI for loading and using the reasoning model
for text generation and mathematical reasoning.
"""

import argparse
import torch
from pathlib import Path

from core import ModelConfig, Qwen3Model, Qwen3Tokenizer
from inference import TextGenerator, GreedySampler, TemperatureSampler, TopPSampler
from evaluation import MathEvaluator
from utils import setup_device, download_qwen3_model, get_logger, print_device_info


def generate_text(args):
    """Generate text from a prompt."""
    logger = get_logger()

    # Setup device
    device = setup_device(optimize_for_device=True)
    print_device_info(device)

    # Download and load model
    logger.info("Loading model...")
    paths = download_qwen3_model(
        model_type=args.model_type, size=args.model_size, out_dir=args.model_dir
    )

    # Load tokenizer
    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path=str(paths["tokenizer"]),
        apply_chat_template=args.chat_template,
        add_generation_prompt=args.chat_template,
        add_thinking=args.model_type == "reasoning",
    )

    # Load model
    config = ModelConfig.qwen3_0_6b()
    model = Qwen3Model(config)
    model.load_state_dict(torch.load(paths["model"], map_location=device))
    model.to(device)
    model.eval()

    # Create sampler
    if args.sampler == "greedy":
        sampler = GreedySampler()
    elif args.sampler == "temperature":
        sampler = TemperatureSampler(temperature=args.temperature)
    elif args.sampler == "top_p":
        sampler = TopPSampler(top_p=args.top_p, temperature=args.temperature)
    else:
        raise ValueError(f"Unknown sampler: {args.sampler}")

    # Create generator
    generator = TextGenerator(model, tokenizer, device, sampler=sampler)

    # Generate
    logger.info("Generating text...")
    if args.stream:
        print("\nGenerated text: ", end="", flush=True)
        for token in generator.generate_stream(
            args.prompt,
            max_new_tokens=args.max_tokens,
            use_cache=True,
            chat_wrapped=False,
        ):
            print(token, end="", flush=True)
        print("\n")
    else:
        result = generator.generate(
            args.prompt,
            max_new_tokens=args.max_tokens,
            use_cache=True,
            chat_wrapped=False,
        )
        print(f"\nGenerated text:\n{result}\n")


def evaluate_model(args):
    """Evaluate model on a dataset."""
    logger = get_logger()

    # Setup device
    device = setup_device(optimize_for_device=True)
    print_device_info(device)

    # Download and load model
    logger.info("Loading model...")
    paths = download_qwen3_model(
        model_type=args.model_type, size=args.model_size, out_dir=args.model_dir
    )

    # Load tokenizer
    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path=str(paths["tokenizer"]),
        apply_chat_template=True,
        add_generation_prompt=True,
        add_thinking=args.model_type == "reasoning",
    )

    # Load model
    config = ModelConfig.qwen3_0_6b()
    model = Qwen3Model(config)
    model.load_state_dict(torch.load(paths["model"], map_location=device))
    model.to(device)
    model.eval()

    # Create generator
    sampler = GreedySampler() if args.greedy else TopPSampler(top_p=0.95, temperature=0.7)
    generator = TextGenerator(model, tokenizer, device, sampler=sampler)

    # Create evaluator
    def generate_fn(prompt: str) -> str:
        return generator.generate(
            prompt, max_new_tokens=args.max_tokens, use_cache=True, chat_wrapped=False
        )

    evaluator = MathEvaluator(generate_fn=generate_fn)

    # Evaluate
    logger.info(f"Evaluating on dataset: {args.dataset}")
    results = evaluator.load_and_evaluate(
        args.dataset,
        problem_key=args.problem_key,
        answer_key=args.answer_key,
        verbose=True,
        max_problems=args.max_problems,
    )

    # Print summary
    evaluator.print_summary(results)

    # Save results if requested
    if args.output:
        evaluator.save_results(results, args.output)


def main():
    parser = argparse.ArgumentParser(
        description="Reasoning Model CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate text from a prompt")
    gen_parser.add_argument("prompt", type=str, help="Input prompt")
    gen_parser.add_argument(
        "--model-type",
        type=str,
        default="base",
        choices=["base", "reasoning"],
        help="Model type",
    )
    gen_parser.add_argument("--model-size", type=str, default="0.6B", help="Model size")
    gen_parser.add_argument(
        "--model-dir", type=str, default="./models", help="Model directory"
    )
    gen_parser.add_argument(
        "--max-tokens", type=int, default=100, help="Maximum tokens to generate"
    )
    gen_parser.add_argument(
        "--sampler",
        type=str,
        default="greedy",
        choices=["greedy", "temperature", "top_p"],
        help="Sampling strategy",
    )
    gen_parser.add_argument(
        "--temperature", type=float, default=0.7, help="Temperature for sampling"
    )
    gen_parser.add_argument("--top-p", type=float, default=0.9, help="Top-p value")
    gen_parser.add_argument(
        "--stream", action="store_true", help="Stream output token by token"
    )
    gen_parser.add_argument(
        "--chat-template", action="store_true", help="Use chat template"
    )

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate on a dataset")
    eval_parser.add_argument("dataset", type=str, help="Path to dataset JSON file")
    eval_parser.add_argument(
        "--model-type",
        type=str,
        default="reasoning",
        choices=["base", "reasoning"],
        help="Model type",
    )
    eval_parser.add_argument(
        "--model-size", type=str, default="0.6B", help="Model size"
    )
    eval_parser.add_argument(
        "--model-dir", type=str, default="./models", help="Model directory"
    )
    eval_parser.add_argument(
        "--max-tokens", type=int, default=512, help="Maximum tokens to generate"
    )
    eval_parser.add_argument(
        "--problem-key", type=str, default="problem", help="Key for problem text"
    )
    eval_parser.add_argument(
        "--answer-key", type=str, default="answer", help="Key for answer"
    )
    eval_parser.add_argument(
        "--max-problems", type=int, default=None, help="Maximum problems to evaluate"
    )
    eval_parser.add_argument(
        "--output", type=str, default=None, help="Output file for results"
    )
    eval_parser.add_argument(
        "--greedy", action="store_true", help="Use greedy decoding"
    )

    args = parser.parse_args()

    if args.command == "generate":
        generate_text(args)
    elif args.command == "evaluate":
        evaluate_model(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
