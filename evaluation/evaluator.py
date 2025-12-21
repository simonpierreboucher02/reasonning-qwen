"""
Model evaluator for mathematical reasoning tasks.

This module provides the MathEvaluator class for evaluating models on
datasets of mathematical problems.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass

from .math_verifier import extract_final_answer, grade_answer


@dataclass
class EvaluationResult:
    """Result of evaluating a single problem.

    Attributes:
        problem_id: Unique identifier for the problem
        problem: Problem text
        ground_truth: Correct answer
        generated_text: Full generated text from model
        extracted_answer: Extracted answer from generated text
        is_correct: Whether the answer is correct
        generation_time: Time taken to generate (seconds)
    """

    problem_id: str
    problem: str
    ground_truth: str
    generated_text: str
    extracted_answer: str
    is_correct: bool
    generation_time: float


class MathEvaluator:
    """Evaluator for mathematical reasoning tasks.

    Args:
        generate_fn: Function that takes a prompt and returns generated text
        answer_extractor: Function to extract answer from generated text
        prompt_template: Template for formatting problems as prompts
    """

    def __init__(
        self,
        generate_fn: Callable[[str], str],
        answer_extractor: Optional[Callable[[str], str]] = None,
        prompt_template: Optional[str] = None,
    ):
        self.generate_fn = generate_fn
        self.answer_extractor = answer_extractor or extract_final_answer

        # Default prompt template
        if prompt_template is None:
            self.prompt_template = (
                "Solve the following math problem. "
                "Put your final answer in \\boxed{{}}.\n\n"
                "Problem: {problem}"
            )
        else:
            self.prompt_template = prompt_template

    def format_prompt(self, problem: str) -> str:
        """Format a problem using the prompt template.

        Args:
            problem: Problem text

        Returns:
            Formatted prompt
        """
        return self.prompt_template.format(problem=problem)

    def evaluate_single(
        self, problem: str, ground_truth: str, problem_id: str = ""
    ) -> EvaluationResult:
        """Evaluate a single problem.

        Args:
            problem: Problem text
            ground_truth: Correct answer
            problem_id: Optional identifier for the problem

        Returns:
            EvaluationResult with all evaluation details
        """
        # Format prompt and generate
        prompt = self.format_prompt(problem)
        start_time = time.time()
        generated_text = self.generate_fn(prompt)
        generation_time = time.time() - start_time

        # Extract and grade answer
        extracted_answer = self.answer_extractor(generated_text)
        is_correct = grade_answer(extracted_answer, ground_truth)

        return EvaluationResult(
            problem_id=problem_id,
            problem=problem,
            ground_truth=ground_truth,
            generated_text=generated_text,
            extracted_answer=extracted_answer,
            is_correct=is_correct,
            generation_time=generation_time,
        )

    def evaluate_dataset(
        self,
        dataset: List[Dict],
        problem_key: str = "problem",
        answer_key: str = "answer",
        id_key: str = "id",
        verbose: bool = True,
        max_problems: Optional[int] = None,
    ) -> Dict:
        """Evaluate on a dataset of problems.

        Args:
            dataset: List of problem dictionaries
            problem_key: Key for problem text in dictionaries
            answer_key: Key for answer in dictionaries
            id_key: Key for problem ID in dictionaries
            verbose: Whether to print progress
            max_problems: Maximum number of problems to evaluate (None for all)

        Returns:
            Dictionary with evaluation results and statistics
        """
        if max_problems is not None:
            dataset = dataset[:max_problems]

        results = []
        correct_count = 0
        total_time = 0

        if verbose:
            print(f"Evaluating {len(dataset)} problems...")

        for i, item in enumerate(dataset):
            problem = item[problem_key]
            ground_truth = item[answer_key]
            problem_id = item.get(id_key, str(i))

            # Evaluate
            result = self.evaluate_single(problem, ground_truth, problem_id)
            results.append(result)

            if result.is_correct:
                correct_count += 1

            total_time += result.generation_time

            # Print progress
            if verbose:
                status = "✓" if result.is_correct else "✗"
                accuracy = correct_count / (i + 1) * 100
                avg_time = total_time / (i + 1)
                eta = avg_time * (len(dataset) - i - 1)

                print(
                    f"[{i+1}/{len(dataset)}] {status} "
                    f"Acc: {accuracy:.1f}% "
                    f"Time: {result.generation_time:.1f}s "
                    f"ETA: {eta/60:.1f}min"
                )

        # Compute statistics
        accuracy = correct_count / len(results) * 100 if results else 0
        avg_time = total_time / len(results) if results else 0

        return {
            "results": results,
            "total_problems": len(results),
            "correct": correct_count,
            "accuracy": accuracy,
            "total_time": total_time,
            "avg_time_per_problem": avg_time,
        }

    def load_and_evaluate(
        self,
        dataset_path: str,
        problem_key: str = "problem",
        answer_key: str = "answer",
        id_key: str = "id",
        verbose: bool = True,
        max_problems: Optional[int] = None,
    ) -> Dict:
        """Load a JSON dataset and evaluate on it.

        Args:
            dataset_path: Path to JSON dataset file
            problem_key: Key for problem text in dictionaries
            answer_key: Key for answer in dictionaries
            id_key: Key for problem ID in dictionaries
            verbose: Whether to print progress
            max_problems: Maximum number of problems to evaluate (None for all)

        Returns:
            Dictionary with evaluation results and statistics
        """
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        with open(dataset_path, "r") as f:
            dataset = json.load(f)

        if verbose:
            print(f"Loaded dataset from {dataset_path}")

        return self.evaluate_dataset(
            dataset,
            problem_key=problem_key,
            answer_key=answer_key,
            id_key=id_key,
            verbose=verbose,
            max_problems=max_problems,
        )

    def save_results(self, results: Dict, output_path: str):
        """Save evaluation results to a JSON file.

        Args:
            results: Results dictionary from evaluate_dataset
            output_path: Path to save results
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert results to serializable format
        serializable_results = {
            "total_problems": results["total_problems"],
            "correct": results["correct"],
            "accuracy": results["accuracy"],
            "total_time": results["total_time"],
            "avg_time_per_problem": results["avg_time_per_problem"],
            "results": [
                {
                    "problem_id": r.problem_id,
                    "problem": r.problem,
                    "ground_truth": r.ground_truth,
                    "extracted_answer": r.extracted_answer,
                    "is_correct": r.is_correct,
                    "generation_time": r.generation_time,
                }
                for r in results["results"]
            ],
        }

        with open(output_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        print(f"Results saved to {output_path}")

    def print_summary(self, results: Dict):
        """Print a summary of evaluation results.

        Args:
            results: Results dictionary from evaluate_dataset
        """
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Total Problems:     {results['total_problems']}")
        print(f"Correct:            {results['correct']}")
        print(f"Accuracy:           {results['accuracy']:.2f}%")
        print(f"Total Time:         {results['total_time']:.1f}s")
        print(f"Avg Time/Problem:   {results['avg_time_per_problem']:.2f}s")
        print("=" * 60 + "\n")
