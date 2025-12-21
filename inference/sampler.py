"""
Sampling strategies for text generation.

This module implements various sampling strategies for controlling the
randomness and diversity of generated text.
"""

from abc import ABC, abstractmethod
from collections import Counter
from typing import List, Callable, Any
import torch


class SamplingStrategy(ABC):
    """Abstract base class for sampling strategies.

    All sampling strategies should inherit from this class and implement
    the sample method.
    """

    @abstractmethod
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample next token from logits.

        Args:
            logits: Logits tensor of shape (batch, vocab_size)

        Returns:
            Sampled token indices
        """
        pass


class GreedySampler(SamplingStrategy):
    """Greedy sampling: always select the most likely token.

    This is deterministic and produces the same output every time.
    """

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Select the token with highest probability.

        Args:
            logits: Logits tensor

        Returns:
            Token with maximum logit value
        """
        return torch.argmax(logits, dim=-1, keepdim=True)


class TemperatureSampler(SamplingStrategy):
    """Temperature-based sampling for controlling randomness.

    Temperature controls the randomness of predictions:
    - temperature < 1: More confident, peaked distribution
    - temperature = 1: Unmodified distribution
    - temperature > 1: More random, flatter distribution

    Args:
        temperature: Temperature value (higher = more random)
    """

    def __init__(self, temperature: float = 1.0):
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        self.temperature = temperature

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample with temperature scaling.

        Args:
            logits: Logits tensor

        Returns:
            Sampled token indices
        """
        scaled_logits = logits / self.temperature
        probs = torch.softmax(scaled_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)


class TopPSampler(SamplingStrategy):
    """Top-p (nucleus) sampling.

    Samples from the smallest set of tokens whose cumulative probability
    exceeds p. This dynamically adjusts the number of tokens considered
    based on the confidence of the model.

    Args:
        top_p: Cumulative probability threshold (0 < top_p <= 1)
        temperature: Optional temperature scaling
    """

    def __init__(self, top_p: float = 0.9, temperature: float = 1.0):
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if temperature <= 0:
            raise ValueError("Temperature must be positive")

        self.top_p = top_p
        self.temperature = temperature

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample using nucleus sampling.

        Args:
            logits: Logits tensor

        Returns:
            Sampled token indices
        """
        # Apply temperature
        scaled_logits = logits / self.temperature
        probs = torch.softmax(scaled_logits, dim=-1)

        # Sort probabilities in descending order
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

        # Compute cumulative probabilities
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

        # Create mask for tokens to keep
        # Keep tokens until cumulative probability exceeds top_p
        mask = cumsum_probs <= self.top_p

        # Always keep at least one token
        mask[..., 0] = True

        # Zero out probabilities of tokens to remove
        sorted_probs = sorted_probs * mask

        # Renormalize
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

        # Sample from the filtered distribution
        sampled_sorted_indices = torch.multinomial(sorted_probs, num_samples=1)

        # Map back to original indices
        sampled_indices = torch.gather(sorted_indices, -1, sampled_sorted_indices)

        return sampled_indices


class SelfConsistencySampler:
    """Self-consistency sampling for improved reasoning.

    Generates multiple solutions using sampling and selects the most common
    answer. This helps improve accuracy on reasoning tasks by leveraging
    the wisdom of multiple generations.

    Args:
        num_samples: Number of samples to generate
        sampler: Base sampling strategy to use
        answer_extractor: Function to extract answer from generated text
    """

    def __init__(
        self,
        num_samples: int = 5,
        sampler: SamplingStrategy = None,
        answer_extractor: Callable[[str], Any] = None,
    ):
        if num_samples < 1:
            raise ValueError("num_samples must be at least 1")

        self.num_samples = num_samples
        self.sampler = sampler or TopPSampler(top_p=0.95, temperature=0.7)
        self.answer_extractor = answer_extractor or (lambda x: x)

    def generate_and_vote(
        self, generate_fn: Callable[[], str]
    ) -> tuple[str, List[str]]:
        """Generate multiple samples and vote on the most common answer.

        Args:
            generate_fn: Function that generates a single sample

        Returns:
            Tuple of (most common answer, all generated texts)
        """
        # Generate multiple samples
        generations = [generate_fn() for _ in range(self.num_samples)]

        # Extract answers
        answers = [self.answer_extractor(gen) for gen in generations]

        # Vote on most common answer
        if not answers:
            return None, generations

        counter = Counter(answers)
        most_common_answer = counter.most_common(1)[0][0]

        return most_common_answer, generations
