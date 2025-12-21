"""
Text generation utilities for Qwen3 models.

This module provides the TextGenerator class for generating text from
Qwen3 models with various sampling strategies and generation modes.
"""

import torch
from typing import Optional, Iterator, Union
from core.model import Qwen3Model, KVCache
from core.tokenizer import Qwen3Tokenizer
from inference.sampler import SamplingStrategy, GreedySampler


class TextGenerator:
    """Text generator for Qwen3 models.

    Handles text generation with support for different sampling strategies,
    KV caching, and streaming output.

    Args:
        model: Qwen3 model instance
        tokenizer: Qwen3 tokenizer instance
        device: Device to run generation on
        sampler: Sampling strategy (default: greedy)
    """

    def __init__(
        self,
        model: Qwen3Model,
        tokenizer: Qwen3Tokenizer,
        device: torch.device,
        sampler: Optional[SamplingStrategy] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.sampler = sampler or GreedySampler()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        use_cache: bool = True,
        stop_at_eos: bool = True,
        chat_wrapped: Optional[bool] = None,
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            use_cache: Whether to use KV caching for efficiency
            stop_at_eos: Whether to stop at end-of-sequence token
            chat_wrapped: Whether to wrap prompt in chat template

        Returns:
            Generated text
        """
        # Encode prompt
        token_ids = self.tokenizer.encode(prompt, chat_wrapped=chat_wrapped)
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        # Initialize cache if needed
        cache = None
        if use_cache:
            cache = KVCache(self.model.config.n_layers)
            self.model.reset_kv_cache()

        # Generate tokens
        generated_ids = []
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                logits = self.model(input_ids, cache=cache)

                # Sample next token
                next_token = self.sampler.sample(logits[:, -1, :])

                # Check for EOS
                if stop_at_eos and next_token.item() == self.tokenizer.eos_token_id:
                    break

                generated_ids.append(next_token.item())

                # Prepare input for next iteration
                if use_cache:
                    input_ids = next_token
                else:
                    input_ids = torch.cat([input_ids, next_token], dim=1)

        # Decode generated tokens
        generated_text = self.tokenizer.decode(generated_ids)
        return generated_text

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        use_cache: bool = True,
        stop_at_eos: bool = True,
        chat_wrapped: Optional[bool] = None,
    ) -> Iterator[str]:
        """Generate text with streaming output.

        Yields each generated token as it's produced, allowing for
        real-time display of generation progress.

        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            use_cache: Whether to use KV caching for efficiency
            stop_at_eos: Whether to stop at end-of-sequence token
            chat_wrapped: Whether to wrap prompt in chat template

        Yields:
            Generated tokens one at a time
        """
        # Encode prompt
        token_ids = self.tokenizer.encode(prompt, chat_wrapped=chat_wrapped)
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        # Initialize cache if needed
        cache = None
        if use_cache:
            cache = KVCache(self.model.config.n_layers)
            self.model.reset_kv_cache()

        # Generate tokens
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                logits = self.model(input_ids, cache=cache)

                # Sample next token
                next_token = self.sampler.sample(logits[:, -1, :])

                # Check for EOS
                if stop_at_eos and next_token.item() == self.tokenizer.eos_token_id:
                    break

                # Decode and yield token
                token_text = self.tokenizer.decode([next_token.item()])
                yield token_text

                # Prepare input for next iteration
                if use_cache:
                    input_ids = next_token
                else:
                    input_ids = torch.cat([input_ids, next_token], dim=1)

    def batch_generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 50,
        use_cache: bool = False,
        stop_at_eos: bool = True,
        chat_wrapped: Optional[bool] = None,
    ) -> list[str]:
        """Generate text for multiple prompts in batch.

        Note: Batch generation does not support KV caching.

        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum number of tokens to generate
            use_cache: Not supported for batch generation
            stop_at_eos: Whether to stop at end-of-sequence token
            chat_wrapped: Whether to wrap prompts in chat template

        Returns:
            List of generated texts
        """
        if use_cache:
            raise ValueError("KV caching is not supported for batch generation")

        # Encode all prompts
        all_token_ids = [
            self.tokenizer.encode(prompt, chat_wrapped=chat_wrapped)
            for prompt in prompts
        ]

        # Pad sequences to same length
        max_len = max(len(ids) for ids in all_token_ids)
        padded_ids = []
        attention_masks = []

        for ids in all_token_ids:
            padding_len = max_len - len(ids)
            padded = [self.tokenizer.pad_token_id] * padding_len + ids
            mask = [0] * padding_len + [1] * len(ids)
            padded_ids.append(padded)
            attention_masks.append(mask)

        input_ids = torch.tensor(padded_ids, dtype=torch.long, device=self.device)

        # Generate tokens
        generated_sequences = []
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.model(input_ids)

                # Sample next token for each sequence
                next_tokens = self.sampler.sample(logits[:, -1, :])

                # Append to sequences
                input_ids = torch.cat([input_ids, next_tokens], dim=1)

        # Decode each sequence
        results = []
        for seq in input_ids:
            # Remove padding and prompt tokens
            seq_len = len(all_token_ids[len(results)])
            generated = seq[max_len:].tolist()

            # Find EOS if present
            if stop_at_eos and self.tokenizer.eos_token_id in generated:
                eos_idx = generated.index(self.tokenizer.eos_token_id)
                generated = generated[:eos_idx]

            text = self.tokenizer.decode(generated)
            results.append(text)

        return results

    def set_sampler(self, sampler: SamplingStrategy):
        """Change the sampling strategy.

        Args:
            sampler: New sampling strategy to use
        """
        self.sampler = sampler
