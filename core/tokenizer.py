"""
Tokenizer for Qwen3 models.

This module provides a tokenizer wrapper that handles special tokens and
chat templates for Qwen3 models.
"""

import re
from pathlib import Path
from typing import List, Optional


class Qwen3Tokenizer:
    """Tokenizer for Qwen3 models with chat template support.

    Handles special tokens and provides chat formatting capabilities for
    both base and reasoning models.

    Args:
        tokenizer_file_path: Path to the tokenizer JSON file
        apply_chat_template: Whether to automatically apply chat template
        add_generation_prompt: Whether to add generation prompt
        add_thinking: Whether to add thinking tags for reasoning models
    """

    _SPECIALS = [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|object_ref_start|>",
        "<|object_ref_end|>",
        "<|box_start|>",
        "<|box_end|>",
        "<|quad_start|>",
        "<|quad_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|vision_pad|>",
        "<|image_pad|>",
        "<|video_pad|>",
    ]
    _SPLIT_RE = re.compile(r"(<\|[^>]+?\|>)")

    def __init__(
        self,
        tokenizer_file_path: str = "tokenizer-base.json",
        apply_chat_template: bool = False,
        add_generation_prompt: bool = False,
        add_thinking: bool = False,
    ):
        from tokenizers import Tokenizer

        self.apply_chat_template = apply_chat_template
        self.add_generation_prompt = add_generation_prompt
        self.add_thinking = add_thinking

        tok_path = Path(tokenizer_file_path)
        if not tok_path.is_file():
            raise FileNotFoundError(
                f"Tokenizer file '{tok_path}' not found. Please ensure it's available."
            )

        self._tok = Tokenizer.from_file(str(tok_path))
        self._special_to_id = {t: self._tok.token_to_id(t) for t in self._SPECIALS}

        self.pad_token = "<|endoftext|>"
        self.pad_token_id = self._special_to_id.get(self.pad_token)

        # Match HF behavior: chat model → <|im_end|>, base model → <|endoftext|>
        fname = tok_path.name.lower()
        if "base" in fname and "reasoning" not in fname:
            self.eos_token = "<|endoftext|>"
        else:
            self.eos_token = "<|im_end|>"
        self.eos_token_id = self._special_to_id.get(self.eos_token)

    def encode(
        self, prompt: str, chat_wrapped: Optional[bool] = None
    ) -> List[int]:
        """Encode text to token IDs.

        Args:
            prompt: Text to encode
            chat_wrapped: Override chat template setting (None uses default)

        Returns:
            List of token IDs
        """
        if chat_wrapped is None:
            chat_wrapped = self.apply_chat_template

        stripped = prompt.strip()
        if stripped in self._special_to_id and "\n" not in stripped:
            return [self._special_to_id[stripped]]

        if chat_wrapped:
            prompt = self._wrap_chat(prompt)

        ids = []
        for part in filter(None, self._SPLIT_RE.split(prompt)):
            if part in self._special_to_id:
                ids.append(self._special_to_id[part])
            else:
                ids.extend(self._tok.encode(part).ids)
        return ids

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text.

        Args:
            token_ids: List of token IDs to decode

        Returns:
            Decoded text
        """
        return self._tok.decode(token_ids, skip_special_tokens=False)

    def _wrap_chat(self, user_msg: str) -> str:
        """Wrap user message in chat template.

        Args:
            user_msg: User message to wrap

        Returns:
            Formatted chat message
        """
        s = f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        if self.add_generation_prompt:
            s += "<|im_start|>assistant"
            if self.add_thinking:
                s += "\n"  # insert no <think> tag, just a new line
            else:
                s += "\n<think>\n\n</think>\n\n"
        return s
