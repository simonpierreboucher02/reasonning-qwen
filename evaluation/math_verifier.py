"""
Mathematical answer verification utilities.

This module provides utilities for extracting, normalizing, and verifying
mathematical answers from model-generated text.
"""

import re
from typing import Optional, List
from sympy import simplify
from sympy.parsing import sympy_parser as spp
from sympy.core.sympify import SympifyError
from tokenize import TokenError


# Regular expressions
RE_NUMBER = re.compile(r"-?(?:\d+/\d+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
RE_SPECIAL = re.compile(r"<\|[^>]+?\|>")  # Strip chat special tokens

# LaTeX fixes for normalization
LATEX_FIXES = [
    (r"\\left\s*", ""),
    (r"\\right\s*", ""),
    (r"\\,|\\!|\\;|\\:", ""),
    (r"\\cdot", "*"),
    (r"\u00B7|\u00D7", "*"),
    (r"\\\^\\circ", ""),
    (r"\\dfrac", r"\\frac"),
    (r"\\tfrac", r"\\frac"),
    (r"°", ""),
]


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract the last boxed answer from LaTeX text.

    Searches for the last occurrence of \\boxed{...} and extracts the content,
    handling nested braces correctly.

    Args:
        text: Text containing LaTeX boxed answer

    Returns:
        Content inside the last \\boxed{...} or None if not found

    Example:
        >>> extract_boxed_answer("The answer is \\boxed{42}")
        '42'
        >>> extract_boxed_answer("\\boxed{\\frac{1}{2}}")
        '\\frac{1}{2}'
    """
    # Find the last occurrence of "\\boxed"
    boxed_start_idx = text.rfind(r"\boxed")
    if boxed_start_idx == -1:
        return None

    # Get position after "\\boxed"
    current_idx = boxed_start_idx + len(r"\boxed")

    # Skip any whitespace after "\\boxed"
    while current_idx < len(text) and text[current_idx].isspace():
        current_idx += 1

    # Expect an opening brace "{"
    if current_idx >= len(text) or text[current_idx] != "{":
        return None

    # Parse the braces with nesting
    current_idx += 1
    brace_depth = 1
    content_start_idx = current_idx

    while current_idx < len(text) and brace_depth > 0:
        char = text[current_idx]
        if char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1
        current_idx += 1

    # Account for unbalanced braces
    if brace_depth != 0:
        return None

    # Extract content inside the outermost braces
    return text[content_start_idx : current_idx - 1]


def extract_final_answer(text: str, fallback: str = "number_then_full") -> str:
    """Extract the final answer from generated text.

    First attempts to find a boxed answer. If not found, falls back to
    extracting the last number or returning the full text.

    Args:
        text: Generated text
        fallback: Fallback strategy ("number_then_full", "number_only", or None)

    Returns:
        Extracted answer string

    Example:
        >>> extract_final_answer("The answer is \\boxed{42}")
        '42'
        >>> extract_final_answer("The result is 3.14", fallback="number_then_full")
        '3.14'
    """
    result = ""

    if text:
        # Prefer the last boxed expression if present
        boxed = extract_boxed_answer(text.strip())
        if boxed:
            result = boxed.strip().strip("$ ")

        # If no boxed expression, try fallback
        elif fallback in ("number_then_full", "number_only"):
            m = RE_NUMBER.findall(text)
            if m:
                # Use last number
                result = m[-1]
            elif fallback == "number_then_full":
                # Else return full text if no number found
                result = text

    return result


def normalize_latex(text: str) -> str:
    """Normalize LaTeX mathematical expressions.

    Converts LaTeX expressions to a canonical form suitable for symbolic
    comparison. Handles fractions, exponents, roots, and other common LaTeX.

    Args:
        text: LaTeX mathematical expression

    Returns:
        Normalized expression string

    Example:
        >>> normalize_latex("\\frac{1}{2}")
        '(1)/(2)'
        >>> normalize_latex("x^2")
        'x**2'
    """
    if not text:
        return ""

    # Remove special tokens
    text = RE_SPECIAL.sub("", text).strip()

    # Remove angle-degree markers
    text = re.sub(r"\^\s*\{\s*\\circ\s*\}", "", text)  # ^{\\circ}
    text = re.sub(r"\^\s*\\circ", "", text)  # ^\\circ
    text = text.replace("°", "")  # Unicode degree

    # Unwrap \\text{...} if the whole string is wrapped
    match = re.match(r"^\\text\{(?P<x>.+?)\}$", text)
    if match:
        text = match.group("x")

    # Strip inline/display math wrappers \\( \\) \\[ \\]
    text = re.sub(r"\\\(|\\\)|\\\[|\\\]", "", text)

    # Light LaTeX canonicalization
    for pat, rep in LATEX_FIXES:
        text = re.sub(pat, rep, text)

    # Numbers/roots
    text = text.replace("\\%", "%").replace("$", "").replace("%", "")
    text = re.sub(
        r"\\sqrt\s*\{([^}]*)\}", lambda match: f"sqrt({match.group(1)})", text
    )
    text = re.sub(r"\\sqrt\s+([^\\\s{}]+)", lambda match: f"sqrt({match.group(1)})", text)

    # Fractions
    text = re.sub(
        r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}",
        lambda match: f"({match.group(1)})/({match.group(2)})",
        text,
    )
    text = re.sub(
        r"\\frac\s+([^\s{}]+)\s+([^\s{}]+)",
        lambda match: f"({match.group(1)})/({match.group(2)})",
        text,
    )

    # Exponent and mixed numbers
    text = text.replace("^", "**")
    text = re.sub(r"(?<=\d)\s+(\d+/\d+)", lambda match: "+" + match.group(1), text)

    # 1,234 -> 1234
    text = re.sub(r"(?<=\d),(?=\d\d\d(\D|$))", "", text)

    return text.replace("{", "").replace("}", "").strip().lower()


def parse_with_sympy(expr: str):
    """Parse a mathematical expression using SymPy.

    Args:
        expr: Mathematical expression string

    Returns:
        SymPy expression object or None if parsing fails

    Example:
        >>> parse_with_sympy("2*x + 3")
        2*x + 3
        >>> parse_with_sympy("invalid@@") is None
        True
    """
    try:
        return spp.parse_expr(
            expr,
            transformations=(
                # Standard transformations like handling parentheses
                *spp.standard_transformations,
                # Allow omitted multiplication symbols (e.g., "2x" -> 2*x")
                spp.implicit_multiplication_application,
            ),
            # Evaluate during parsing so simple constants simplify (e.g., 2+3 -> 5)
            evaluate=True,
        )
    except (SympifyError, SyntaxError, TypeError, IndexError, TokenError):
        return None


def check_mathematical_equality(expr1: str, expr2: str) -> bool:
    """Check if two mathematical expressions are equivalent.

    Uses symbolic mathematics to determine if expressions are equivalent,
    not just string matching.

    Args:
        expr1: First mathematical expression
        expr2: Second mathematical expression

    Returns:
        True if expressions are mathematically equivalent

    Example:
        >>> check_mathematical_equality("1/2", "0.5")
        True
        >>> check_mathematical_equality("x+1", "1+x")
        True
        >>> check_mathematical_equality("2", "3")
        False
    """
    # First, check if the two expressions are exactly the same string
    if expr1 == expr2:
        return True

    # Parse both expressions into SymPy objects
    sympy1 = parse_with_sympy(expr1)
    sympy2 = parse_with_sympy(expr2)

    # If both expressions were parsed successfully, try symbolic comparison
    if sympy1 is not None and sympy2 is not None:
        try:
            # If the difference is 0, they are equivalent
            return simplify(sympy1 - sympy2) == 0
        except (SympifyError, TypeError):
            pass

    return False


def split_into_parts(text: str) -> List[str]:
    """Split text into parts if it represents a tuple or list.

    Args:
        text: Text that might be a tuple like "(a, b)" or list like "[a, b]"

    Returns:
        List of parts, or single-element list if not a tuple/list

    Example:
        >>> split_into_parts("(1, 2, 3)")
        ['1', '2', '3']
        >>> split_into_parts("42")
        ['42']
    """
    result = [text]

    if text:
        # Check if text looks like a tuple or list, e.g. "(a, b)" or "[a, b]"
        if (
            len(text) >= 2
            and text[0] in "(["
            and text[-1] in ")]"
            and "," in text[1:-1]
        ):
            # Split on commas inside brackets and strip whitespace
            items = [p.strip() for p in text[1:-1].split(",")]
            if all(items):
                result = items
    else:
        # If text is empty, return an empty list
        result = []

    return result


def grade_answer(predicted: str, ground_truth: str) -> bool:
    """Grade a predicted answer against the ground truth.

    Handles tuples, lists, and complex mathematical expressions with
    symbolic verification.

    Args:
        predicted: Predicted answer from model
        ground_truth: Correct answer

    Returns:
        True if answers are equivalent

    Example:
        >>> grade_answer("1/2", "0.5")
        True
        >>> grade_answer("(1, 2)", "(1, 2)")
        True
        >>> grade_answer("\\frac{1}{2}", "0.5")
        True
    """
    result = False

    # Only continue if both inputs are non-empty strings
    if predicted is not None and ground_truth is not None:
        # Normalize and split into parts
        gt_parts = split_into_parts(normalize_latex(ground_truth))
        pred_parts = split_into_parts(normalize_latex(predicted))

        # Ensure both sides have same number of valid parts
        if gt_parts and pred_parts and len(gt_parts) == len(pred_parts):
            result = all(
                check_mathematical_equality(gt, pred)
                for gt, pred in zip(gt_parts, pred_parts)
            )

    return result
