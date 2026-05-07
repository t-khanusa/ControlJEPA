"""SYNTH scoring helpers shared by evaluate scripts (torch-free).

- Strict EM used in galilai-group/llm-jepa: stripped equality.
- Engineering-relaxed: same as fork ``evaluate.py`` generic branch (prefix +
  non-alphanumeric boundary after gold).
"""


def synth_engineering_relaxed_match(generated: str, ground_truth_content: str) -> bool:
    """Credits correct short answers when trailing text starts after a delimiter (non-alphanumeric)."""
    gt = (ground_truth_content or "").strip()
    gen = (generated or "").strip()
    if gen == gt:
        return True
    if gen.startswith(gt):
        remainder = gen[len(gt):]
        if not remainder or not remainder[0].isalnum():
            return True
    return False


def synth_strict_match(generated: str, ground_truth_content: str) -> bool:
    """Paper/upstream SYNCH-style exact match after strip."""
    return (generated or "").strip() == (ground_truth_content or "").strip()
