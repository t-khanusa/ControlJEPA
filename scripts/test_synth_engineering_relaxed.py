#!/usr/bin/env python3
"""Unit checks for SYNTH engineering-relaxed vs strict (no model / GPU)."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from synth_metrics import synth_engineering_relaxed_match, synth_strict_match


def main() -> None:
    cases = [
        ("abc", "abc", True, True),
        (" abc ", "abc", True, True),
        ("abc.", "abc", False, True),
        ("abc xtra", "abc", False, True),
        ("abcdef", "abc", False, False),
        ("xabc", "abc", False, False),
    ]
    for gen, gt, want_strict, want_rel in cases:
        s = synth_strict_match(gen, gt)
        r = synth_engineering_relaxed_match(gen, gt)
        assert s == want_strict, (gen, gt, s, want_strict)
        assert r == want_rel, (gen, gt, r, want_rel)
    print("ok: synth_engineering_relaxed_match + strict strip equality")


if __name__ == "__main__":
    main()
