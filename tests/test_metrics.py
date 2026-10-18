"""Scoring rules of evaluate.py (run with `pytest tests/`)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pytest.importorskip("sklearn")
pytest.importorskip("transformers")
import evaluate as ev  # noqa: E402


def conv(answer, user="question"):
    return [{"role": "system", "content": "s"}, {"role": "user", "content": user},
            {"role": "assistant", "content": answer}]


def score(generated, answer, dataset, **kw):
    return ev.eval(generated, conv(answer, **kw), f"datasets/{dataset}_test.jsonl", spider_path="")


def test_routing_uses_basename():
    assert ev.dataset_kind("datasets/gsm8k_test.jsonl") == "gsm8k"
    assert ev.dataset_kind("/abs/path/nq_open_test.jsonl") == "nq_open"
    assert ev.dataset_kind("my_data.jsonl") == "generic"


def test_gsm8k_final_answer():
    assert score("so 3 + 4 = 7\n#### 1,234.", "...\n#### 1234", "gsm8k")
    assert score("#### 5 then #### 7", "#### 7", "gsm8k")
    assert not score("the answer is 7", "#### 7", "gsm8k")


def test_nq_open_normalised_containment():
    assert score("It was The Beatles!", "Beatles; the beatles", "nq_open")
    assert not score("Rolling Stones", "Beatles", "nq_open")


def test_hellaswag_and_exact_match():
    assert score(" B) because", "B", "hellaswag")
    assert score("a.*b \n", "a.*b", "turk")
    assert score("a.*b", "a.*b", "synth")
    assert not score("a.*b ", "a.*b", "synth")
