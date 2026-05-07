from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import sys
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import GenerationConfig

# Reuse evaluate.py (same directory)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import evaluate as ev  # noqa: E402


# Basename must route eval() to synth strict equality (evaluate.py:_dataset_kind)
_SYNTH_ROUTING_NAME = "synth_stress_routing.jsonl"


def _corrupt_adjacent_swap(text: str, severity: float, rng: random.Random) -> str:
    """Swap adjacent non-space pairs; count ~ severity * min(len, 20) ops."""
    if len(text) < 2:
        return text
    s = list(text)
    n_ops = max(1, int(severity * min(len(s), 40)))
    for _ in range(n_ops):
        i = rng.randint(0, len(s) - 2)
        if s[i].isspace() and s[i + 1].isspace():
            continue
        s[i], s[i + 1] = s[i + 1], s[i]
    return "".join(s)


def _corrupt_char_drop(text: str, severity: float, rng: random.Random) -> str:
    """Delete random non-space characters."""
    idxs = [i for i, c in enumerate(text) if not c.isspace()]
    if not idxs:
        return text
    k = max(1, int(severity * len(idxs)))
    k = min(k, len(idxs))
    drop = set(rng.sample(idxs, k))
    return "".join(c for i, c in enumerate(text) if i not in drop)


def _corrupt_token_replace(
    text: str, severity: float, rng: random.Random
) -> str:
    """Replace a fraction of whitespace-separated tokens with a dummy token."""
    parts = re.split(r"(\s+)", text)
    # parts alternate word / whitespace if we split with capture... actually split keeps delims
    words = [p for p in parts if p and not p.isspace()]
    if not words:
        return text
    k = max(1, int(severity * len(words)))
    word_set = list(dict.fromkeys(words))
    dummy = rng.choice(["thing", "stuff", "item", "pattern"])
    out = text
    for w in rng.sample(words, min(k, len(words))):
        # replace whole word boundaries
        out = re.sub(r"\b" + re.escape(w) + r"\b", dummy, out, count=1)
    return out


def _corrupt_space_jitter(text: str, severity: float, rng: random.Random) -> str:
    """Insert/delete spaces to break tokenization slightly (mild stress)."""
    s = list(text)
    n = max(1, int(severity * max(1, len(s) // 8)))
    for _ in range(n):
        if rng.random() < 0.5 and " " in s:
            # delete one space
            sp = [i for i, c in enumerate(s) if c == " "]
            if sp:
                del s[rng.choice(sp)]
        else:
            i = rng.randint(0, len(s))
            s.insert(i, " ")
    return "".join(s)


CORRUPTORS: Dict[str, Callable[[str, float, random.Random], str]] = {
    "clean": lambda t, sev, rng: t,
    "adjacent_swap": _corrupt_adjacent_swap,
    "char_drop": _corrupt_char_drop,
    "token_replace": _corrupt_token_replace,
    "space_jitter": _corrupt_space_jitter,
}


def _apply_noise(
    example: Dict[str, Any],
    mode: str,
    severity: float,
    rng: random.Random,
) -> Dict[str, Any]:
    out = copy.deepcopy(example)
    msgs = out["messages"]
    if len(msgs) < 3:
        return out
    fn = CORRUPTORS.get(mode)
    if fn is None:
        raise ValueError(f"Unknown mode {mode}")
    u = msgs[1].get("content", "")
    msgs[1]["content"] = fn(u, severity, rng)
    return out


@dataclass
class StressResult:
    model_tag: str
    model_path: str
    noise_mode: str
    severity: float
    seed: int
    n_examples: int
    n_correct: int
    accuracy: float


def run_one_setting(
    model,
    tokenizer,
    generation_config: GenerationConfig,
    examples: List[Dict[str, Any]],
    original_model_name: str,
    noise_mode: str,
    severity: float,
    seed: int,
    max_new_tokens: int,
    unmask_assistant_special_tokens: bool,
    model_tag: str,
    model_path: str,
) -> StressResult:
    rng = random.Random(seed)
    n_ok = 0
    for ex in tqdm(
        examples,
        desc=f"{model_tag} {noise_mode} sev={severity} s={seed}",
        leave=False,
    ):
        ex2 = _apply_noise(ex, noise_mode, severity, rng)
        messages = ex2["messages"]
        full_messages = ev.get_messages(original_model_name, messages)
        prompt = ev.format_conversation(full_messages, tokenizer, plain=False)
        gen = ev.generate_response(
            model,
            tokenizer,
            prompt,
            generation_config,
            max_new_tokens,
            unmask_assistant_special_tokens=unmask_assistant_special_tokens,
        )
        if ev.eval(gen, messages, _SYNTH_ROUTING_NAME, "", startswith=False, debug=0):
            n_ok += 1
    n = len(examples)
    return StressResult(
        model_tag=model_tag,
        model_path=model_path,
        noise_mode=noise_mode,
        severity=severity,
        seed=seed,
        n_examples=n,
        n_correct=n_ok,
        accuracy=n_ok / max(n, 1),
    )


def load_examples(path: str, max_examples: Optional[int]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    if max_examples is not None:
        out = out[: max_examples]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Synth stress test: STP vs control_JEPA")
    p.add_argument(
        "--model_stp",
        type=str,
        default="ft-j-synth-2e-5-0.02-0-84",
        help="Checkpoint dir (STP / random_span)",
    )
    p.add_argument(
        "--model_control",
        type=str,
        default="ft-c-v_geo-synth-g0.95-t1e-4-2e-5-0.05-0-84",
        help="Checkpoint dir (control_JEPA v_geo)",
    )
    p.add_argument(
        "--original_model_name",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
    )
    p.add_argument(
        "--test_jsonl",
        type=str,
        default="datasets/synth_test.jsonl",
        help="Path to synth test JSONL",
    )
    p.add_argument("--max_examples", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=96)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--device_map", type=str, default="cuda:0")
    p.add_argument(
        "--modes",
        type=str,
        default="clean,adjacent_swap,char_drop,token_replace,space_jitter",
        help="Comma-separated noise modes",
    )
    p.add_argument(
        "--severities",
        type=str,
        default="0.0,0.05,0.1,0.2",
        help="Comma-separated severity values (ignored for clean)",
    )
    p.add_argument("--seeds", type=str, default="0,1,2", help="Comma-separated RNG seeds")
    p.add_argument("--out", type=str, default="stress_synth_results.json")
    p.add_argument(
        "--unmask_assistant_special_tokens",
        action="store_true",
        help="Match training if you used this flag in stp.py",
    )
    args = p.parse_args()

    test_path = args.test_jsonl
    if not os.path.isabs(test_path):
        test_path = os.path.join(_SCRIPT_DIR, test_path)
    if not os.path.isfile(test_path):
        raise FileNotFoundError(test_path)

    examples = load_examples(test_path, args.max_examples)
    if not examples:
        raise RuntimeError("No examples loaded")

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    severities = [float(x) for x in args.severities.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    generation_config = GenerationConfig(
        model_name="stress",
        do_sample=False,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
        repetition_penalty=1.1,
        num_beams=1,
    )

    all_results: List[Dict[str, Any]] = []

    def run_model(model_path: str, tag: str) -> None:
        mp = model_path
        if not os.path.isabs(mp):
            mp = os.path.join(_SCRIPT_DIR, mp)
        if not os.path.isdir(mp):
            raise FileNotFoundError(f"Missing checkpoint dir: {mp}")

        print(f"\n>>> Loading {tag}: {mp}")
        model, tokenizer = ev.load_model_and_tokenizer(
            mp,
            args.original_model_name,
            device_map=args.device_map,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        generation_config.pad_token_id = tokenizer.pad_token_id
        generation_config.eos_token_id = tokenizer.eos_token_id

        for mode in modes:
            sevs = [0.0] if mode == "clean" else severities
            for sev in sevs:
                if mode == "clean" and sev != 0.0:
                    continue
                for seed in seeds:
                    r = run_one_setting(
                        model,
                        tokenizer,
                        generation_config,
                        examples,
                        args.original_model_name,
                        mode,
                        sev,
                        seed,
                        args.max_new_tokens,
                        args.unmask_assistant_special_tokens,
                        tag,
                        mp,
                    )
                    all_results.append(asdict(r))
                    print(
                        f"  {tag}  {mode:16s}  sev={sev:.3f}  seed={seed}  "
                        f"acc={r.accuracy:.4f}  ({r.n_correct}/{r.n_examples})"
                    )

        del model
        torch.cuda.empty_cache()

    stp_path = args.model_stp
    if not os.path.isabs(stp_path):
        stp_path = os.path.join(_SCRIPT_DIR, stp_path)
    ctl_path = args.model_control
    if not os.path.isabs(ctl_path):
        ctl_path = os.path.join(_SCRIPT_DIR, ctl_path)

    run_model(stp_path, "STP")
    run_model(ctl_path, "control_v_geo")

    out_path = args.out
    if not os.path.isabs(out_path):
        out_path = os.path.join(_SCRIPT_DIR, out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "test_jsonl": test_path,
                "n_examples": len(examples),
                "original_model_name": args.original_model_name,
                "modes": modes,
                "severities": severities,
                "seeds": seeds,
                "results": all_results,
            },
            f,
            indent=2,
        )
    print(f"\nWrote {out_path}")

    # Aggregate table: mean acc over seeds
    print("\n=== Mean accuracy over seeds (robustness summary) ===\n")
    key_fn = lambda d: (d["model_tag"], d["noise_mode"], round(d["severity"], 6))
    buckets: Dict[Tuple[str, str, float], List[float]] = {}
    for row in all_results:
        k = key_fn(row)
        buckets.setdefault(k, []).append(row["accuracy"])
    for tag in ["STP", "control_v_geo"]:
        print(f"\n--- {tag} ---")
        for mode in modes:
            sevs = [0.0] if mode == "clean" else sorted(set(severities))
            for sev in sevs:
                if mode == "clean" and sev != 0.0:
                    continue
                accs = buckets.get((tag, mode, sev), [])
                if not accs:
                    continue
                m = float(np.mean(accs))
                s = float(np.std(accs)) if len(accs) > 1 else 0.0
                print(f"  {mode:16s}  sev={sev:.3f}  mean_acc={m:.4f}  std={s:.4f}  (n_seeds={len(accs)})")


if __name__ == "__main__":
    main()
