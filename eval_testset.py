"""
Evaluation metrics:
  - Task accuracy (exact match on NL-RX-style synth by default; extensible).
  - Teacher-forced next-token accuracy on supervised spans (complements PPL).
  - Tube geometry SNR proxy: mean ||parallel||^2 / mean ||perpendicular||^2.

Data-efficiency: use compare_three_method --data_fraction with a seeded subsample of train JSONL.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling


def materialize_train_fraction(
    train_jsonl: Path,
    out_path: Path,
    fraction: float,
    seed: int = 42,
) -> Path:
    """Shuffle (seeded) then keep first ceil(fraction * N) lines — STP-style data-efficiency curve."""
    train_jsonl = Path(train_jsonl)
    lines = train_jsonl.read_text().splitlines()
    lines = [ln for ln in lines if ln.strip()]
    if fraction >= 1.0 or not lines:
        return train_jsonl
    rng = random.Random(seed)
    order = list(range(len(lines)))
    rng.shuffle(order)
    k = max(1, int(math.ceil(len(lines) * fraction)))
    k = min(k, len(lines))
    chosen = [lines[i] for i in order[:k]]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(chosen) + "\n")
    return out_path


def nlrx_star_suffix_bin(gold_regex: str) -> str:
    """Peels trailing literal ``.*`` repeats from the NL-RX string form.

    Mirrors the imbalance discussed in Semantic Tube diversity ablations:

    - ``suffix_one_star``: exactly one trailing ``.*`` fragment.
    - ``suffix_multi_star``: two or more trailing ``.*`` (e.g. ``.*.*``).
    - ``other``: anything else.

    Parsing is lexical on the decoded assistant string rather than POSIX
    regex semantics; it matches how the imbalance is mined from this JSONL corpus.
    """
    t = gold_regex.strip()
    n = 0
    while t.endswith(".*"):
        n += 1
        t = t[:-2]
    if n >= 2:
        return "suffix_multi_star"
    if n == 1:
        return "suffix_one_star"
    return "other"


def prediction_matches_gold(generated: str, messages: List[dict], eval_stem: str) -> bool:
    """Match rules roughly consistent with evaluate.py eval()."""
    g = generated.strip()
    gold = messages[2]["content"].strip()
    stem = eval_stem.lower()
    if stem.startswith("gsm8k"):
        import re

        pat = re.compile(r"\n#### (.+)$")
        gm = pat.search(gold)
        gx = pat.search(g)
        if not gm or not gx:
            return False
        return gm.group(1) == gx.group(1)
    if stem.startswith("spider"):
        return False  # needs DB execution; use evaluate.py
    if stem.startswith("nq_open"):
        parts = g.split("; ")
        return any(p in gold for p in parts)
    return g == gold


@torch.no_grad()
def compute_nlrx_star_suffix_exact_match(
    checkpoint_dir: Path,
    eval_jsonl: Path,
    base_model_name: str,
    max_new_tokens: int = 256,
    max_length: int = 512,
    max_examples: Optional[int] = None,
    eval_profile: str = "fork",
) -> Dict[str, Dict[str, float]]:
    """Per-suffix greedy exact-match (Table 1 style: trailing ``.*`` vs ``.*.*``)."""

    import evaluate as eval_mod

    if eval_profile == "llm_jepa_official":
        max_new_tokens = eval_mod.LLM_JEPA_OFFICIAL_MAX_NEW_TOKENS
        max_length = eval_mod.LLM_JEPA_OFFICIAL_MAX_LENGTH
    elif eval_profile != "fork":
        raise ValueError(
            "eval_profile must be 'fork' or 'llm_jepa_official'; "
            f"got {eval_profile!r}"
        )

    checkpoint_dir = Path(checkpoint_dir)
    eval_stem = Path(eval_jsonl).stem
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint_dir),
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()
    gen_cfg = SimpleNamespace(max_length=max_length)

    lines = Path(eval_jsonl).read_text().splitlines()
    lines = [ln for ln in lines if ln.strip()]
    if max_examples is not None:
        lines = lines[:max_examples]

    buckets = ("suffix_one_star", "suffix_multi_star", "other")
    stats = {k: {"ok": 0, "n": 0} for k in buckets}

    for line in tqdm(lines, desc=f"nlrx_star_suffix[{eval_stem}]"):
        ex = json.loads(line)
        messages = ex["messages"]
        full = eval_mod.get_messages(base_model_name, messages)
        prompt = eval_mod.format_conversation(full, tokenizer, include_assistant=False, plain=False)
        gen = eval_mod.generate_response(model, tokenizer, prompt, gen_cfg, max_new_tokens)
        gold = messages[2]["content"]
        bn = nlrx_star_suffix_bin(gold)
        stats[bn]["n"] += 1
        if prediction_matches_gold(gen, messages, eval_stem):
            stats[bn]["ok"] += 1

    out: Dict[str, Dict[str, float]] = {}
    total_ok = sum(v["ok"] for v in stats.values())
    total_n = sum(v["n"] for v in stats.values())
    out["overall"] = {
        "accuracy": total_ok / total_n if total_n else float("nan"),
        "count": float(total_n),
    }
    for k in buckets:
        n_k = stats[k]["n"]
        out[k] = {
            "accuracy": stats[k]["ok"] / n_k if n_k else float("nan"),
            "count": float(n_k),
        }
    return out


def _tube_snr_energy_sums(
    h: torch.Tensor,
    user_bounds: List[Tuple[int, int]],
    assistant_bounds: List[Tuple[int, int]],
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (sum_parallel_energy, sum_perpendicular_energy, n_positions) for one forward batch.
    n_positions is the count of masked positions used in the means (same as LyapunovTubeLoss geometry).
    """
    B, T, D = h.shape
    device = h.device
    dtype = h.dtype
    d_all = torch.zeros_like(h)
    v_geodesic = torch.zeros(B, D, device=device, dtype=dtype)
    pos_mask = torch.zeros(B, T, device=device, dtype=dtype)

    for i in range(B):
        u_s, u_e = user_bounds[i]
        a_s, a_e = assistant_bounds[i]
        v_user = h[i, u_e] - h[i, u_s]
        d_all[i, u_s : u_e + 1] = h[i, u_s : u_e + 1] - h[i, u_s]
        d_all[i, a_s : a_e + 1] = v_user + (h[i, a_s : a_e + 1] - h[i, a_s])
        v_geodesic[i] = v_user + (h[i, a_e] - h[i, a_s])
        pos_mask[i, u_s : u_e + 1] = 1.0
        pos_mask[i, a_s : a_e + 1] = 1.0

    v_norm = F.normalize(v_geodesic, p=2, dim=1, eps=eps)
    dot = (d_all * v_norm.unsqueeze(1)).sum(dim=-1, keepdim=True)
    p_all = dot * v_norm.unsqueeze(1)
    e_all = d_all - p_all

    par_e = (p_all**2).sum(dim=-1)
    perp_e = (e_all**2).sum(dim=-1)
    sum_par = (par_e * pos_mask).sum()
    sum_perp = (perp_e * pos_mask).sum()
    n_pos = pos_mask.sum().clamp(min=1.0)
    return sum_par, sum_perp, n_pos


def tube_geometry_snr_batch(
    h: torch.Tensor,
    user_bounds: List[Tuple[int, int]],
    assistant_bounds: List[Tuple[int, int]],
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    STP-style decomposition on the same geometry as LyapunovTubeLoss (user+assistant spans).
    Signal ~ component of displacement along secant v_geo; noise ~ perpendicular e.
    Returns per-batch mean energies and SNR (ratio of batch means over masked positions).
    """
    sum_par, sum_perp, n_pos = _tube_snr_energy_sums(h, user_bounds, assistant_bounds, eps=eps)
    mean_par = sum_par / n_pos
    mean_perp = sum_perp / n_pos
    ratio = (mean_par / (mean_perp + eps)).clamp(min=eps)
    snr_db = (10.0 * torch.log10(ratio)).item()
    return {
        "mean_parallel_energy": float(mean_par.detach().float().item()),
        "mean_perpendicular_energy": float(mean_perp.detach().float().item()),
        "snr_linear": float(ratio.detach().float().item()),
        "snr_db": snr_db,
    }


def _collate_bounds(batch: List[dict]) -> dict:
    """Stack tensor fields; keep list bounds as tensors."""
    keys = batch[0].keys()
    out: Dict[str, Any] = {}
    for k in keys:
        vals = [b[k] for b in batch]
        if k in ("user_start_end", "assistant_start_end"):
            out[k] = torch.tensor(vals, dtype=torch.long)
        else:
            out[k] = torch.tensor(vals)
    return out


@torch.no_grad()
def compute_teacher_forced_token_accuracy(
    checkpoint_dir: Path,
    eval_jsonl: Path,
    base_model_name: str,
    method: str,
    max_length: int = 512,
    batch_size: int = 4,
    predictors: int = 0,
    max_batches: Optional[int] = None,
) -> float:
    """Fraction of correct next-token predictions on label positions (assistant / supervised)."""
    import compare_three_method as ctm  # late import avoids cycles with compare_three_method

    build_eval_dataset = ctm.build_eval_dataset

    checkpoint_dir = Path(checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint_dir),
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()
    ds = build_eval_dataset(
        eval_jsonl, tokenizer, base_model_name, max_length, method, predictors=predictors
    )
    cols = ["input_ids", "labels", "attention_mask"]
    ds = ds.remove_columns([c for c in ds.column_names if c not in cols])
    ds.set_format(type="torch", columns=cols)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=torch.utils.data.default_collate)

    device = next(model.parameters()).device
    total_ok = 0
    total = 0
    nbatch = 0
    for batch in tqdm(loader, desc=f"tok_acc[{method}]"):
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        labels = batch["labels"]
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        mask = shift_labels != -100
        pred = shift_logits.argmax(dim=-1)
        ok = (pred == shift_labels) & mask
        total_ok += ok.sum().item()
        total += mask.sum().item()
        nbatch += 1
        if max_batches is not None and nbatch >= max_batches:
            break
    if total == 0:
        return float("nan")
    return total_ok / total


@torch.no_grad()
def compute_tube_snr_proxy(
    checkpoint_dir: Path,
    eval_jsonl: Path,
    base_model_name: str,
    max_length: int = 512,
    batch_size: int = 2,
    predictors: int = 0,
    max_batches: Optional[int] = None,
    random_span_layer: int = -1,
) -> Dict[str, float]:
    """
    Tube SNR proxy over the eval run: ratio of *global* mean energies
    (sum of parallel / sum of perp over all masked positions), not the mean of per-batch ratios.
    Same tokenization as dynamics / linear dataset with bounds.
    """
    import stp

    checkpoint_dir = Path(checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint_dir),
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()
    ds = stp.load_and_prepare_dataset(
        str(eval_jsonl),
        tokenizer,
        base_model_name,
        max_length=max_length,
        predictors=predictors,
        regular=False,
        linear="dynamics",
    )
    cols = [
        "input_ids",
        "labels",
        "attention_mask",
        "user_start_end",
        "assistant_start_end",
    ]
    ds = ds.remove_columns([c for c in ds.column_names if c not in cols])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_collate_bounds)

    device = next(model.parameters()).device
    eps = 1e-8
    total_sum_par = 0.0
    total_sum_perp = 0.0
    total_n_pos = 0.0
    n_batches = 0
    for batch in tqdm(loader, desc="snr_proxy"):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True,
        )
        layer_idx = random_span_layer if random_span_layer >= 0 else len(out.hidden_states) - 1
        h = out.hidden_states[layer_idx]
        bsz = h.shape[0]
        usb = batch["user_start_end"]
        asb = batch["assistant_start_end"]
        user_bounds = []
        assistant_bounds = []
        for i in range(bsz):
            us0 = int(usb[i, 0].item())
            ue0 = int(usb[i, 1].item())
            as0 = int(asb[i, 0].item())
            ae0 = int(asb[i, 1].item())
            user_bounds.append((us0 + 1, ue0))
            assistant_bounds.append((as0 + 1, ae0))
        sum_par, sum_perp, n_pos = _tube_snr_energy_sums(h, user_bounds, assistant_bounds, eps=eps)
        total_sum_par += float(sum_par.detach().float().item())
        total_sum_perp += float(sum_perp.detach().float().item())
        total_n_pos += float(n_pos.detach().float().item())
        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break
    if n_batches == 0 or total_n_pos <= 0:
        nan = float("nan")
        return {
            "mean_parallel_energy": nan,
            "mean_perpendicular_energy": nan,
            "snr_linear": nan,
            "snr_db": nan,
        }
    mean_par = total_sum_par / total_n_pos
    mean_perp = total_sum_perp / total_n_pos
    ratio = max(mean_par / (mean_perp + eps), eps)
    snr_db = 10.0 * math.log10(ratio)
    return {
        "mean_parallel_energy": mean_par,
        "mean_perpendicular_energy": mean_perp,
        "snr_linear": ratio,
        "snr_db": snr_db,
    }


@torch.no_grad()
def compute_exact_match_accuracy(
    checkpoint_dir: Path,
    eval_jsonl: Path,
    base_model_name: str,
    max_new_tokens: int = 256,
    max_length: int = 512,
    max_examples: Optional[int] = None,
    eval_profile: str = "fork",
) -> float:
    """
    Greedy generation exact match vs gold assistant (synth / gsm8k-style rules).
    Uses the same chat formatting as evaluate.py for non-plain models.

    eval_profile: fork uses max_new_tokens/max_length as given; llm_jepa_official
    forces reference generation defaults (128 / 512).
    """
    import evaluate as eval_mod

    if eval_profile == "llm_jepa_official":
        max_new_tokens = eval_mod.LLM_JEPA_OFFICIAL_MAX_NEW_TOKENS
        max_length = eval_mod.LLM_JEPA_OFFICIAL_MAX_LENGTH
    elif eval_profile != "fork":
        raise ValueError(
            "eval_profile must be 'fork' or 'llm_jepa_official'; "
            f"got {eval_profile!r}"
        )

    checkpoint_dir = Path(checkpoint_dir)
    eval_stem = Path(eval_jsonl).stem
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint_dir),
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()
    gen_cfg = SimpleNamespace(max_length=max_length)

    lines = Path(eval_jsonl).read_text().splitlines()
    lines = [ln for ln in lines if ln.strip()]
    if max_examples is not None:
        lines = lines[:max_examples]

    ok = 0
    n = 0
    for line in tqdm(lines, desc=f"exact_match[{eval_stem}]"):
        ex = json.loads(line)
        messages = ex["messages"]
        full = eval_mod.get_messages(base_model_name, messages)
        prompt = eval_mod.format_conversation(full, tokenizer, include_assistant=False, plain=False)
        gen = eval_mod.generate_response(model, tokenizer, prompt, gen_cfg, max_new_tokens)
        if prediction_matches_gold(gen, messages, eval_stem):
            ok += 1
        n += 1
    return ok / n if n else float("nan")
