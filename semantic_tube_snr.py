"""
Signal-to-noise tooling for tube-geometry diagnostics.

This module distinguishes two notions:

**Appendix H (information-theoretic, not directly observable in latents)**  
Additive Gaussian surrogate ``X = Z + N`` with

    SNR_lin = 𝔼‖Z‖² / 𝔼‖N‖²

and (Gauss-channel capacity form, Shannon 1948)

    I(Y; X) ≤ ½·log(1 + SNR_lin)   using natural logarithm ⇒ nats

Corollaries H.2–H.3 relate this SNR *upper bound context* to data efficiency ``m``
and error rate ``P_e``. These are **scaling-law / proof tools**; estimating the true
latent SNR is stated to be **intractable** without the ground-truth manifold.

**Figure 1 + STP intuition (computable geometric proxy used in experiments here)**  
For a full-sequence hidden trajectory ``h_t``, approximate the Semantic Tube axis
with the chord

    v_geo := (h[u_e] − h[u_s]) + (h[a_e] − h[a_s])

matching the NL-RX / chat span convention in ``stp.py`` / ``lyapunov_loss.py``.
For displacement ``d_t`` (user + assistant halves as in Lyapunov diagnostics):

    signal   p_t  := proj_{v̂_geo}( d_t )
    noise    e_t  := d_t − p_t   (orthogonal / “tube normal” residual)

Operational **trajectory SNR proxy**:

    SNR_geom = 𝔼 ‖p_t‖² / 𝔼 ‖e_t‖²

(and angle form 𝔼 ‖e_t‖² / 𝔼 ‖d_t‖², etc.). Interpreting ``SNR_geom`` as plugging
into Appendix H Gaussian formulas is exactly the “validate via predicted impact”
program the authors describe — not a literal measurement of 𝔼‖Z‖²/𝔼‖N‖².
"""

from __future__ import annotations

import math
from typing import TypedDict


class TubeGeometryTotals(TypedDict):
    """Sufficient statistics for pooled SNR over many (token × example) slices."""

    sum_p2_signal: float
    sum_e2_noise: float
    sum_d2: float
    sum_h2: float
    sum_sin2_theta: float
    sum_vgeo2: float
    n_tokens: int
    n_examples: int


def appendix_h_mi_nats_additive_gaussian(snr_linear: float) -> float:
    """Shannon mutual-information surrogate I ≈ (1/2)·ln(1 + SNR) in *nats*.

    Appendix H (Eq. after Lemma H.1) under the Gaussian channel approximation.
    """
    snr_linear = float(snr_linear)
    if snr_linear < 0:
        raise ValueError("SNR must be non-negative.")
    # cap huge values for numerical stability
    snr_linear = min(snr_linear, 1e128)
    return 0.5 * math.log(1.0 + snr_linear)


def appendix_h_mi_bits_additive_gaussian(snr_linear: float) -> float:
    """Same as ``appendix_h_mi_nats_additive_gaussian`` but in bits (/ ln 2)."""
    return appendix_h_mi_nats_additive_gaussian(snr_linear) / math.log(2.0)


def appendix_h_corollary_h2_min_observations(H_Y_nat: float, epsilon_nat: float, snr_linear: float) -> float:
    """Corollary H.2: lower bound ``m ≥ (H(Y) - ε) / (½·log(1+SNR))`` with logarithm matching Appendix H."""
    denom = appendix_h_mi_nats_additive_gaussian(snr_linear)
    if denom <= 0:
        return float("inf")
    num = max(0.0, float(H_Y_nat) - float(epsilon_nat))
    return num / denom


def appendix_h_corollary_h3_fanoe_pe_lower_bound_approx(
    H_Y_nat: float,
    *,
    m: int,
    snr_linear: float,
    log_vocab_minus_one_nat: float,
) -> float:
    """Rough Fano-linked lower bound (Eq. 8 style, natural logs).

    P_e ≳ ( H(Y) - m · ½·log(1+SNR) ) / log(|𝒱|-1)

    Requires an estimate of ``H(Y)`` for the target-token distribution (or an upper bound
    such as uniform ``log(|𝒱|)``).
    """
    mi_one = appendix_h_mi_nats_additive_gaussian(snr_linear)
    num = float(H_Y_nat) - int(m) * mi_one
    return max(0.0, num / max(log_vocab_minus_one_nat, 1e-12))


try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[misc, assignment]


def accumulate_tube_projection_stats(
    h: "torch.Tensor",
    *,
    u_s: int,
    u_e: int,
    a_s: int,
    a_e: int,
    eps: float = 1e-8,
) -> dict[str, float]:
    """Accumulate summed energies for one sequence ``h`` of shape `(T, D)`.

    Indexing follows ``diagnose_signal_noise.py``:

    ``u_s..u_e`` and ``a_s..a_e`` are **inclusive** integer bounds on USER and ASSISTANT
    **content spans** (after resolving ``find_start_end`` + "+1 start" conventions).
    """
    if torch is None:
        raise ImportError("torch is required for accumulate_tube_projection_stats")

    device = h.device
    T, D = h.shape[0], h.shape[-1]
    if not (0 <= u_s <= u_e < T and 0 <= a_s <= a_e < T):
        raise ValueError(f"Bad spans {(u_s, u_e, a_s, a_e)} vs T={T}")

    v_user = h[u_e] - h[u_s]
    v_asst = h[a_e] - h[a_s]
    v_geo = v_user + v_asst
    v_norm = v_geo.norm().clamp_min(eps)
    v_hat = v_geo / v_norm

    valid_idx = torch.cat([
        torch.arange(u_s, u_e + 1, device=device, dtype=torch.long),
        torch.arange(a_s, a_e + 1, device=device, dtype=torch.long),
    ])

    d = torch.zeros(valid_idx.numel(), D, device=device, dtype=h.dtype)
    in_user = valid_idx <= u_e
    if in_user.any():
        d[in_user] = h[valid_idx[in_user]] - h[u_s]
    if (~in_user).any():
        d[~in_user] = v_user + (h[valid_idx[~in_user]] - h[a_s])

    proj = (d @ v_hat).unsqueeze(-1) * v_hat.unsqueeze(0)
    e = d - proj

    d2 = (d * d).sum(-1)
    p2 = (proj * proj).sum(-1)
    e2 = (e * e).sum(-1)
    sin2 = e2 / d2.clamp_min(eps)
    h2_sel = (h[valid_idx] * h[valid_idx]).sum(-1)

    return {
        "sum_p2_signal": float(p2.sum().item()),
        "sum_e2_noise": float(e2.sum().item()),
        "sum_d2": float(d2.sum().item()),
        "sum_h2": float(h2_sel.sum().item()),
        "sum_sin2_theta": float(sin2.sum().item()),
        "sum_vgeo2": float((v_geo * v_geo).sum().item()),
        "n_tokens": int(valid_idx.numel()),
        "example_batch": int(1),
    }


def pool_tube_geometry_stats(partials: list[dict[str, float]]) -> dict[str, float]:
    """Fold list of partial dicts from ``accumulate_tube_projection_stats``."""
    agg = TubeGeometryTotals(
        sum_p2_signal=0.0,
        sum_e2_noise=0.0,
        sum_d2=0.0,
        sum_h2=0.0,
        sum_sin2_theta=0.0,
        sum_vgeo2=0.0,
        n_tokens=0,
        n_examples=0,
    )
    for p in partials:
        agg["sum_p2_signal"] += p["sum_p2_signal"]
        agg["sum_e2_noise"] += p["sum_e2_noise"]
        agg["sum_d2"] += p["sum_d2"]
        agg["sum_h2"] += p["sum_h2"]
        agg["sum_sin2_theta"] += p["sum_sin2_theta"]
        agg["sum_vgeo2"] += p["sum_vgeo2"]
        agg["n_tokens"] += int(p["n_tokens"])
        agg["n_examples"] += int(p.get("example_batch", 1))

    nt = max(agg["n_tokens"], 1)
    ne = max(agg["n_examples"], 1)
    mean_p2 = agg["sum_p2_signal"] / nt
    mean_e2 = agg["sum_e2_noise"] / nt
    mean_d2 = agg["sum_d2"] / nt
    mean_h2 = agg["sum_h2"] / nt

    eps = 1e-12
    return {
        "mean_p2_signal": mean_p2,
        "mean_e2_noise": mean_e2,
        "mean_d2": mean_d2,
        "mean_h2": mean_h2,
        "mean_sin2_theta": agg["sum_sin2_theta"] / nt,
        "mean_vgeo2": agg["sum_vgeo2"] / ne,
        "snr_geom_p_over_e": mean_p2 / max(mean_e2, eps),
        "mean_e2_over_mean_d2_angle_form": mean_e2 / max(mean_d2, eps),
        "vgeo2_over_mean_h2": (agg["sum_vgeo2"] / ne) / max(mean_h2, eps),
        "n_tokens": float(agg["n_tokens"]),
        "n_examples": float(agg["n_examples"]),
    }


def appendix_h_linked_summary(snr_geom: float, *, vocab_size: int | None = None, m_seen: int | None = None) -> dict[str, float]:
    """Attach Gaussian-channel surrogates to a geometric SNR estimate (explicitly labelled *proxy*)."""
    mi_n = appendix_h_mi_nats_additive_gaussian(snr_geom)
    mi_b = appendix_h_mi_bits_additive_gaussian(snr_geom)
    out: dict[str, float] = {
        "snr_geom_linear_input": float(snr_geom),
        "appendix_h_gaussian_MI_nats_if_snr_geom": mi_n,
        "appendix_h_gaussian_MI_bits_if_snr_geom": mi_b,
    }
    if vocab_size is not None and vocab_size > 2:
        H_max_nat = math.log(float(vocab_size))
        log_vm1_nat = math.log(float(vocab_size - 1))
        # uniform upper bound entropy of next-token targets
        out["entropy_Y_uniform_upper_bound_nats"] = H_max_nat
        if m_seen is not None:
            out["appendix_H3_fanoe_pe_lower_bound_uniform_HY"] = appendix_h_corollary_h3_fanoe_pe_lower_bound_approx(
                H_Y_nat=H_max_nat,
                m=int(m_seen),
                snr_linear=snr_geom,
                log_vocab_minus_one_nat=log_vm1_nat,
            )
    return out
