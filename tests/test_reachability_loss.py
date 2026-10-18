"""Unit tests for the experimental Lyapunov-Reachability loss (reach_JEPA)."""

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reachability_loss import LyapunovReachabilityLoss  # noqa: E402

USER = [(1, 6), (2, 9)]
ASSISTANT = [(10, 17), (12, 19)]


def random_states(seed=0, B=2, T=20, D=16):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, D, generator=g, dtype=torch.float64)


def reference_loss(h, user, assistant, alpha, beta, gamma, tau, progress_mode):
    """Per-token transcription with transverse_norm="L"."""
    terms = []
    for b in range(h.shape[0]):
        (u_s, u_e), (a_s, a_e) = user[b], assistant[b]
        v_user = h[b, u_e] - h[b, u_s]
        v_geo = v_user + h[b, a_e] - h[b, a_s]
        L = v_geo.norm()
        v_hat = v_geo / L
        order = list(range(u_s, u_e + 1)) + list(range(a_s, a_e + 1))
        rank = {t: k / (len(order) - 1) for k, t in enumerate(order)}

        def V(t):
            d = h[b, t] - h[b, u_s] if t <= u_e else v_user + h[b, t] - h[b, a_s]
            p = d @ v_hat
            e = d - p * v_hat
            if progress_mode == "endpoint":
                progress = (1 - p / L) ** 2
            else:
                progress = torch.clamp(rank[t] - p / L, min=0) ** 2
            return alpha * (e @ e) / L ** 2 + beta * progress

        # The last assistant transition (a_e - 1 -> a_e) is not constrained.
        for lo, hi in ((u_s, u_e), (a_s, a_e - 1)):
            for t in range(lo, hi):
                terms.append(F.softplus(V(t + 1) - gamma * V(t) - tau))
    return torch.stack(terms).mean()


@pytest.mark.parametrize("progress_mode", ["schedule_asym", "endpoint"])
@pytest.mark.parametrize("alpha,beta", [(1.0, 1.0), (1.0, 0.0), (0.0, 1.0), (0.3, 2.0)])
def test_matches_reference(progress_mode, alpha, beta):
    h = random_states()
    loss = LyapunovReachabilityLoss(alpha=alpha, beta=beta, gamma=0.9, tau=1e-4,
                                    progress_mode=progress_mode)(h, USER, ASSISTANT)
    ref = reference_loss(h, USER, ASSISTANT, alpha, beta, 0.9, 1e-4, progress_mode)
    assert torch.allclose(loss, ref, atol=1e-10)


@pytest.mark.parametrize("transverse_norm", ["L", "d_t"])
def test_scale_invariance(transverse_norm):
    h = random_states()
    loss_fn = LyapunovReachabilityLoss(transverse_norm=transverse_norm)
    base = loss_fn(h, USER, ASSISTANT)
    for scale in (1e-2, 10.0, 1e2):
        assert torch.allclose(loss_fn(scale * h, USER, ASSISTANT), base, atol=1e-8)


def test_angle_form_masks_anchor_transitions():
    h = random_states()
    loss_fn = LyapunovReachabilityLoss(transverse_norm="d_t")
    _, diag_l = LyapunovReachabilityLoss(transverse_norm="L")(h, USER, ASSISTANT, return_diagnostics=True)
    _, diag_a = loss_fn(h, USER, ASSISTANT, return_diagnostics=True)
    # Transitions leaving the start anchor (d_t = 0) are dropped in the angle form.
    assert diag_a["num_valid"].item() == diag_l["num_valid"].item() - len(USER)


def test_gradients_and_diagnostics():
    h = random_states().float().requires_grad_()
    loss, diag = LyapunovReachabilityLoss()(h, USER, ASSISTANT, return_diagnostics=True)
    loss.backward()
    assert torch.isfinite(h.grad).all() and h.grad.abs().sum() > 0
    assert {"mean_V", "mean_V_transversal", "mean_V_progress", "mean_p_over_L", "lag_rate",
            "mean_raw_violation", "viol_rate", "num_valid"} <= diag.keys()
    expected = sum(u_e - u_s for u_s, u_e in USER) + sum(a_e - 1 - a_s for a_s, a_e in ASSISTANT)
    assert diag["num_valid"].item() == expected


def test_malformed_spans_are_skipped():
    h = random_states().requires_grad_()
    loss = LyapunovReachabilityLoss()(h, [(5, 100), (-1, 3)], ASSISTANT)
    loss.backward()
    assert loss.item() == 0.0


def test_invalid_hyperparameters_raise():
    with pytest.raises(ValueError):
        LyapunovReachabilityLoss(alpha=0.0, beta=0.0)
    with pytest.raises(ValueError):
        LyapunovReachabilityLoss(alpha=-1.0)
    with pytest.raises(ValueError):
        LyapunovReachabilityLoss(progress_mode="linear")
    with pytest.raises(ValueError):
        LyapunovReachabilityLoss(transverse_norm="d_model")
