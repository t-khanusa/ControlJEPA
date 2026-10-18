"""Unit tests for the Lyapunov Tube Loss (run with `pytest tests/`)."""

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lyapunov_loss import LyapunovTubeLoss, chord_decomposition  # noqa: E402

USER = [(1, 6), (2, 9)]
ASSISTANT = [(10, 17), (12, 19)]


def random_states(seed=0, B=2, T=20, D=16):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, D, generator=g, dtype=torch.float64)


def reference_loss(h, user, assistant, gamma, tau):
    """Direct, per-token transcription of Eqs. (9)-(11), (17) and (18)."""
    terms = []
    for b in range(h.shape[0]):
        (u_s, u_e), (a_s, a_e) = user[b], assistant[b]
        v_user = h[b, u_e] - h[b, u_s]
        v_geo = v_user + h[b, a_e] - h[b, a_s]
        v_hat = v_geo / v_geo.norm()

        def V(t):
            d = h[b, t] - h[b, u_s] if t <= u_e else v_user + h[b, t] - h[b, a_s]
            e = d - (d @ v_hat) * v_hat
            return (e @ e) / (v_geo @ v_geo)

        for lo, hi in ((u_s, u_e), (a_s, a_e)):
            for t in range(lo, hi):
                terms.append(F.softplus(V(t + 1) - gamma * V(t) - tau))
    return torch.stack(terms).mean()


def test_matches_reference_equations():
    h = random_states()
    loss = LyapunovTubeLoss(gamma=0.95, tau=1e-4)(h, USER, ASSISTANT)
    ref = reference_loss(h, USER, ASSISTANT, gamma=0.95, tau=1e-4)
    assert torch.allclose(loss, ref, atol=1e-10)


def test_lyapunov_candidate_vanishes_at_anchors():
    h = random_states()
    loss_fn = LyapunovTubeLoss()
    V = loss_fn.lyapunov_candidate(chord_decomposition(h, USER, ASSISTANT))
    for b, ((u_s, _), (_, a_e)) in enumerate(zip(USER, ASSISTANT)):
        assert V[b, u_s].abs() < 1e-12  # d = 0 at the start anchor
        assert V[b, a_e].abs() < 1e-12  # d = v_geo at the end anchor


@pytest.mark.parametrize("norm_mode", ["v_geo", "d_t"])
def test_scale_invariance(norm_mode):
    h = random_states()
    loss_fn = LyapunovTubeLoss(norm_mode=norm_mode)
    base = loss_fn(h, USER, ASSISTANT)
    for scale in (1e-3, 10.0, 1e3):
        assert torch.allclose(loss_fn(scale * h, USER, ASSISTANT), base, atol=1e-8)


def test_unnormalised_ablation_is_not_scale_invariant():
    h = random_states()
    loss_fn = LyapunovTubeLoss(norm_mode="d_model", use_softplus=False, tau=0.0)
    assert loss_fn(100.0 * h, USER, ASSISTANT) > loss_fn(h, USER, ASSISTANT)


def test_trajectory_on_secant_satisfies_iss():
    # Hidden states moving along a single direction have e_t = 0, hence V_t = 0.
    B, T, D = 2, 20, 8
    direction = torch.randn(D, dtype=torch.float64)
    h = torch.cumsum(torch.rand(B, T, 1, dtype=torch.float64), dim=1) * direction
    relu = LyapunovTubeLoss(use_softplus=False, tau=1e-4)
    assert relu(h, USER, ASSISTANT) == 0.0
    # The softplus surrogate bottoms out at softplus(-tau) ~ ln 2 (Sec. 5.4).
    softplus = LyapunovTubeLoss(tau=1e-4)
    assert abs(softplus(h, USER, ASSISTANT).item() - F.softplus(torch.tensor(-1e-4)).item()) < 1e-8


def test_prompt_gap_is_not_penalised():
    h = random_states()
    valid = chord_decomposition(h, USER, ASSISTANT)["valid"]
    for b, ((u_s, u_e), (a_s, a_e)) in enumerate(zip(USER, ASSISTANT)):
        expected = torch.zeros(h.shape[1] - 1, dtype=torch.bool)
        expected[u_s:u_e] = True
        expected[a_s:a_e] = True
        assert torch.equal(valid[b], expected)
    # Perturbing a template token between the spans leaves the loss unchanged.
    loss_fn = LyapunovTubeLoss()
    h2 = h.clone()
    h2[0, 8] += 100.0
    assert torch.allclose(loss_fn(h, USER, ASSISTANT), loss_fn(h2, USER, ASSISTANT))


def test_gradients_are_finite_and_diagnostics_present():
    h = random_states().float().requires_grad_()
    loss, diag = LyapunovTubeLoss()(h, USER, ASSISTANT, return_diagnostics=True)
    loss.backward()
    assert torch.isfinite(h.grad).all() and h.grad.abs().sum() > 0
    assert {"mean_V", "viol_rate", "mean_e_norm", "mean_v_geo_norm", "num_valid"} <= diag.keys()
    assert diag["num_valid"].item() == sum(e - s for s, e in USER + ASSISTANT)


def test_malformed_spans_are_skipped():
    h = random_states().requires_grad_()
    loss = LyapunovTubeLoss()(h, [(5, 100), (-1, 3)], ASSISTANT)
    loss.backward()
    assert loss.item() == 0.0


def test_invalid_hyperparameters_raise():
    with pytest.raises(ValueError):
        LyapunovTubeLoss(gamma=1.0)
    with pytest.raises(ValueError):
        LyapunovTubeLoss(norm_mode="cosine")
