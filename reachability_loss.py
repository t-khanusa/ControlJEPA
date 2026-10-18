"""Lyapunov-Reachability loss (``--linear=reach_JEPA`` / ``reach_JEPA_angle``).

Experimental variant kept for completeness; it is **not** part of the paper.
It uses the same chord construction as :mod:`lyapunov_loss` and adds a
longitudinal (progress) term to the Lyapunov candidate::

    V_t = alpha * V_perp(t) + beta * V_par(t)

    V_perp(t) = ||e_t||^2 / L^2                 transverse_norm="L"
              = ||e_t||^2 / ||d_t||^2           transverse_norm="d_t" (sin^2 of the angle;
                                                anchors with d_t = 0 are masked out)
    V_par(t)  = max(0, tau_t - p_t / L)^2       progress_mode="schedule_asym"
              = (1 - p_t / L)^2                 progress_mode="endpoint"

where ``L = ||v_geo||``, ``p_t = <d_t, v_hat>`` and ``tau_t = rank(t) / (N - 1)``
is a token-rank schedule over the ``N`` positions of the user and assistant
spans.  The contraction ``V_{t+1} <= gamma * V_t + tau`` is penalised with the
same softplus / ReLU surrogate as the Lyapunov Tube Loss.  ``beta = 0`` gives a
pure transverse loss and ``alpha = 0`` a progress-only loss.
"""

from __future__ import annotations

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from lyapunov_loss import Bounds, _coerce_bounds

VALID_PROGRESS_MODES: Tuple[str, ...] = ("endpoint", "schedule_asym")
VALID_REACH_TRANSVERSE_NORMS: Tuple[str, ...] = ("L", "d_t")


class LyapunovReachabilityLoss(nn.Module):
    """Joint transverse-longitudinal Lyapunov loss.

    Args:
        alpha: weight of the transverse term (``>= 0``).
        beta: weight of the progress term (``>= 0``); ``alpha + beta > 0``.
        gamma: contraction rate of ``V_{t+1} <= gamma * V_t + tau``.
        tau: additive slack.
        progress_mode: ``"schedule_asym"`` or ``"endpoint"``.
        transverse_norm: ``"L"`` or ``"d_t"`` (angle form, ``reach_JEPA_angle``).
        use_softplus: softplus surrogate if True, ReLU otherwise.
        eps: numerical floor.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.9,
        tau: float = 1e-4,
        progress_mode: str = "schedule_asym",
        transverse_norm: str = "L",
        use_softplus: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if alpha < 0 or beta < 0:
            raise ValueError(f"alpha and beta must be non-negative; got alpha={alpha}, beta={beta}")
        if alpha == 0 and beta == 0:
            raise ValueError("at least one of alpha, beta must be positive")
        if progress_mode not in VALID_PROGRESS_MODES:
            raise ValueError(f"progress_mode must be one of {VALID_PROGRESS_MODES}; got {progress_mode!r}")
        if transverse_norm not in VALID_REACH_TRANSVERSE_NORMS:
            raise ValueError(
                f"transverse_norm must be one of {VALID_REACH_TRANSVERSE_NORMS}; got {transverse_norm!r}"
            )
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.progress_mode = progress_mode
        self.transverse_norm = transverse_norm
        self.use_softplus = bool(use_softplus)
        self.eps = float(eps)

    def forward(
        self,
        h: torch.Tensor,
        user_bounds: Bounds,
        assistant_bounds: Bounds,
        return_diagnostics: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if h.dim() != 3:
            raise ValueError(f"h must be (B, T, D); got {tuple(h.shape)}")
        B, T, D = h.shape
        ub = _coerce_bounds(user_bounds, "user_bounds", B)
        ab = _coerce_bounds(assistant_bounds, "assistant_bounds", B)

        d = torch.zeros_like(h)
        v_geo = h.new_zeros(B, D)
        valid = torch.zeros(B, T - 1, device=h.device, dtype=torch.bool)
        positions = torch.zeros(B, T, device=h.device, dtype=torch.bool)
        schedule = h.new_zeros(B, T)  # token-rank progress target tau_t

        for i, ((u_s, u_e), (a_s, a_e)) in enumerate(zip(ub, ab)):
            if not (0 <= u_s <= u_e < T and 0 <= a_s <= a_e < T):
                continue
            v_user = h[i, u_e] - h[i, u_s]
            d[i, u_s:u_e + 1] = h[i, u_s:u_e + 1] - h[i, u_s]
            d[i, a_s:a_e + 1] = v_user + (h[i, a_s:a_e + 1] - h[i, a_s])
            v_geo[i] = v_user + (h[i, a_e] - h[i, a_s])
            valid[i, u_s:u_e] = True
            # As in the original reach_JEPA runs, the last assistant transition
            # (a_e - 1 -> a_e) is not constrained.
            valid[i, a_s:max(a_s, a_e - 1)] = True
            n_u, n_a = u_e - u_s + 1, a_e - a_s + 1
            if n_u + n_a > 1:
                ranks = torch.arange(n_u + n_a, device=h.device, dtype=h.dtype) / float(n_u + n_a - 1)
                schedule[i, u_s:u_e + 1] = ranks[:n_u]
                schedule[i, a_s:a_e + 1] = ranks[n_u:]
            positions[i, u_s:u_e + 1] = True
            positions[i, a_s:a_e + 1] = True

        L_sq = (v_geo * v_geo).sum(dim=-1).clamp(min=self.eps)
        L = L_sq.sqrt()
        v_hat = v_geo / L.unsqueeze(-1).clamp(min=self.eps)
        p = (d * v_hat.unsqueeze(1)).sum(dim=-1)
        e = d - p.unsqueeze(-1) * v_hat.unsqueeze(1)
        e_sq = e.pow(2).sum(dim=-1)
        d_sq = (d * d).sum(dim=-1)

        if self.transverse_norm == "L":
            transversal = e_sq / L_sq.unsqueeze(1)
            anchor_mask = None
        else:
            # Scale-invariant mask of positions with a non-trivial displacement.
            anchor_mask = d_sq > self.eps * L_sq.unsqueeze(1)
            transversal = torch.where(anchor_mask, e_sq / d_sq.clamp(min=self.eps), torch.zeros_like(e_sq))

        p_over_L = p / L.unsqueeze(1)
        if self.progress_mode == "endpoint":
            gap = 1.0 - p_over_L
            progress = gap ** 2
        else:
            gap = schedule - p_over_L  # positive = lagging behind the schedule
            progress = gap.clamp(min=0.0) ** 2

        V = self.alpha * transversal + self.beta * progress
        violation = V[:, 1:] - self.gamma * V[:, :-1] - self.tau
        penalty = F.softplus(violation) if self.use_softplus else F.relu(violation)

        if anchor_mask is not None:
            # Transitions leaving an anchor (where V_perp is forced to 0) are dropped.
            valid = valid & anchor_mask[:, :-1]
        mask = valid.to(h.dtype)
        num_valid = valid.sum()
        n = num_valid.clamp(min=1).to(h.dtype)
        if num_valid.item() > 0:
            loss = (penalty * mask).sum() / n
        else:  # keep the graph connected so DDP sees a gradient for every rank
            loss = h.sum() * 0.0
        if not return_diagnostics:
            return loss

        with torch.no_grad():
            mask_full = torch.zeros_like(V)
            mask_full[:, :-1] = mask
            n_full = mask_full.sum().clamp(min=1.0)
            pos = positions.to(h.dtype)
            n_pos = pos.sum().clamp(min=1.0)
            diag = {
                "mean_V": (V * mask_full).sum() / n_full,
                "mean_V_transversal": (self.alpha * transversal * mask_full).sum() / n_full,
                "mean_V_progress": (self.beta * progress * mask_full).sum() / n_full,
                "mean_p_over_L": (p_over_L * pos).sum() / n_pos,
                "mean_e2_over_L2": (transversal * pos).sum() / n_pos,
                "mean_tau": (schedule * pos).sum() / n_pos,
                "mean_progress_gap": (gap * pos).sum() / n_pos,
                "lag_rate": ((gap > 0).to(h.dtype) * pos).sum() / n_pos,
                "mean_raw_violation": (violation * mask).sum() / n,
                "viol_rate": ((violation > 0).to(h.dtype) * mask).sum() / n,
                "mean_e_sq": (e_sq * pos).sum() / n_pos,
                "mean_e_norm": (e_sq.clamp(min=0).sqrt() * pos).sum() / n_pos,
                "mean_v_geo_sq": L_sq.mean(),
                "mean_v_geo_norm": L.mean(),
                "num_valid": num_valid,
            }
        return loss, diag
