"""Lyapunov Tube Loss (ControlJEPA, NeurIPS 2026).

Given hidden states ``h`` of shape ``(B, T, D)`` and, for every example, the
inclusive token spans of the user turn ``[u_s, u_e]`` and of the assistant turn
``[a_s, a_e]``, the reference secant and the displacement (Eq. 9) are
built over the concatenation of the two spans, skipping the prompt-template
tokens between them::

    v_user = h[u_e] - h[u_s]
    v_geo  = v_user + (h[a_e] - h[a_s])                       # reference secant

    d_t = h[t] - h[u_s]                  for t in [u_s, u_e]  # user span
    d_t = v_user + (h[t] - h[a_s])       for t in [a_s, a_e]  # assistant span

The transversal residual (Eq. 10) and the dimensionless Lyapunov candidate
(Eq. 11, Def. 3.1) are::

    e_t = (I - v_hat v_hat^T) d_t,        v_hat = v_geo / ||v_geo||
    V_t = ||e_t||^2 / ||v_geo||^2

and the Lyapunov Tube Loss (Eqs. 17-18) penalises violations of the
discrete-time practical-ISS condition ``V_{t+1} <= gamma * V_t + tau`` (Eq. 12)::

    L_Lyap = 1/|T| * sum_{t in T} sigma(V_{t+1} - gamma * V_t - tau)

with ``sigma`` in {softplus, ReLU}.  ``T`` contains the user-internal and the
assistant-internal transitions; the jump across the prompt template
(``u_e -> a_s``) is not penalised.  By construction ``V = 0`` at both anchors.

Two alternative normalisations are kept for the ablation of Sec. 6.2:

* ``"d_model"``: ``V_t = ||e_t||^2 / D``    (unnormalised; collapses the scale)
* ``"d_t"``:     ``V_t = ||e_t||^2 / (anchor_eps * ||v_geo||^2 + ||d_t||^2)``
  (local-displacement normalisation; heteroscedastic)
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

VALID_NORMS: Tuple[str, ...] = ("v_geo", "d_model", "d_t")

Bounds = Sequence[Sequence[int]]


def _coerce_bounds(bounds: Bounds, name: str, batch_size: int) -> List[Tuple[int, int]]:
    """Convert a list / tensor of ``(start, end)`` pairs to Python ints."""
    if len(bounds) != batch_size:
        raise ValueError(f"{name} must have length B={batch_size}; got {len(bounds)}")
    out = []
    for i, pair in enumerate(bounds):
        if len(pair) != 2:
            raise ValueError(f"{name}[{i}] must have 2 elements; got {len(pair)}")
        out.append((int(pair[0]), int(pair[1])))
    return out


def chord_decomposition(
    h: torch.Tensor,
    user_bounds: Bounds,
    assistant_bounds: Bounds,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """Decompose a hidden-state trajectory relative to its reference secant.

    Returns a dict with

    * ``d``      ``(B, T, D)`` displacement ``d_t`` (zero outside the two spans),
    * ``e``      ``(B, T, D)`` transversal residual ``e_t``,
    * ``v_geo``  ``(B, D)``    reference secant,
    * ``valid``  ``(B, T-1)``  bool mask; ``valid[b, t]`` marks the transition
      ``t -> t+1`` as part of ``T``.

    Examples whose spans fall outside ``[0, T)`` are skipped (all-zero rows).
    """
    if h.dim() != 3:
        raise ValueError(f"h must be (B, T, D); got {tuple(h.shape)}")
    B, T, D = h.shape
    ub = _coerce_bounds(user_bounds, "user_bounds", B)
    ab = _coerce_bounds(assistant_bounds, "assistant_bounds", B)

    d = torch.zeros_like(h)
    v_geo = h.new_zeros(B, D)
    valid = torch.zeros(B, T - 1, device=h.device, dtype=torch.bool)

    for i, ((u_s, u_e), (a_s, a_e)) in enumerate(zip(ub, ab)):
        if not (0 <= u_s <= u_e < T and 0 <= a_s <= a_e < T):
            continue
        v_user = h[i, u_e] - h[i, u_s]
        d[i, u_s:u_e + 1] = h[i, u_s:u_e + 1] - h[i, u_s]
        d[i, a_s:a_e + 1] = v_user + (h[i, a_s:a_e + 1] - h[i, a_s])
        v_geo[i] = v_user + (h[i, a_e] - h[i, a_s])
        valid[i, u_s:u_e] = True
        valid[i, a_s:a_e] = True

    v_hat = F.normalize(v_geo, p=2, dim=-1, eps=eps).unsqueeze(1)
    e = d - (d * v_hat).sum(dim=-1, keepdim=True) * v_hat
    return {"d": d, "e": e, "v_geo": v_geo, "valid": valid}


class LyapunovTubeLoss(nn.Module):
    """Lyapunov Tube Loss, Eq. (18) of the paper.

    Args:
        gamma: contraction rate ``gamma`` in ``(0, 1)`` of the ISS condition.
        tau: ISS slack ``tau >= 0``. Together with ``gamma`` it sets the tube
            radius ``r* = sqrt(tau / (1 - gamma)) * ||v_geo||`` (Eq. 16).
        norm_mode: ``"v_geo"`` (the method), or ``"d_model"`` / ``"d_t"``
            (ablations of Sec. 6.2).
        use_softplus: use ``softplus`` as the violation penalty ``sigma``;
            ``ReLU`` otherwise.
        eps: numerical floor for ``||v_geo||``.
        anchor_eps: smoothing of the ``"d_t"`` denominator near the anchors,
            relative to ``||v_geo||^2`` (keeps the ablation scale-invariant).
    """

    def __init__(
        self,
        gamma: float = 0.95,
        tau: float = 1e-4,
        norm_mode: str = "v_geo",
        use_softplus: bool = True,
        eps: float = 1e-8,
        anchor_eps: float = 1e-3,
    ) -> None:
        super().__init__()
        if norm_mode not in VALID_NORMS:
            raise ValueError(f"norm_mode must be one of {VALID_NORMS}; got {norm_mode!r}")
        if not 0.0 < gamma < 1.0:
            raise ValueError(f"gamma must lie in (0, 1); got {gamma}")
        if tau < 0.0 or anchor_eps < 0.0:
            raise ValueError("tau and anchor_eps must be non-negative")
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.norm_mode = norm_mode
        self.use_softplus = bool(use_softplus)
        self.eps = float(eps)
        self.anchor_eps = float(anchor_eps)

    def lyapunov_candidate(self, geom: Dict[str, torch.Tensor]) -> torch.Tensor:
        """``V_t`` of shape ``(B, T)`` for the configured normalisation."""
        e_sq = geom["e"].pow(2).sum(dim=-1)
        v_sq = geom["v_geo"].pow(2).sum(dim=-1).clamp(min=self.eps).unsqueeze(1)
        if self.norm_mode == "v_geo":
            return e_sq / v_sq
        if self.norm_mode == "d_model":
            return e_sq / geom["e"].shape[-1]
        d_sq = geom["d"].pow(2).sum(dim=-1)
        if self.anchor_eps > 0.0:
            return e_sq / (self.anchor_eps * v_sq + d_sq)
        return e_sq / d_sq.clamp(min=self.eps)

    def forward(
        self,
        h: torch.Tensor,
        user_bounds: Bounds,
        assistant_bounds: Bounds,
        return_diagnostics: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Compute ``L_Lyap``.

        Args:
            h: ``(B, T, D)`` hidden states of the chosen decoder layer.
            user_bounds: per-example inclusive ``(u_s, u_e)`` token indices.
            assistant_bounds: per-example inclusive ``(a_s, a_e)`` token indices.
            return_diagnostics: also return detached scalars for logging.
        """
        geom = chord_decomposition(h, user_bounds, assistant_bounds, eps=self.eps)
        V = self.lyapunov_candidate(geom)
        violation = V[:, 1:] - self.gamma * V[:, :-1] - self.tau
        penalty = F.softplus(violation) if self.use_softplus else F.relu(violation)

        mask = geom["valid"].to(h.dtype)
        num_valid = geom["valid"].sum()
        n = num_valid.clamp(min=1).to(h.dtype)
        # Keep the graph connected (DDP) even if no example has a valid transition.
        loss = (penalty * mask).sum() / n + h.sum() * 0.0
        if not return_diagnostics:
            return loss

        with torch.no_grad():
            e_norm = geom["e"].norm(dim=-1)[:, :-1]
            v_norm = geom["v_geo"].norm(dim=-1)
            diag = {
                "mean_V": (V[:, 1:] * mask).sum() / n,
                "viol_rate": ((violation > 0).to(h.dtype) * mask).sum() / n,
                "mean_violation": (violation * mask).sum() / n,
                "mean_e_norm": (e_norm * mask).sum() / n,
                "mean_v_geo_norm": v_norm.mean(),
                "num_valid": num_valid,
            }
        return loss, diag
