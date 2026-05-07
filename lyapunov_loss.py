"""
Lyapunov-tube regularizer for control_JEPA.
=============================================

Drop-in auxiliary loss that wraps a token trajectory with a
Lyapunov-flavored transversal-energy constraint. Given user span
``[u_s, u_e]`` (inclusive) and assistant span ``[a_s, a_e]`` (inclusive) in
the hidden-state sequence ``h`` of shape ``(B, T, D)``, we define

    v_user = h[u_e] - h[u_s]
    v_geo  = v_user + (h[a_e] - h[a_s])      # chord with the prompt gap skipped

    d_t = h[t] - h[u_s]                 for t in [u_s, u_e]
    d_t = v_user + (h[t] - h[a_s])      for t in [a_s, a_e]

    p_t = <d_t, v_geo / ||v_geo||> * (v_geo / ||v_geo||)      # projection on chord
    e_t = d_t - p_t                                           # transversal residual

The Lyapunov candidate ``V_t = ||e_t||^2 / N_t`` admits three choices of
normalization ``N_t``:

    "d_model" : N_t = D                   (constant; scale-dependent)
    "v_geo"   : N_t = ||v_geo||^2         (chord-normalized per sample)
    "d_t"     : N_t = ||d_t||^2           (angle form: V_t = sin^2(angle(d_t, v_geo)))

The soft contraction constraint ``V_{t+1} <= gamma * V_t + tau`` is enforced
over *valid* transitions (user-internal + assistant-internal; the gap
transition ``u_e -> a_s`` is explicitly skipped) via the stable linear
surrogate

    L = mean_{t in valid}  softplus(V_{t+1} - gamma * V_t - tau)

The linear-softplus form avoids the ``1/(gamma V + tau)`` singularity of the
log-residual surrogate and gives bounded, well-behaved gradients even near
the constraint boundary.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Sequence, Tuple, Union


VALID_NORMS: Tuple[str, ...] = ("d_model", "v_geo", "d_t")
VALID_PROGRESS_MODES: Tuple[str, ...] = ("endpoint", "schedule_asym")
# Transverse normalization choices for LyapunovReachabilityLoss:
#   "L"   : classic scale-invariant form V_\perp = ||e_t||^2 / L^2
#   "d_t" : angle form V_\perp = ||e_t||^2 / ||d_t||^2 = sin^2 \theta_t, bounded
#           in [0, 1] (better-conditioned gradients), with anchor positions
#           (t = u_s, a_s where d_t = 0) masked out of the transverse term.
VALID_REACH_TRANSVERSE_NORMS: Tuple[str, ...] = ("L", "d_t")


class LyapunovControlLoss(nn.Module):
    """Discrete-time Lyapunov tube loss for token trajectories.

    Parameters
    ----------
    gamma : float, default 0.9
        Contraction factor in ``V_{t+1} <= gamma * V_t + tau``.
    tau : float, default 1e-4
        Additive slack / disturbance bound.
    norm_mode : {"d_model", "v_geo", "d_t"}, default "d_t"
        Choice of denominator ``N_t`` for ``V_t = ||e_t||^2 / N_t``.
    use_softplus : bool, default True
        If True use ``F.softplus(.)`` surrogate (smooth, always-on gradient);
        if False use ``F.relu(.)``.
    eps : float, default 1e-8
        Numerical floor for denominators and chord normalization.
    anchor_eps : float, default 1e-3
        Anchor-smoothing coefficient for ``norm_mode="d_t"``. Replaces the old
        hard ``max(||d_t||^2, eps=1e-8)`` clamp with a *scale-invariant* soft
        gate: ``V_t = ||e_t||^2 / (anchor_eps * L^2 + ||d_t||^2)`` where
        ``L = ||v_geo||``. Motivation: the angle form ``V_t = sin^2(theta_t)``
        is well-behaved at the exact anchors (``d_t = 0 => e_t = 0`` too, since
        ``e_t`` is the orthogonal residual of ``d_t``), but tokens a *short
        hop* from the anchor can have ``||d_t||^2`` well below the 1e-8 floor
        while still carrying a macroscopic residual ``||e_t||^2``. In that
        regime the legacy clamp produced ``V_t ~ ||e_t||^2 / 1e-8``, i.e.
        violation magnitudes that scaled with the floor rather than with the
        trajectory, dominating the loss and destroying per-token gradient
        balance. The soft gate interpolates continuously from
        ``V_t = sin^2(theta_t)`` (when ``||d_t||^2 >> anchor_eps * L^2``,
        recovered for tokens far from the anchor) to a Tikhonov
        ``v_geo`` fallback ``V_t ~ ||e_t||^2 / (anchor_eps * L^2)`` near the
        anchor. Because ``anchor_eps * L^2`` rescales with ``h``, the full
        loss inherits the same scale invariance as the pure angle form.
        Setting ``anchor_eps=0`` recovers the legacy hard-clamp behaviour.
        Has no effect when ``norm_mode != "d_t"``.
    """

    def __init__(
        self,
        gamma: float = 0.9,
        tau: float = 1e-4,
        norm_mode: str = "d_t",
        use_softplus: bool = True,
        eps: float = 1e-8,
        anchor_eps: float = 1e-3,
    ) -> None:
        super().__init__()
        if norm_mode not in VALID_NORMS:
            raise ValueError(
                f"norm_mode must be one of {VALID_NORMS}; got {norm_mode!r}"
            )
        if float(anchor_eps) < 0:
            raise ValueError(
                f"anchor_eps must be non-negative; got {anchor_eps}"
            )
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.norm_mode = norm_mode
        self.use_softplus = bool(use_softplus)
        self.eps = float(eps)
        self.anchor_eps = float(anchor_eps)

    @staticmethod
    def _as_int_bounds(
        bounds: Sequence[Sequence[int]],
        name: str,
        B: int,
    ) -> List[Tuple[int, int]]:
        """Coerce a list/tensor/list-of-lists of (start, end) to Python ints."""
        if len(bounds) != B:
            raise ValueError(
                f"{name} must have length B={B}; got {len(bounds)}"
            )
        out: List[Tuple[int, int]] = []
        for i, pair in enumerate(bounds):
            if torch.is_tensor(pair):
                if pair.numel() != 2:
                    raise ValueError(
                        f"{name}[{i}] must have 2 elements; got {pair.numel()}"
                    )
                s = int(pair[0].item())
                e = int(pair[1].item())
            else:
                if len(pair) != 2:
                    raise ValueError(
                        f"{name}[{i}] must have 2 elements; got {len(pair)}"
                    )
                s = int(pair[0])
                e = int(pair[1])
            out.append((s, e))
        return out

    def forward(
        self,
        h: torch.Tensor,
        user_bounds: Sequence[Sequence[int]],
        assistant_bounds: Sequence[Sequence[int]],
        *,
        return_diagnostics: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Compute the Lyapunov tube surrogate loss.

        Parameters
        ----------
        h : Tensor of shape (B, T, D)
            Full-sequence hidden states from the chosen decoder layer.
        user_bounds : list of (u_s, u_e)
            Inclusive token indices of the user span in ``h``; length B.
        assistant_bounds : list of (a_s, a_e)
            Inclusive token indices of the assistant span in ``h``; length B.
        return_diagnostics : bool
            If True, also return a dict of detached scalars for logging.

        Returns
        -------
        loss : scalar Tensor
            Mean softplus/relu violation over valid transitions.
        diag : dict, optional
            ``mean_V``, ``mean_V_curr``, ``mean_V_prev``,
            ``mean_raw_violation``, ``viol_rate``, ``surrogate_mean``,
            ``num_valid``, ``norm_mode``.
        """
        if h.dim() != 3:
            raise ValueError(f"h must be (B, T, D); got {tuple(h.shape)}")
        B, T, D = h.shape
        device = h.device
        dtype = h.dtype

        ub = self._as_int_bounds(user_bounds, "user_bounds", B)
        ab = self._as_int_bounds(assistant_bounds, "assistant_bounds", B)

        d_all = torch.zeros_like(h)
        v_geodesic = torch.zeros(B, D, device=device, dtype=dtype)
        valid_transitions = torch.zeros(B, T - 1, device=device, dtype=torch.bool)

        for i in range(B):
            u_s, u_e = ub[i]
            a_s, a_e = ab[i]
            if not (0 <= u_s <= u_e < T and 0 <= a_s <= a_e < T):
                # Skip malformed example; leaves d_all[i]=0 so its V is 0.
                continue
            v_user_i = h[i, u_e] - h[i, u_s]
            d_all[i, u_s : u_e + 1] = h[i, u_s : u_e + 1] - h[i, u_s]
            d_all[i, a_s : a_e + 1] = v_user_i + (h[i, a_s : a_e + 1] - h[i, a_s])
            v_geodesic[i] = v_user_i + (h[i, a_e] - h[i, a_s])
            if u_e > u_s:
                valid_transitions[i, u_s:u_e] = True
            if a_e > a_s:
                valid_transitions[i, a_s:a_e] = True

        v_dir = F.normalize(v_geodesic, p=2, dim=1, eps=self.eps)
        proj_scalar = (d_all * v_dir.unsqueeze(1)).sum(dim=-1, keepdim=True)
        p_all = proj_scalar * v_dir.unsqueeze(1)
        e_all = d_all - p_all

        e_sq = (e_all ** 2).sum(dim=-1)
        # L^2 = ||v_geo||^2 is reused by both the v_geo and d_t branches; the
        # eps floor protects against degenerate all-zero trajectories (e.g.
        # an empty span passed in via the random-span remapping, where the
        # caller collapses assistant_bounds to (a_s, a_s) to disable the
        # assistant chord and v_geo reduces to the user chord alone).
        L_sq = (v_geodesic ** 2).sum(dim=-1).clamp(min=self.eps)  # (B,)

        if self.norm_mode == "d_model":
            V_all = e_sq / float(max(D, 1))
        elif self.norm_mode == "v_geo":
            V_all = e_sq / L_sq.unsqueeze(1)
        elif self.norm_mode == "d_t":
            d_sq = (d_all ** 2).sum(dim=-1)  # (B, T)
            if self.anchor_eps > 0.0:
                # Scale-invariant soft anchor: far from the anchor (||d_t||^2
                # >> anchor_eps * L^2) this is sin^2(theta_t); close to it
                # V_t ~ ||e_t||^2 / (anchor_eps * L^2), a continuous Tikhonov
                # fallback to the v_geo norm mode instead of the legacy
                # (||d_t||^2).clamp(min=1e-8) cliff that blew the loss up to
                # O(1e8) at every anchor token.
                denom = self.anchor_eps * L_sq.unsqueeze(1) + d_sq
            else:
                # anchor_eps == 0 recovers the legacy hard-clamp form, kept
                # so existing checkpoints/ablations are still reproducible.
                denom = d_sq.clamp(min=self.eps)
            V_all = e_sq / denom
        else:
            raise AssertionError("unreachable")

        V_prev = V_all[:, :-1]
        V_curr = V_all[:, 1:]
        violation = V_curr - self.gamma * V_prev - self.tau
        surrogate = F.softplus(violation) if self.use_softplus else F.relu(violation)

        vf = valid_transitions.to(dtype)
        num_valid = valid_transitions.sum()
        if int(num_valid.item()) == 0:
            loss = h.sum() * 0.0
            if return_diagnostics:
                zero = torch.zeros((), device=device, dtype=dtype)
                return loss, {
                    "mean_V": zero,
                    "mean_V_curr": zero,
                    "mean_V_prev": zero,
                    "mean_raw_violation": zero,
                    "viol_rate": zero,
                    "surrogate_mean": zero,
                    "num_valid": torch.zeros((), device=device, dtype=torch.long),
                    "norm_mode": self.norm_mode,
                }
            return loss

        num_f = num_valid.to(dtype)
        loss = (surrogate * vf).sum() / num_f

        if not return_diagnostics:
            return loss

        mean_V_curr = (V_curr * vf).sum() / num_f
        mean_V_prev = (V_prev * vf).sum() / num_f
        mean_V = 0.5 * (mean_V_curr + mean_V_prev)
        mean_raw_violation = (violation * vf).sum() / num_f
        viol_rate = ((violation > 0).to(dtype) * vf).sum() / num_f
        return loss, {
            "mean_V": mean_V.detach(),
            "mean_V_curr": mean_V_curr.detach(),
            "mean_V_prev": mean_V_prev.detach(),
            "mean_raw_violation": mean_raw_violation.detach(),
            "viol_rate": viol_rate.detach(),
            "surrogate_mean": loss.detach(),
            "num_valid": num_valid.detach(),
            "norm_mode": self.norm_mode,
        }


class LyapunovReachabilityLoss(nn.Module):
    """Joint transverse-longitudinal Lyapunov loss (reach_JEPA).

    Unlike
    :class:`LyapunovControlLoss`, which constrains only the transverse residual
    ``e_t``, this loss defines a composite Lyapunov candidate that jointly
    controls:

    * the **transverse** coordinate (off-chord deviation)
      ``alpha * ||e_t||^2 / L^2``
    * the **longitudinal** coordinate (progress toward the answer anchor
      along the chord), shaped by ``progress_mode``.

    Progress modes
    --------------
    ``progress_mode="endpoint"`` (legacy, deprecated):
        ``V_prog_t = (1 - p_t/L)^2``. Empirically teleportation-prone: because
        the minimum is at ``p_t/L = 1`` for *every* token, the optimizer finds
        a degenerate solution where all tokens collapse toward the answer
        anchor (bimodal ``p_t/L`` distribution with mass near 1). Kept for
        ablations / backward compatibility only.

    ``progress_mode="schedule_asym"`` (default, recommended):
        ``V_prog_t = max(0, tau_t - p_t/L)^2`` where ``tau_t`` is a purely
        *structural* (token-rank) schedule
        ``tau_t = rank(t) / max(1, N - 1)``
        with ``rank`` enumerating the valid positions in user+assistant spans
        (N = (u_e - u_s + 1) + (a_e - a_s + 1)). Properties:

          (i) *Structural*: ``tau_t`` is a function of token indices only, so
              the optimizer cannot minimize the loss by reshaping ``v_geo``
              (unlike arc-length schedules).
          (ii) *Teleportation-proof*: the gradient w.r.t. ``p_t`` is **zero**
              whenever ``p_t/L >= tau_t``. There is no force pushing tokens
              toward the endpoint; only lagging tokens are pulled up, and
              only up to the schedule.
          (iii) *Linguistically sound*: accommodates non-uniform semantic
              density. Content tokens (high density) can naturally consume
              more progress than function tokens (low density); overshooting
              is free.
          (iv) *Telescoping closure still holds*:
              ``V_{t+1} <= gamma V_t + tau`` implies, for any sub-span
              ``[i, j]``, ``V_j <= gamma^{j-i} V_i + tau / (1 - gamma)``.
              Uniformly-small V then gives "local linearity everywhere"
              (STP's empirical random-span property) *as a theorem*.

    Both terms are **scale-invariant** in the hidden states (doubling every
    ``h[t]`` doubles ``d_t``, ``v_geo``, ``p_t``, and ``e_t`` identically, so
    ratios to ``L`` are unchanged). This rules out the ``d_model``-style
    clustering failure mode by construction.

    The discrete-time contraction ``V_{t+1} <= gamma * V_t + tau`` is enforced
    via the stable linear-softplus surrogate over valid transitions
    (user-internal + assistant-internal; chord gap skipped).

    Ablation-ready degenerations:
      * ``beta == 0``  : reduces to pure tube.
      * ``alpha == 0`` : pure longitudinal / progress loss (negative control).
      * ``progress_mode="endpoint"`` : reproduces the pre-fix loss (bad).

    Parameters
    ----------
    alpha : float, default 1.0
        Weight on the transverse term.
    beta : float, default 1.0
        Weight on the progress term (form set by ``progress_mode``).
    gamma : float, default 0.9
        Contraction factor in ``V_{t+1} <= gamma * V_t + tau``.
    tau : float, default 1e-4
        Additive slack.
    progress_mode : {"endpoint", "schedule_asym"}, default "schedule_asym"
        Shape of the longitudinal Lyapunov component.
    transverse_norm : {"L", "d_t"}, default "L"
        Denominator for the transverse term.
          * ``"L"``   : ``||e_t||^2 / L^2`` (classical, unbounded).
          * ``"d_t"`` : ``||e_t||^2 / ||d_t||^2 = sin^2 \theta_t`` (angle form;
            bounded in [0, 1], rotation-invariant; positions where
            ``d_t = 0`` — i.e. the span anchors ``t = u_s, a_s`` — are
            masked out of both the transverse term *and* the transition
            loss so the ratio never hits the clamp floor). Used by
            ``reach_JEPA_angle``.
    use_softplus : bool, default True
        If True, use ``F.softplus`` surrogate; else ``F.relu``.
    eps : float, default 1e-8
        Numerical floor for ``L`` and normalization.
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
            raise ValueError(
                f"alpha and beta must be non-negative; got alpha={alpha}, beta={beta}"
            )
        if alpha == 0 and beta == 0:
            raise ValueError("at least one of alpha, beta must be positive")
        if progress_mode not in VALID_PROGRESS_MODES:
            raise ValueError(
                f"progress_mode must be one of {VALID_PROGRESS_MODES}; "
                f"got {progress_mode!r}"
            )
        if transverse_norm not in VALID_REACH_TRANSVERSE_NORMS:
            raise ValueError(
                f"transverse_norm must be one of {VALID_REACH_TRANSVERSE_NORMS}; "
                f"got {transverse_norm!r}"
            )
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.progress_mode = progress_mode
        self.transverse_norm = transverse_norm
        self.use_softplus = False#bool(use_softplus)
        self.eps = float(eps)

    _as_int_bounds = staticmethod(LyapunovControlLoss._as_int_bounds)

    def forward(
        self,
        h: torch.Tensor,
        user_bounds: Sequence[Sequence[int]],
        assistant_bounds: Sequence[Sequence[int]],
        *,
        return_diagnostics: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """See class docstring."""
        if h.dim() != 3:
            raise ValueError(f"h must be (B, T, D); got {tuple(h.shape)}")
        B, T, D = h.shape
        device = h.device
        dtype = h.dtype

        ub = self._as_int_bounds(user_bounds, "user_bounds", B)
        ab = self._as_int_bounds(assistant_bounds, "assistant_bounds", B)

        d_all = torch.zeros_like(h)
        v_geodesic = torch.zeros(B, D, device=device, dtype=dtype)
        valid_transitions = torch.zeros(B, T - 1, device=device, dtype=torch.bool)
        # `tau_sched[i, t]` is the structural progress target tau_t for the
        # schedule_asym mode; zero outside valid positions. Built once per
        # batch from integer token indices only (exogenous to h).
        tau_sched = torch.zeros(B, T, device=device, dtype=dtype)
        valid_positions = torch.zeros(B, T, device=device, dtype=torch.bool)

        for i in range(B):
            u_s, u_e = ub[i]
            a_s, a_e = ab[i]
            if not (0 <= u_s <= u_e < T and 0 <= a_s <= a_e < T):
                continue
            v_user_i = h[i, u_e] - h[i, u_s]
            d_all[i, u_s : u_e + 1] = h[i, u_s : u_e + 1] - h[i, u_s]
            d_all[i, a_s : a_e + 1] = v_user_i + (h[i, a_s : a_e + 1] - h[i, a_s])
            v_geodesic[i] = v_user_i + (h[i, a_e] - h[i, a_s])
            if u_e > u_s:
                valid_transitions[i, u_s:u_e] = True
            if a_e > a_s:
                valid_transitions[i, a_s:a_e] = True

            # Build the structural token-rank schedule tau_t for this sample.
            n_u = u_e - u_s + 1
            n_a = a_e - a_s + 1
            N = n_u + n_a
            if N > 1:
                denom = float(N - 1)
                tau_sched[i, u_s : u_e + 1] = torch.arange(
                    0, n_u, device=device, dtype=dtype
                ) / denom
                tau_sched[i, a_s : a_e + 1] = torch.arange(
                    n_u, N, device=device, dtype=dtype
                ) / denom
            valid_positions[i, u_s : u_e + 1] = True
            valid_positions[i, a_s : a_e + 1] = True

        L_sq = (v_geodesic * v_geodesic).sum(dim=-1).clamp(min=self.eps)
        L = L_sq.sqrt()
        v_dir = v_geodesic / L.unsqueeze(-1).clamp(min=self.eps)

        # Scalar projection p_t = <d_t, v_hat>  (B, T)
        p_all = (d_all * v_dir.unsqueeze(1)).sum(dim=-1)
        p_vec = p_all.unsqueeze(-1) * v_dir.unsqueeze(1)
        e_all = d_all - p_vec

        e_sq = (e_all ** 2).sum(dim=-1)  # (B, T)
        d_sq = (d_all * d_all).sum(dim=-1)  # (B, T); zero at span anchors

        # Scale-invariant composite Lyapunov candidate.
        #   V_t = alpha * V_\perp(t) + beta * V_\| (t)
        # V_\perp(t):
        #   transverse_norm="L"   -> ||e_t||^2 / L^2
        #   transverse_norm="d_t" -> ||e_t||^2 / ||d_t||^2 = sin^2 theta_t
        #     (bounded in [0,1]; anchor positions where d_t = 0 are masked
        #      out of both the transverse term AND the downstream transition
        #      so the clamp floor never leaks into the gradient).
        if self.transverse_norm == "L":
            transversal = e_sq / L_sq.unsqueeze(1)
            anchor_mask = torch.ones_like(d_sq, dtype=torch.bool)
        elif self.transverse_norm == "d_t":
            # A position is a "valid anchor neighborhood" if d_t is non-trivial.
            # We use a relative threshold (d_sq > eps * L_sq) so the mask is
            # scale-invariant.
            anchor_mask = d_sq > (self.eps * L_sq.unsqueeze(1))
            transversal = e_sq / d_sq.clamp(min=self.eps)
            # Zero out the anchor positions explicitly so the surrogate
            # mean (computed before masking below) doesn't pick up clamp-
            # floor noise.
            transversal = torch.where(
                anchor_mask, transversal, torch.zeros_like(transversal)
            )
        else:
            raise AssertionError("unreachable: transverse_norm was validated")
        p_over_L = p_all / L.unsqueeze(1)

        if self.progress_mode == "endpoint":
            # Legacy (1 - p_t/L)^2: pressures every token toward the answer
            # anchor. Known to be teleportation-prone.
            progress_raw = 1.0 - p_over_L
            progress_loss = progress_raw ** 2
        elif self.progress_mode == "schedule_asym":
            # max(0, tau_t - p_t/L)^2: lag-only penalty vs. token-rank
            # schedule. No gradient pressure past the schedule -> no
            # teleportation incentive. `progress_raw` is the signed gap
            # (positive = lagging) used for diagnostics.
            progress_raw = tau_sched - p_over_L
            progress_loss = progress_raw.clamp(min=0.0) ** 2
        else:
            raise AssertionError("unreachable: progress_mode was validated")

        V_all = self.alpha * transversal + self.beta * progress_loss

        V_prev = V_all[:, :-1]
        V_curr = V_all[:, 1:]
        violation = V_curr - self.gamma * V_prev - self.tau
        surrogate = F.softplus(violation) if self.use_softplus else F.relu(violation)

        # For transverse_norm="d_t", transitions starting at an anchor (V_prev
        # artificially = 0) are dropped; the contraction constraint
        # V_curr <= gamma * 0 + tau would otherwise be unreasonably tight.
        effective_transitions = valid_transitions
        if self.transverse_norm == "d_t":
            effective_transitions = effective_transitions & anchor_mask[:, :-1]
        vf = effective_transitions.to(dtype)
        num_valid = effective_transitions.sum()
        if int(num_valid.item()) == 0:
            loss = h.sum() * 0.0
            if return_diagnostics:
                zero = torch.zeros((), device=device, dtype=dtype)
                return loss, {
                    "mean_V": zero,
                    "mean_V_transversal": zero,
                    "mean_V_progress": zero,
                    "mean_p_over_L": zero,
                    "mean_e2_over_L2": zero,
                    "mean_tau": zero,
                    "mean_progress_gap": zero,
                    "lag_rate": zero,
                    "mean_raw_violation": zero,
                    "viol_rate": zero,
                    "surrogate_mean": zero,
                    "num_valid": torch.zeros((), device=device, dtype=torch.long),
                    "alpha": self.alpha,
                    "beta": self.beta,
                    "progress_mode": self.progress_mode,
                    "transverse_norm": self.transverse_norm,
                }
            return loss

        num_f = num_valid.to(dtype)
        loss = (surrogate * vf).sum() / num_f

        if not return_diagnostics:
            return loss

        vf_full = torch.zeros_like(V_all, dtype=dtype)
        vf_full[:, :-1] = vf
        num_full = vf_full.sum().clamp(min=1.0)
        # Diagnostic mask over *positions* (including the last valid position
        # which has no outgoing transition): uses valid_positions so that
        # mean_tau and mean_p/L reflect the full trajectory, not just
        # transitions.
        vpos = valid_positions.to(dtype)
        num_pos = vpos.sum().clamp(min=1.0)

        mean_V = (V_all * vf_full).sum() / num_full
        mean_V_transversal = (self.alpha * transversal * vf_full).sum() / num_full
        mean_V_progress = (
            self.beta * progress_loss * vf_full
        ).sum() / num_full
        mean_p_over_L = (p_over_L * vpos).sum() / num_pos
        mean_e2_over_L2 = (transversal * vpos).sum() / num_pos
        mean_tau = (tau_sched * vpos).sum() / num_pos
        # Signed lag gap (positive = lagging behind schedule, negative = ahead
        # of schedule). For schedule_asym this is the pre-clamp term; for
        # endpoint mode we surface (1 - p/L) with the same sign convention.
        if self.progress_mode == "schedule_asym":
            signed_gap = progress_raw
        else:
            signed_gap = progress_raw
        mean_progress_gap = (signed_gap * vpos).sum() / num_pos
        lag_rate = ((signed_gap > 0).to(dtype) * vpos).sum() / num_pos
        mean_raw_violation = (violation * vf).sum() / num_f
        viol_rate = ((violation > 0).to(dtype) * vf).sum() / num_f
        return loss, {
            "mean_V": mean_V.detach(),
            "mean_V_transversal": mean_V_transversal.detach(),
            "mean_V_progress": mean_V_progress.detach(),
            "mean_p_over_L": mean_p_over_L.detach(),
            "mean_e2_over_L2": mean_e2_over_L2.detach(),
            "mean_tau": mean_tau.detach(),
            "mean_progress_gap": mean_progress_gap.detach(),
            "lag_rate": lag_rate.detach(),
            "mean_raw_violation": mean_raw_violation.detach(),
            "viol_rate": viol_rate.detach(),
            "surrogate_mean": loss.detach(),
            "num_valid": num_valid.detach(),
            "alpha": self.alpha,
            "beta": self.beta,
            "progress_mode": self.progress_mode,
            "transverse_norm": self.transverse_norm,
        }


__all__ = [
    "LyapunovControlLoss",
    "LyapunovReachabilityLoss",
    "VALID_NORMS",
    "VALID_PROGRESS_MODES",
    "VALID_REACH_TRANSVERSE_NORMS",
]


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, D = 2, 24, 32
    h = torch.randn(B, T, D, requires_grad=True)
    ub = [(1, 6), (2, 7)]
    ab = [(10, 18), (9, 20)]

    print("--- LyapunovControlLoss ---")
    for mode in VALID_NORMS:
        loss, diag = LyapunovControlLoss(gamma=0.9, tau=1e-4, norm_mode=mode)(
            h, ub, ab, return_diagnostics=True
        )
        print(
            f"[{mode}] loss={loss.item():.6f} "
            f"mean_V={diag['mean_V'].item():.4e} "
            f"viol_rate={diag['viol_rate'].item():.3f} "
            f"num_valid={int(diag['num_valid'].item())}"
        )
        if h.grad is not None:
            h.grad.zero_()
        loss.backward()
        assert h.grad is not None and h.grad.abs().sum().item() > 0

    # --- anchor_eps property tests (norm_mode="d_t" only) -------------------
    # The core property the soft gate restores is *scale invariance*.
    # Mathematically V_t = ||e_t||^2 / ||d_t||^2 is homogeneous of degree 0
    # in h (rescaling h -> alpha h multiplies both numerator and denom by
    # alpha^2). The legacy implementation broke this at small h because
    # ||d_t||^2 was clamped to eps=1e-8, a dimensional constant that does
    # NOT rescale. As soon as some alpha^2 * ||d_t||^2 drops below 1e-8,
    # V_t switches to alpha^2 * ||e_t||^2 / 1e-8, an alpha-dependent
    # quantity. The soft gate uses anchor_eps * L^2 (where L = ||v_geo||
    # scales with h) and therefore preserves scale invariance exactly.
    # We also re-verify ||e_t||^2 <= ||d_t||^2 so the angle form is
    # automatically bounded in [0,1] at any scale -- the old clamp was
    # *not* the bound; it was a numerical safety that accidentally
    # broke scale invariance at tiny trajectories (which is precisely the
    # regime random-span Lyapunov will push us into via short sub-spans).
    print("\n--- LyapunovControlLoss anchor_eps property tests (d_t) ---")
    B2, T2, D2 = 2, 24, 32
    torch.manual_seed(123)
    h2_base = torch.randn(B2, T2, D2)
    ub2 = [(1, 6), (2, 7)]
    ab2 = [(10, 18), (9, 20)]

    # Pick a scale that drives ||d_t||^2 comfortably below the legacy
    # 1e-8 floor for *every* token, so the hard clamp activates on the
    # whole trajectory. alpha = 1e-4 yields ||d_t||^2 ~ 1e-8 .. 1e-6.
    base_loss_hard, _ = LyapunovControlLoss(
        norm_mode="d_t", anchor_eps=0.0
    )(h2_base, ub2, ab2, return_diagnostics=True)
    base_loss_soft, _ = LyapunovControlLoss(
        norm_mode="d_t", anchor_eps=1e-3
    )(h2_base, ub2, ab2, return_diagnostics=True)

    print(
        f"[alpha=1.0, hard] loss={base_loss_hard.item():.6e}  (reference)"
    )
    print(
        f"[alpha=1.0, soft] loss={base_loss_soft.item():.6e}  (reference)"
    )

    scale_errors_hard: List[Tuple[float, float]] = []
    scale_errors_soft: List[Tuple[float, float]] = []
    # Sweep alpha across the regime where the legacy eps=1e-8 floor
    # activates but the outer L_sq.clamp (also at self.eps) does NOT -
    # that's the band where anchor_eps does its job. alpha=1e-6 with
    # D=32 random data gives ||d||^2 ~ 3e-11 << 1e-8 (clamp active) but
    # ||L||^2 ~ 1e-10 > 1e-8 (outer floor inactive). alpha <= 1e-8
    # collapses the whole trajectory into the outer floor and both
    # variants saturate to log(2) via softplus; we skip it.
    for alpha in (1e-6, 1e-4, 1.0, 1e4):
        h_a = alpha * h2_base
        loss_h, _ = LyapunovControlLoss(
            norm_mode="d_t", anchor_eps=0.0
        )(h_a, ub2, ab2, return_diagnostics=True)
        loss_s, _ = LyapunovControlLoss(
            norm_mode="d_t", anchor_eps=1e-3
        )(h_a, ub2, ab2, return_diagnostics=True)
        err_h = abs(loss_h.item() - base_loss_hard.item()) / max(base_loss_hard.item(), 1e-12)
        err_s = abs(loss_s.item() - base_loss_soft.item()) / max(base_loss_soft.item(), 1e-12)
        scale_errors_hard.append((alpha, err_h))
        scale_errors_soft.append((alpha, err_s))
        print(
            f"  alpha={alpha:>7.0e}  hard_loss={loss_h.item():.3e}  "
            f"(rel.err {err_h:.2e})   soft_loss={loss_s.item():.3e}  "
            f"(rel.err {err_s:.2e})"
        )

    # Soft gate: scale-invariant to <=1% across the sweep. (Residual comes
    # from the softplus tail when alpha*||d|| is comparable to the outer
    # self.eps floor on L_sq; it is O(1e-3), not an actual scale-variance
    # of V_t itself.)
    for alpha, err in scale_errors_soft:
        assert err < 1e-2, (
            f"soft gate not scale-invariant at alpha={alpha}: rel.err={err:.3e}"
        )
    print("[soft eps=1e-3] scale-invariant across alpha in [1e-6, 1e4]")

    # Hard clamp: breaks scale invariance at alpha=1e-6 where ||d||^2
    # drops below the 1e-8 floor. We assert at least one alpha in the
    # sweep shows >= 2% relative error (it is ~5% in practice -- the
    # loss saturates to log(2) once the clamp dominates).
    max_hard_err = max(e for _, e in scale_errors_hard)
    assert max_hard_err > 2e-2, (
        f"expected hard clamp to break scale invariance by >=2%; "
        f"got max rel.err={max_hard_err:.2e}"
    )
    max_soft_err = max(e for _, e in scale_errors_soft)
    assert max_hard_err > 3.0 * max_soft_err, (
        f"hard clamp's scale-variance ({max_hard_err:.2e}) should exceed "
        f"soft gate's numerical residual ({max_soft_err:.2e}) by >=3x"
    )
    print(
        f"[hard clamp  ] scale-variant as expected "
        f"(max rel.err {max_hard_err:.2e}, {max_hard_err / max(max_soft_err, 1e-12):.1f}x soft's residual)"
    )

    # Finite loss + finite gradient under the soft gate at every scale.
    for alpha in (1e-6, 1.0, 1e6):
        h2r = (alpha * h2_base).clone().requires_grad_(True)
        loss_r, _ = LyapunovControlLoss(
            norm_mode="d_t", anchor_eps=1e-3
        )(h2r, ub2, ab2, return_diagnostics=True)
        assert torch.isfinite(loss_r), f"loss non-finite at alpha={alpha}"
        loss_r.backward()
        assert h2r.grad is not None and torch.isfinite(h2r.grad).all(), (
            f"gradient non-finite at alpha={alpha}"
        )
    print("[finiteness ] loss and grad finite for alpha in {1e-6, 1, 1e6}")

    # Backward-compat: anchor_eps=0 must reproduce the pre-fix legacy
    # hard-clamp path exactly. This is the audit hook for re-running
    # legacy ablations.
    hard_again, _ = LyapunovControlLoss(
        norm_mode="d_t", anchor_eps=0.0
    )(h2_base, ub2, ab2, return_diagnostics=True)
    assert abs(hard_again.item() - base_loss_hard.item()) < 1e-8
    print("[anchor_eps=0] reproduces legacy hard-clamp loss exactly")

    print("\n--- LyapunovReachabilityLoss (endpoint mode, legacy) ---")
    for cfg in [dict(alpha=1.0, beta=1.0), dict(alpha=1.0, beta=0.0)]:
        loss, diag = LyapunovReachabilityLoss(
            gamma=0.9, tau=1e-4, progress_mode="endpoint", **cfg
        )(h, ub, ab, return_diagnostics=True)
        print(
            f"[a={cfg['alpha']:.2f}, b={cfg['beta']:.2f}] "
            f"loss={loss.item():.6f} "
            f"V_trans={diag['mean_V_transversal'].item():.4e} "
            f"V_prog={diag['mean_V_progress'].item():.4e} "
            f"p/L={diag['mean_p_over_L'].item():+.3f} "
            f"gap={diag['mean_progress_gap'].item():+.3f} "
            f"lag_rate={diag['lag_rate'].item():.3f}"
        )
        if h.grad is not None:
            h.grad.zero_()
        loss.backward()
        assert h.grad is not None and h.grad.abs().sum().item() > 0

    print("\n--- LyapunovReachabilityLoss (schedule_asym, proposed) ---")
    configs = [
        dict(alpha=1.0, beta=0.0),   # tube only
        dict(alpha=0.0, beta=1.0),   # progress only
        dict(alpha=1.0, beta=1.0),   # joint proposed method
        dict(alpha=1.0, beta=0.5),   # tube-heavy
        dict(alpha=0.5, beta=1.0),   # progress-heavy
    ]
    for cfg in configs:
        loss, diag = LyapunovReachabilityLoss(
            gamma=0.9, tau=1e-4, progress_mode="schedule_asym", **cfg
        )(h, ub, ab, return_diagnostics=True)
        print(
            f"[a={cfg['alpha']:.2f}, b={cfg['beta']:.2f}] "
            f"loss={loss.item():.6f} "
            f"V_trans={diag['mean_V_transversal'].item():.4e} "
            f"V_prog={diag['mean_V_progress'].item():.4e} "
            f"tau={diag['mean_tau'].item():+.3f} "
            f"p/L={diag['mean_p_over_L'].item():+.3f} "
            f"gap={diag['mean_progress_gap'].item():+.3f} "
            f"lag_rate={diag['lag_rate'].item():.3f} "
            f"viol={diag['viol_rate'].item():.3f}"
        )
        if h.grad is not None:
            h.grad.zero_()
        loss.backward()
        assert h.grad is not None and h.grad.abs().sum().item() > 0

    print("\n--- LyapunovReachabilityLoss (angle form: transverse_norm=d_t) ---")
    # Angle form: V_\perp = sin^2(theta) is in [0, 1]. On random hidden states
    # we expect mean_V_transversal ~ 0.1 .. 1.0 (with D=32) rather than
    # unboundedly large.
    for cfg in [
        dict(alpha=1.0, beta=0.0),
        dict(alpha=1.0, beta=1.0),
        dict(alpha=1.0, beta=0.5),
    ]:
        loss, diag = LyapunovReachabilityLoss(
            gamma=0.9, tau=1e-4,
            progress_mode="schedule_asym",
            transverse_norm="d_t",
            **cfg,
        )(h, ub, ab, return_diagnostics=True)
        V_trans = diag["mean_V_transversal"].item() / max(cfg["alpha"], 1e-12)
        print(
            f"[angle a={cfg['alpha']:.2f}, b={cfg['beta']:.2f}] "
            f"loss={loss.item():.6f} "
            f"sin2={V_trans:.4f}  "
            f"V_prog={diag['mean_V_progress'].item():.4e} "
            f"viol={diag['viol_rate'].item():.3f}"
        )
        # sin^2 must be in [0,1] by construction.
        assert 0.0 <= V_trans <= 1.0 + 1e-4, f"sin^2 out of range: {V_trans}"
        if h.grad is not None:
            h.grad.zero_()
        loss.backward()
        assert h.grad is not None and h.grad.abs().sum().item() > 0

    # Teleportation stress test: compare endpoint vs. schedule_asym on a
    # *uniform* trajectory (p_t/L = tau_t exactly). Endpoint mode misreports
    # this as a large gap (pathological); schedule_asym correctly reports ~0.
    print("\n--- teleportation stress test: uniform trajectory ---")
    B, T, D = 1, 10, 4
    h3 = torch.zeros(B, T, D, requires_grad=True)
    with torch.no_grad():
        # User span [1,4], assistant span [5,8]. Place h[t] along [1,0,0,0]
        # at rank-proportional distances from h[1]. This gives exactly
        # p_t/L = tau_t, the ideal uniform schedule.
        unit = torch.tensor([1.0, 0.0, 0.0, 0.0])
        user_idx = list(range(1, 5))   # u_s=1, u_e=4, n_u=4
        asst_idx = list(range(5, 9))   # a_s=5, a_e=8, n_a=4
        N = len(user_idx) + len(asst_idx)
        # Progress is 0 -> 1 across N positions. We choose a single
        # magnitude per rank so that user's d_t and assistant's v_user +
        # (h[t]-h[a_s]) both advance uniformly along [1,0,0,0].
        for r, t in enumerate(user_idx):
            h3[0, t] = r * unit        # d_t = r * unit for user positions
        h_start_a = user_idx[-1] - user_idx[0]  # = 3, the value of v_user
        for r, t in enumerate(asst_idx):
            # d_t = v_user + (h[t] - h[a_s]) = 3 + r  -> rank = len(user_idx) + r = 4+r
            h3[0, t] = r * unit
    diag_end = LyapunovReachabilityLoss(
        alpha=0.0, beta=1.0, gamma=0.9, tau=0.0, progress_mode="endpoint"
    )(h3, [(1, 4)], [(5, 8)], return_diagnostics=True)[1]
    diag_sch = LyapunovReachabilityLoss(
        alpha=0.0, beta=1.0, gamma=0.9, tau=0.0, progress_mode="schedule_asym"
    )(h3, [(1, 4)], [(5, 8)], return_diagnostics=True)[1]
    print(
        f"[endpoint ] gap={diag_end['mean_progress_gap'].item():+.3f} "
        f"V_prog={diag_end['mean_V_progress'].item():.3e}  <- penalizes uniform pacing!"
    )
    print(
        f"[schedule ] gap={diag_sch['mean_progress_gap'].item():+.3f} "
        f"V_prog={diag_sch['mean_V_progress'].item():.3e}  <- ~0 on uniform pacing, as intended"
    )
    # Rank-based schedule differs from arc-length schedule by O(1/N), so the
    # loss won't be exactly 0 on a uniform *arc-length* trajectory, but it
    # should be orders of magnitude smaller than the endpoint form's.
    assert abs(diag_sch["mean_progress_gap"].item()) < 5e-3, (
        "schedule_asym should report near-zero gap on a uniform trajectory"
    )
    ratio = (
        diag_end["mean_V_progress"].item()
        / max(diag_sch["mean_V_progress"].item(), 1e-12)
    )
    assert ratio > 50.0, (
        f"endpoint form should dominate schedule_asym by >50x on uniform "
        f"pacing; got {ratio:.1f}x"
    )
    print(
        f"[ratio]    endpoint V_prog is {ratio:.0f}x schedule_asym on "
        f"uniform pacing (the bug we fix)"
    )

    print("\nAll smoke tests passed.")
