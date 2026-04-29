"""Implicit task-shift detector for FAME (Approach 2 in the proposal).

Design improvements over the proposal's bare reward-model detector
-----------------------------------------------------------------
The proposal suggests a *single* reward model g_v(phi, a) -> r and uses its
normalised prediction error as a drift signal.  That captures reward-function
shifts but is blind to changes in the observation/dynamics distribution that
leave the reward signature unchanged (e.g. Rotated/Color-Swap shifts in the
Reactive Exploration paper).  Worse, a single scalar-regression error is
extremely noisy step-to-step, so a simple windowed mean has poor power.

We therefore replace the single reward head with a compact, two-head
**Task-Signature Network (TSN)** that re-uses the fast learner's penultimate
feature `phi` as encoder (no extra encoder, no extra env interactions):

    body(phi_t, a_onehot) -> h_t                       (shared MLP, 64-d)
       |-- reward head      h_t -> r_hat_{t+1}         (1-d, MSE)
       |-- forward-dyn head h_t -> phi_hat_{t+1}       (latent_dim-d, MSE)

This is (a) exactly as heavy as a bare reward model plus a tiny linear head
and (b) captures both reward-function shifts (reward head) and
observation/dynamics shifts (forward-dynamics head), following the ICM idea
of Reactive Exploration without its intrinsic-reward feedback loop.

Per-step shift score
--------------------
We standardise each head's error online via Welford's running
(mean, variance) estimator, then take the positive z-score:

    z^r_t = max(0, (e^r_t - mu_r) / (sigma_r + eps))
    z^d_t = max(0, (e^d_t - mu_d) / (sigma_d + eps))
    S_t  = z^r_t + z^d_t

The positive-only clip prevents low-error windows from *reducing* the drift
signal (we only care about upward deviations).  The Welford standardisation
makes the score scale-invariant across tasks with different reward magnitudes
and latent norms.

Window-level drift statistic
----------------------------
Rather than the proposal's ad-hoc  D = (mu_cur - mu_ref)/(sigma_ref + eps),
we use a **Welch's one-sided t-test** between the two rolling windows:

    t = (mu_cur - mu_ref) / sqrt(sigma_cur^2 / L_D + sigma_ref^2 / L_D)
    dof = Welch-Satterthwaite
    p   = 1 - CDF(t, dof)

Welch's t is principled, handles unequal variances, and returns a true
p-value we can threshold at a desired false-positive rate alpha.  It is
closer in spirit to the SWOKS KS-test than the proposal's ratio statistic,
making both detectors directly comparable.

Fallbacks / guards mirror SWOKS to keep the downstream FAME trigger sane:
a warmup period (no detection allowed), a stable_phase guard after each
fire, a detection_interval to avoid per-step testing, an optional max_wait
fallback for false-negative recovery.

Interface (mirrors `SwoksDetector`)
-----------------------------------
    det = ImplicitDetector(latent_dim, num_actions, device='cpu', ...)
    fired = det.step(phi_t, action_t, reward_t, next_phi_t)
    det.stats(); det.last_pval; det.last_score

The detector owns a small torch module and its own optimiser; it trains
asynchronously on a FIFO replay of the last few L_D transitions so the
policy's gradient updates are unaffected.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats as _scistats


# ----------------------------------------------------------------------
# Task-Signature Network
# ----------------------------------------------------------------------
class TaskSignatureNet(nn.Module):
    """Dual-head: (phi, a) -> (r_hat, phi_hat_next).

    Intentionally small (~ few tens of thousands of params): the detector
    must be cheap relative to the policy network.
    """

    def __init__(self, latent_dim: int, num_actions: int, hidden: int = 64):
        super().__init__()
        self.num_actions = num_actions
        self.body = nn.Sequential(
            nn.Linear(latent_dim + num_actions, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.reward_head = nn.Linear(hidden, 1)
        self.dyn_head = nn.Linear(hidden, latent_dim)

    def forward(self, phi: torch.Tensor, a_onehot: torch.Tensor):
        h = self.body(torch.cat([phi, a_onehot], dim=-1))
        return self.reward_head(h).squeeze(-1), self.dyn_head(h)


# ----------------------------------------------------------------------
# Welford online (mean, variance) estimator
# ----------------------------------------------------------------------
class Welford:
    """Numerically stable running mean/variance."""

    __slots__ = ("n", "mean", "M2")

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (x - self.mean)

    @property
    def var(self) -> float:
        return self.M2 / self.n if self.n > 1 else 1.0

    @property
    def std(self) -> float:
        return math.sqrt(max(self.var, 1e-12))


# ----------------------------------------------------------------------
# Implicit detector
# ----------------------------------------------------------------------
class ImplicitDetector:
    """Online implicit task-shift detector via dual-head prediction errors.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the encoder feature phi.
    num_actions : int
        Discrete action space cardinality (for one-hot embedding).
    L_D : int
        Window size over which statistics are compared.
    alpha : float
        Desired false-positive rate (upper p-value threshold).
    stable_phase, warmup, detection_interval, max_wait : int
        Same semantics as in `SwoksDetector`.
    lr : float
        Adam learning rate for the TSN.
    replay_capacity : int
        FIFO replay buffer capacity for training the TSN.
    update_every : int
        Train the TSN once every this many env steps.
    batch_size : int
        Minibatch size for TSN training.
    score_batch_size : int or None
        Number of transitions to accumulate before doing a single batched
        forward pass to compute their errors.  Defaults to ``update_every``.
        Increasing this reduces PyTorch dispatch overhead at the cost of
        slightly coarser (but statistically identical) score granularity.
    device : str
        torch device (cpu/cuda).
    seed : int
    """

    def __init__(
        self,
        latent_dim: int,
        num_actions: int,
        L_D: int = 1200,
        alpha: float = 1e-3,
        stable_phase: int = 36000,
        warmup: int = 5000,
        detection_interval: int = 240,
        max_wait: int = 0,
        lr: float = 1e-4,
        replay_capacity: Optional[int] = None,
        update_every: int = 16,
        batch_size: int = 64,
        score_batch_size: Optional[int] = None,
        device: str = "cpu",
        seed: int = 0,
    ):
        self.latent_dim = int(latent_dim)
        self.num_actions = int(num_actions)
        self.L_D = int(L_D)
        self.alpha = float(alpha)
        self.stable_phase = int(stable_phase)
        self.warmup = int(warmup)
        self.detection_interval = int(detection_interval)
        self.max_wait = int(max_wait)
        self.lr = float(lr)
        self.update_every = int(update_every)
        self.batch_size = int(batch_size)
        # Batched scoring: accumulate this many transitions before a single
        # batched TSN forward pass.  Defaults to update_every so that the
        # scoring flush and training step happen at the same cadence,
        # avoiding redundant forward passes.
        self.score_batch_size = int(score_batch_size) if score_batch_size \
            else int(update_every)
        self.device = torch.device(device)

        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)

        self.net = TaskSignatureNet(latent_dim, num_actions).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        cap = int(replay_capacity) if replay_capacity else 4 * self.L_D
        self._replay = deque(maxlen=cap)

        # Buffer for batched error scoring.  Transitions accumulate here
        # until score_batch_size entries are ready, then one batched forward
        # pass computes all their errors in one call (replacing the old
        # per-step single-sample _compute_errors approach).
        self._score_buf: list = []

        # Windows of per-step task-signature scores S_t.
        self._scores = deque(maxlen=2 * self.L_D)

        # Running stats for standardising each head's error.
        self._w_r = Welford()
        self._w_d = Welford()

        self.ts = 0
        self.last_shift_step = 0
        self.detections = []
        self.last_pval = 1.0
        self.last_score = 0.0
        self.last_err_r = 0.0
        self.last_err_d = 0.0
        # Fire-time snapshot of detection statistics, preserved across
        # `_on_detected()` so callers can log them after the fact.
        self.last_fire_info = {}
        # Warmup-end re-normalisation: during warmup the TSN is still
        # converging, so its errors are dominated by training-phase noise
        # that biases Welford upwards permanently.  We reset the running
        # stats + the score history exactly at `warmup` so the detector
        # starts from a clean, calibrated baseline when it first becomes
        # allowed to fire.
        self._warmup_reset_done = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def step(self, phi, action, reward, next_phi) -> bool:
        """Consume one transition and maybe fire.

        Parameters
        ----------
        phi : np.ndarray (latent_dim,)
            Penultimate feature at state s_t.
        action : int
            Action a_t.
        reward : float
            r_{t+1}.
        next_phi : np.ndarray (latent_dim,)
            Penultimate feature at state s_{t+1}.

        Returns
        -------
        bool
            True iff a shift was newly detected at this step.
        """
        self.ts += 1

        phi = np.asarray(phi, dtype=np.float32).ravel()
        next_phi = np.asarray(next_phi, dtype=np.float32).ravel()
        self._replay.append((phi, int(action), float(reward), next_phi))

        # Batched error scoring: buffer the transition and flush once we
        # have score_batch_size entries.  This replaces the old per-step
        # single-sample _compute_errors() call, reducing PyTorch dispatch
        # overhead by score_batch_size x.
        self._score_buf.append((phi, int(action), float(reward), next_phi))
        if len(self._score_buf) >= self.score_batch_size:
            self._flush_score_buf()

        # Online training of the TSN on fresh transitions.
        if self.ts % self.update_every == 0 and len(self._replay) >= self.batch_size:
            self._train_step()

        # Gating (same guard structure as SwoksDetector).
        if self.ts < self.warmup:
            return False
        if self.ts - self.last_shift_step < self.stable_phase:
            return False
        # One-shot reset at the edge between the TSN's "settling" phase and
        # its first eligible firing window.  Welford accumulated during
        # warmup / stable_phase is biased by convergence noise, so we drop
        # it and re-calibrate from the first fully-stable sample onward.
        if not self._warmup_reset_done:
            self._w_r = Welford()
            self._w_d = Welford()
            self._scores.clear()
            self._score_buf.clear()   # discard any pre-flush buffered scores
            self._warmup_reset_done = True
            return False
        if self.ts % self.detection_interval != 0:
            return False
        if len(self._scores) < 2 * self.L_D:
            return False

        # Welch's one-sided t-test on the two halves.
        arr = np.fromiter(self._scores, dtype=np.float64)
        ref = arr[: self.L_D]
        cur = arr[-self.L_D:]
        t_stat, pval = _welch_one_sided(cur, ref)
        self.last_pval = float(pval)

        fired = pval < self.alpha
        if not fired and self.max_wait > 0:
            quiet = self.ts - self.last_shift_step
            if quiet > self.max_wait and cur.mean() > 1.5 * ref.mean():
                fired = True

        if fired:
            # Snapshot live stats before `_on_detected()` wipes them, so
            # callers can introspect *why* we fired.
            self.last_fire_info = {
                "step": self.ts,
                "pval": self.last_pval,
                "score": self.last_score,
                "err_r": self.last_err_r,
                "err_d": self.last_err_d,
                "cur_mean": float(cur.mean()),
                "ref_mean": float(ref.mean()),
                "t_stat": float(t_stat),
            }
            self._on_detected()
        return fired

    def reset_after_boundary(self) -> None:
        self._on_detected()

    def stats(self) -> dict:
        return {
            "ts": self.ts,
            "last_shift_step": self.last_shift_step,
            "num_detections": len(self.detections),
            "last_pval": self.last_pval,
            "last_score": self.last_score,
            "last_err_r": self.last_err_r,
            "last_err_d": self.last_err_d,
            "replay_size": len(self._replay),
            "score_hist_size": len(self._scores),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _one_hot(self, a):
        v = np.zeros(self.num_actions, dtype=np.float32)
        v[int(a)] = 1.0
        return v

    def _flush_score_buf(self):
        """Run a single batched TSN forward pass over all buffered transitions,
        update Welford stats, and append per-transition scores to self._scores.

        This replaces the old per-step single-sample _compute_errors() pattern.
        By batching score_batch_size transitions together we reduce PyTorch
        dispatch calls by score_batch_size x while producing identical per-step
        scores (same arithmetic, same Welford update order).
        """
        if not self._score_buf:
            return
        batch = self._score_buf
        self._score_buf = []

        phis = np.stack([b[0] for b in batch])          # (K, latent_dim)
        acts = np.stack([self._one_hot(b[1]) for b in batch])  # (K, num_actions)
        rews = np.array([b[2] for b in batch], dtype=np.float32)  # (K,)
        nphs = np.stack([b[3] for b in batch])          # (K, latent_dim)

        self.net.eval()
        with torch.no_grad():
            phi_t = torch.from_numpy(phis).to(self.device)
            act_t = torch.from_numpy(acts).to(self.device)
            nph_t = torch.from_numpy(nphs).to(self.device)
            r_hat, n_hat = self.net(phi_t, act_t)
            # Move results to numpy once (avoids repeated .item() calls)
            err_rs = (r_hat.squeeze(-1).cpu().numpy() - rews) ** 2   # (K,)
            err_ds = ((n_hat.cpu().numpy() - nphs) ** 2).mean(axis=1)  # (K,)

        for err_r, err_d in zip(err_rs, err_ds):
            err_r, err_d = float(err_r), float(err_d)
            self._w_r.update(err_r)
            self._w_d.update(err_d)
            z_r = max(0.0, (err_r - self._w_r.mean) / (self._w_r.std + 1e-8))
            z_d = max(0.0, (err_d - self._w_d.mean) / (self._w_d.std + 1e-8))
            S_t = z_r + z_d
            self._scores.append(S_t)
            self.last_score = S_t
            self.last_err_r = err_r
            self.last_err_d = err_d

    def _compute_errors(self, phi, action, reward, next_phi):
        """Single-sample prediction errors (kept for external callers / tests).

        Not used by step() anymore — step() uses _flush_score_buf() instead.
        """
        self.net.eval()
        with torch.no_grad():
            p = torch.from_numpy(phi).to(self.device).unsqueeze(0)
            n = torch.from_numpy(next_phi).to(self.device).unsqueeze(0)
            a = torch.from_numpy(self._one_hot(action)).to(self.device).unsqueeze(0)
            r_hat, n_hat = self.net(p, a)
            err_r = float(((r_hat.item() - reward) ** 2))
            err_d = float(((n_hat - n) ** 2).mean().item())
        return err_r, err_d

    def _train_step(self):
        """Single Adam step on a minibatch sampled from the replay."""
        self.net.train()
        idx = self.rng.integers(0, len(self._replay), size=self.batch_size)
        batch = [self._replay[i] for i in idx]
        phis = np.stack([b[0] for b in batch]).astype(np.float32)
        acts = np.stack([self._one_hot(b[1]) for b in batch]).astype(np.float32)
        rews = np.array([b[2] for b in batch], dtype=np.float32)
        nphs = np.stack([b[3] for b in batch]).astype(np.float32)

        phi_t = torch.from_numpy(phis).to(self.device)
        act_t = torch.from_numpy(acts).to(self.device)
        rew_t = torch.from_numpy(rews).to(self.device)
        nph_t = torch.from_numpy(nphs).to(self.device)

        r_hat, n_hat = self.net(phi_t, act_t)
        loss = F.mse_loss(r_hat, rew_t) + F.mse_loss(n_hat, nph_t)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
        self.opt.step()

    def _on_detected(self):
        self.detections.append(self.ts)
        self.last_shift_step = self.ts
        self._scores.clear()
        self._score_buf.clear()   # discard pre-flush buffered transitions
        # Reset the cached p-value/score so a stale post-fire value doesn't
        # trip a downstream aggregator (e.g. HybridDetector) on the next tick.
        self.last_pval = 1.0
        self.last_score = 0.0
        # NB: we deliberately keep the TSN weights and replay -- the body has
        # already learned useful structure; clearing them would waste the
        # detector's learning.  The running error-norm stats are reset so the
        # new regime's scale is re-estimated from scratch.
        self._w_r = Welford()
        self._w_d = Welford()
        # Re-arm the one-shot "warmup-end" rebaseline so the post-shift
        # regime also starts from a clean Welford once its stable-phase
        # guard expires -- this mirrors the initial warmup recalibration.
        self._warmup_reset_done = False
        # Point that future stale_phase checks treat as the new warmup edge:
        # we still require stable_phase steps before firing, and the first
        # tick after that window triggers the reset above.


# ----------------------------------------------------------------------
# Welch's one-sided t-test (upper tail of t distribution)
# ----------------------------------------------------------------------
def _welch_one_sided(cur: np.ndarray, ref: np.ndarray):
    """Return (t_statistic, p_value) for H1: mean(cur) > mean(ref).

    Uses scipy.stats.ttest_ind with ``equal_var=False`` and selects the
    correct one-sided p-value depending on scipy's direction convention.
    """
    try:
        res = _scistats.ttest_ind(cur, ref, equal_var=False,
                                  alternative="greater")
        return float(res.statistic), float(res.pvalue)
    except TypeError:
        # Very old scipy (< 1.6) doesn't take `alternative`; fall back to
        # a manual upper-tail p-value from the two-sided result.
        t, p_two = _scistats.ttest_ind(cur, ref, equal_var=False)
        if t > 0:
            return float(t), float(p_two / 2.0)
        return float(t), float(1.0 - p_two / 2.0)


__all__ = ["ImplicitDetector", "TaskSignatureNet", "Welford",
           "_welch_one_sided"]
