"""FAME for MinAtar, with optional boundary-free task-shift detection.

Detection modes (selected by `--detector`):
  * oracle   -- original FAME: fires at every `--switch` step (ceiling).
  * swoks    -- Approach 1: statistical SWD+KS on latent action-reward.
  * implicit -- Approach 2: dual-head Task-Signature Network (reward +
                forward-dynamics prediction errors) with Welch's t-test.
  * hybrid   -- Approach 3: 3-state cascade + log-evidence fusion of
                implicit (fast, reactive) and swoks (slow, reliable).

In all non-oracle modes the environment still switches every `--switch` steps
so we retain ground truth for post-hoc detection metrics, but the agent only
reacts when its detector fires.  Results are pickled to
`results/FAME_{mode}_...pkl` and consumed by `compare_oracle_vs_swoks.py`.

Legacy: `--use_swoks 0/1` is still accepted as an alias for
`--detector oracle` / `--detector swoks`.
"""

import copy
import glob
import os
import pickle
import random
import time
from argparse import ArgumentParser
from configparser import ConfigParser

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from scipy import stats
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from CL_envs import CL_envs_func_replacement
from detector_heuristics import format_suggestion, suggest_params
from hybrid_detector import HybridDetector
from implicit_detector import ImplicitDetector
from model import CNN
from replay import expReplay, expReplay_Meta
from swoks_detector import SwoksDetector


# ----------------------------------------------------------------------
# Arguments
# ----------------------------------------------------------------------
def build_parser():
    p = ArgumentParser(description="Parameters for FAME in MinAtar")
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--env-name', type=str, default="all")
    p.add_argument('--t-steps', type=int, default=3500000)
    p.add_argument('--switch', type=int, default=500000)
    p.add_argument('--lr1', type=float, default=1e-3, help="meta learner lr")
    p.add_argument('--lr2', type=float, default=1e-5, help="fast learner lr")
    p.add_argument('--update', type=int, default=50000)
    p.add_argument('--decay', type=float, default=0.75)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--save', action="store_true")
    p.add_argument('--plot', action="store_true")
    p.add_argument('--save-model', action="store_true")
    p.add_argument('--gpu', type=int, default=0)

    p.add_argument('--seq', type=int, default=0)
    p.add_argument('--size_fast2meta', type=int, default=12000)
    p.add_argument('--size_meta', type=int, default=100000)
    p.add_argument('--detection_step', type=int, default=1200)
    p.add_argument('--epoch_meta', type=int, default=200)
    p.add_argument('--reset', type=int, default=1)

    p.add_argument('--warmstep', type=int, default=50000)
    p.add_argument('--lambda_reg', type=float, default=1.0)
    p.add_argument('--use_ttest', type=int, default=0)

    # --- Boundary-free extension ---
    p.add_argument('--detector', type=str, default=None,
                   choices=["oracle", "swoks", "implicit", "hybrid"],
                   help="Detector to use.  If omitted, falls back to "
                        "--use_swoks for backwards compatibility.")
    p.add_argument('--use_swoks', type=int, default=0,
                   help="DEPRECATED alias: 0 -> --detector oracle, "
                        "1 -> --detector swoks")
    p.add_argument('--auto_params', type=int, default=0,
                   help="If 1, detector_heuristics.suggest_params() fills in "
                        "any detector arg left at its default.")

    # --- SWOKS (used by swoks & hybrid) ---
    p.add_argument('--swoks_L_D', type=int, default=1200)
    p.add_argument('--swoks_L_W', type=int, default=30)
    p.add_argument('--swoks_alpha', type=float, default=1e-3)
    p.add_argument('--swoks_beta', type=float, default=2.0)
    p.add_argument('--swoks_stable_phase', type=int, default=36000)
    p.add_argument('--swoks_interval', type=int, default=240)
    p.add_argument('--swoks_warmup', type=int, default=5000)
    p.add_argument('--swoks_snapshot', type=int, default=1,
                   help="Use a pre-detection snapshot of the fast learner as "
                        "the 'fast' warm-up candidate (mitigates detection "
                        "latency).")
    p.add_argument('--swoks_snapshot_interval', type=int, default=6000)
    p.add_argument('--swoks_max_wait', type=int, default=0,
                   help="0 disables the FN-fallback probe.")

    # --- Implicit (used by implicit & hybrid) ---
    p.add_argument('--imp_L_D', type=int, default=1200)
    p.add_argument('--imp_alpha', type=float, default=1e-3)
    p.add_argument('--imp_stable_phase', type=int, default=36000)
    p.add_argument('--imp_interval', type=int, default=240)
    p.add_argument('--imp_warmup', type=int, default=5000)
    p.add_argument('--imp_lr', type=float, default=1e-4)
    p.add_argument('--imp_replay', type=int, default=0,
                   help="0 -> 4*imp_L_D")
    p.add_argument('--imp_update_every', type=int, default=16)
    p.add_argument('--imp_max_wait', type=int, default=0)

    # --- Hybrid-specific ---
    p.add_argument('--hyb_tau_imp_loose', type=float, default=1e-2)
    p.add_argument('--hyb_tau_imp_strict', type=float, default=1e-6)
    p.add_argument('--hyb_tau_stat_strict', type=float, default=1e-3)
    p.add_argument('--hyb_tau_stat_failsafe', type=float, default=1e-6)
    p.add_argument('--hyb_tau_combined', type=float, default=15.0)
    p.add_argument('--hyb_horizon', type=int, default=480)
    p.add_argument('--hyb_persistence', type=int, default=2)

    p.add_argument('--results_dir', type=str, default="results")
    p.add_argument('--models_dir', type=str, default="models")

    # --- Checkpointing ---
    p.add_argument('--checkpoint_dir', type=str, default=None,
                   help="Directory to save/load training checkpoints. "
                        "If None, checkpointing is disabled.")
    p.add_argument('--checkpoint_interval', type=int, default=100000,
                   help="Save a checkpoint every this many training steps "
                        "(default: 100 000).")

    return p


# ----------------------------------------------------------------------
# Detector factory + unified adapter interface
# ----------------------------------------------------------------------
class DetectorAdapter:
    """Unifies `.step(phi, action, reward, next_phi) -> bool` across all
    detectors.  SWOKS doesn't use next_phi, the others do."""

    def __init__(self, kind: str, core):
        self.kind = kind
        self.core = core

    def step(self, phi, action, reward, next_phi) -> bool:
        if self.kind == "swoks":
            return self.core.step(phi, action, reward)
        # implicit / hybrid both take next_phi
        return self.core.step(phi, action, reward, next_phi)

    def stats(self):
        return self.core.stats()

    def reset_after_boundary(self):
        self.core.reset_after_boundary()


def resolve_detector_kind(args) -> str:
    """Resolve --detector vs legacy --use_swoks into a canonical kind."""
    if args.detector is not None:
        return args.detector
    return "swoks" if args.use_swoks else "oracle"


def build_detector(kind, args, latent_dim, num_actions, device, on_suspect):
    """Construct the right detector (or None for oracle)."""
    if kind == "oracle":
        return None
    if kind == "swoks":
        return DetectorAdapter("swoks", SwoksDetector(
            latent_dim=latent_dim,
            L_D=args.swoks_L_D, L_W=args.swoks_L_W,
            alpha=args.swoks_alpha, beta=args.swoks_beta,
            stable_phase=args.swoks_stable_phase,
            detection_interval=args.swoks_interval,
            warmup=args.swoks_warmup,
            max_wait=args.swoks_max_wait,
            seed=args.seed,
        ))
    if kind == "implicit":
        return DetectorAdapter("implicit", ImplicitDetector(
            latent_dim=latent_dim, num_actions=num_actions,
            L_D=args.imp_L_D, alpha=args.imp_alpha,
            stable_phase=args.imp_stable_phase,
            warmup=args.imp_warmup,
            detection_interval=args.imp_interval,
            max_wait=args.imp_max_wait,
            lr=args.imp_lr,
            replay_capacity=(args.imp_replay if args.imp_replay > 0
                             else None),
            update_every=args.imp_update_every,
            device=str(device), seed=args.seed,
        ))
    if kind == "hybrid":
        # Inner detectors have alpha=0.0 so their `p < alpha` check is
        # always False and they never self-fire; the hybrid FSM reads
        # their live `last_pval` to make firing decisions.  If the
        # inner auto-fired, it would reset its cached p-value and the
        # hybrid would lose the signal.
        imp = ImplicitDetector(
            latent_dim=latent_dim, num_actions=num_actions,
            L_D=args.imp_L_D, alpha=0.0,
            stable_phase=args.imp_stable_phase,
            warmup=args.imp_warmup,
            detection_interval=args.imp_interval,
            max_wait=0,        # hybrid controls firing, not inner
            lr=args.imp_lr,
            replay_capacity=(args.imp_replay if args.imp_replay > 0
                             else None),
            update_every=args.imp_update_every,
            device=str(device), seed=args.seed,
        )
        swk = SwoksDetector(
            latent_dim=latent_dim,
            L_D=args.swoks_L_D, L_W=args.swoks_L_W,
            alpha=0.0, beta=args.swoks_beta,
            stable_phase=args.swoks_stable_phase,
            detection_interval=args.swoks_interval,
            warmup=args.swoks_warmup,
            max_wait=0,        # hybrid controls firing, not inner
            seed=args.seed,
        )
        hyb = HybridDetector(
            imp, swk,
            tau_imp_loose=args.hyb_tau_imp_loose,
            tau_imp_strict=args.hyb_tau_imp_strict,
            tau_stat_strict=args.hyb_tau_stat_strict,
            tau_stat_failsafe=args.hyb_tau_stat_failsafe,
            tau_combined=args.hyb_tau_combined,
            horizon=args.hyb_horizon,
            persistence=args.hyb_persistence,
            on_suspect=on_suspect,
        )
        return DetectorAdapter("hybrid", hyb)
    raise ValueError(f"Unknown detector kind: {kind}")


# ----------------------------------------------------------------------
# Action helpers
# ----------------------------------------------------------------------
def _obs_to_tensor(obs, device):
    obs = np.moveaxis(obs, 2, 0)
    return torch.tensor(obs, dtype=torch.float, device=device).unsqueeze(0)


def get_action_detection(c_obs, net, device):
    """Greedy action (used in the detection/warm-up phase)."""
    x = _obs_to_tensor(c_obs, device)
    with torch.no_grad():
        q = net(x)
    a = q.max(1)[1].item()
    return a, q[0][a]


def get_action_and_latent(c_obs, net, epsilon, action_space, device):
    """Epsilon-greedy action; also return the penultimate latent vector."""
    x = _obs_to_tensor(c_obs, device)
    with torch.no_grad():
        q, phi = net(x, return_latent=True)
    if np.random.random() <= epsilon:
        a = action_space.sample()
    else:
        a = q.max(1)[1].item()
    return a, phi[0].detach().cpu().numpy()


# ----------------------------------------------------------------------
# Training helpers
# ----------------------------------------------------------------------
def train_fast(Fast_Learner, Fast_opt, Target_net, Fast_criterion,
               exp_replay_fast, gamma, lambda_reg, reg=None):
    states, actions, next_states, rewards, done = exp_replay_fast.sample()
    with torch.no_grad():
        fast_next_pred = Target_net(next_states)
    targets = rewards + (1 - done) * gamma * fast_next_pred.max(1)[0].reshape(-1, 1)
    fast_pred = Fast_Learner(states).gather(1, actions)
    loss = Fast_criterion(fast_pred, targets)

    if reg is not None and lambda_reg > 0:
        with torch.no_grad():
            soft_target = F.softmax(reg(states), dim=-1)
        logit_input = F.log_softmax(Fast_Learner(states), dim=-1)
        loss_reg = F.kl_div(logit_input, soft_target, reduction="batchmean")
        loss = loss + lambda_reg * loss_reg

    Fast_opt.zero_grad()
    loss.backward()
    Fast_opt.step()
    return loss.item()


def train_meta(Meta_Learner, Meta_opt, Meta_scheduler, Meta_criterion2,
               exp_replay_meta, exp_replay_fast2meta, epoch_meta, batch_size,
               gameid, device):
    """Meta-learner MLE update on union(past meta buffer, new fast2meta)."""
    u_steps = max(1, (exp_replay_meta.size() // batch_size) - 1)
    weight_every = max(1, gameid)  # match original behaviour

    for epoch in range(epoch_meta):
        for i in range(u_steps):
            states_meta, actions_meta = exp_replay_meta.sample()
            logits = Meta_Learner(states_meta.to(device))
            log_probs = F.log_softmax(logits, dim=-1)
            loss1 = Meta_criterion2(log_probs, actions_meta.to(device).view(-1))

            if i % weight_every == 0 and exp_replay_fast2meta.size() > 0:
                states_fast, actions_fast = exp_replay_fast2meta.sample()
                logits = Meta_Learner(states_fast.to(device))
                log_probs = F.log_softmax(logits, dim=-1)
                loss2 = Meta_criterion2(log_probs,
                                        actions_fast.to(device).view(-1))
                loss = loss1 + loss2
            else:
                loss = loss1

            Meta_opt.zero_grad()
            loss.backward()
            Meta_opt.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  meta-epoch {epoch+1}/{epoch_meta} loss={loss.item():.2e}",
                  time.strftime("%H:%M:%S"))
        if (epoch + 1) % 2 == 0:
            Meta_scheduler.step()


# ----------------------------------------------------------------------
# Boundary trigger: shared by oracle and SWOKS paths
# ----------------------------------------------------------------------
def hypothesis_test(list1, list2, avg1, avg2):
    if len(list1) < 2 or len(list2) < 2:
        return avg1 > avg2
    _, p = stats.ttest_ind(list1, list2, alternative="greater",
                           equal_var=False)
    return p < 0.05


class FameBoundaryTrigger:
    """Encapsulates the FAME response to a boundary.

    Used for both `--use_swoks 0` (oracle trigger at every --switch step) and
    `--use_swoks 1` (trigger when the SWOKS detector fires).
    """

    def __init__(self, args, env_factory, device, in_channels, num_actions,
                 epsilon, gamma):
        self.args = args
        self.env_factory = env_factory
        self.device = device
        self.in_channels = in_channels
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.gamma = gamma

    def run(self, *, env, gameid, step, cs, Fast_Learner, Meta_Learner,
            Random_Learner, Fast_opt, Target_net, exp_replay_fast,
            returns_array, avg_return, pbar, fast_snapshot=None):
        """Execute the post-boundary detection + warm-up pass.

        Returns a dict with the possibly-updated fast learner, optimiser,
        avg_return, step counter, env state, and the chosen Flag_Reg label.
        """
        args = self.args
        device = self.device
        MAX_STEP = 300
        max_step = 0

        cs_initial = cs
        fast_candidate = fast_snapshot if fast_snapshot is not None else Fast_Learner

        FLAG_ENV2 = gameid >= 2  # as in original: skip meta eval on 2nd env
        if not FLAG_ENV2:
            print("  (no meta detection on 2nd env)")

        Num_detection_meta = args.detection_step * int(FLAG_ENV2)
        Num_detection_fast = args.detection_step

        # --- evaluate fast candidate ---
        epi_ret = 0.0
        rewards_fast = []
        for _ in range(Num_detection_fast):
            a, _ = get_action_detection(cs, fast_candidate, device)
            ns, rew, done, _ = env.step(a)
            exp_replay_fast.store(cs, a, ns, rew, done)
            epi_ret += rew
            cs = ns
            step += 1
            max_step += 1
            if done or max_step > MAX_STEP:
                cs = env.reset()
                avg_return = 0.99 * avg_return + 0.01 * epi_ret
                rewards_fast.append(epi_ret)
                epi_ret = 0.0
                max_step = 0
            if step < len(returns_array):
                returns_array[step] = avg_return
            pbar.update(1)
        avg_fast = np.mean(rewards_fast) if rewards_fast else -1e3
        print(f"  fast-candidate eval: mean={avg_fast:.2f} "
              f"episodes={len(rewards_fast)}")

        # --- evaluate meta (if applicable) ---
        rewards_meta = []
        if Num_detection_meta > 0:
            cs = env.reset()
            max_step = 0
            epi_ret = 0.0
            for _ in range(Num_detection_meta):
                a, _ = get_action_detection(cs, Meta_Learner, device)
                ns, rew, done, _ = env.step(a)
                exp_replay_fast.store(cs, a, ns, rew, done)
                epi_ret += rew
                cs = ns
                step += 1
                max_step += 1
                if done or max_step > MAX_STEP:
                    cs = env.reset()
                    avg_return = 0.99 * avg_return + 0.01 * epi_ret
                    rewards_meta.append(epi_ret)
                    epi_ret = 0.0
                    max_step = 0
                if step < len(returns_array):
                    returns_array[step] = avg_return
                pbar.update(1)
        avg_meta = np.mean(rewards_meta) if rewards_meta else -1e3
        print(f"  meta-candidate eval: mean={avg_meta:.2f} "
              f"episodes={len(rewards_meta)}")

        # --- random reference ---
        _, v_rand = get_action_detection(cs_initial, Random_Learner, device)
        avg_rand = float(v_rand.cpu().numpy())

        # --- hypothesis test ---
        if args.use_ttest == 1:
            meta_wins = hypothesis_test(rewards_meta, rewards_fast,
                                        avg_meta, avg_fast)
            fast_wins = hypothesis_test(rewards_fast, rewards_meta,
                                        avg_fast, avg_meta)
        else:
            meta_wins = avg_meta > avg_fast
            fast_wins = avg_fast > avg_meta

        flag = "Random"
        meta_warmup = 0
        if meta_wins and avg_meta > avg_rand:
            print("  -> use Meta initialisation + BC warm-up")
            meta_warmup = 1
            flag = "Meta"
        elif fast_wins and avg_fast > avg_rand:
            print("  -> keep Fast initialisation (fine-tune)")
            flag = "Fast"
            if fast_snapshot is not None:
                # Adopt the snapshot as our starting fast network.
                Fast_Learner.load_state_dict(fast_snapshot.state_dict())
        else:
            if args.reset == 1:
                print("  -> random re-init of the fast learner")
                Fast_Learner = CNN(self.in_channels,
                                   self.num_actions).to(device)
                Fast_opt = optim.Adam(Fast_Learner.parameters(), lr=args.lr2)
                flag = "Random"
            else:
                print("  -> keep Fast initialisation (reset=0)")
                flag = "Fast"

        if Num_detection_meta + Num_detection_fast > 0:
            Target_net.load_state_dict(Fast_Learner.state_dict())
            cs = env.reset()

        return {
            "Fast_Learner": Fast_Learner,
            "Fast_opt": Fast_opt,
            "Target_net": Target_net,
            "cs": cs,
            "step": step,
            "avg_return": avg_return,
            "meta_warmup": meta_warmup,
            "flag": flag,
        }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Checkpointing helpers
# ----------------------------------------------------------------------
def _checkpoint_path(checkpoint_dir: str, filename: str) -> str:
    """Return the path for the latest checkpoint file for this run."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    return os.path.join(checkpoint_dir, f"{filename}_checkpoint.pkl")


def save_checkpoint(
    path: str,
    *,
    step: int,
    gameid: int,
    avg_return: float,
    epi_return: float,
    meta_warmup: int,
    returns_array,
    oracle_boundaries,
    detected_boundaries,
    detection_log,
    flag_history,
    Games,
    snapshots,
    Fast_Learner,
    Fast_opt,
    Target_net,
    Meta_Learner,
    Meta_opt,
    Meta_scheduler,
    exp_replay_fast,
    exp_replay_fast2meta,
    exp_replay_meta,
    detector,
    elapsed_time_so_far: float,
) -> None:
    """Persist all training state needed to resume from *step*."""
    tmp_path = path + ".tmp"
    blob = {
        "step": step,
        "gameid": gameid,
        "avg_return": avg_return,
        "epi_return": epi_return,
        "meta_warmup": meta_warmup,
        "returns_array": returns_array,
        "oracle_boundaries": list(oracle_boundaries),
        "detected_boundaries": list(detected_boundaries),
        "detection_log": list(detection_log),
        "flag_history": list(flag_history),
        "Games": list(Games),
        "snapshots": [(s, sd) for s, sd in snapshots],
        # Neural-network weights + optimiser states
        "Fast_Learner_sd": Fast_Learner.state_dict(),
        "Fast_opt_sd": Fast_opt.state_dict(),
        "Target_net_sd": Target_net.state_dict(),
        "Meta_Learner_sd": Meta_Learner.state_dict(),
        "Meta_opt_sd": Meta_opt.state_dict(),
        "Meta_scheduler_sd": Meta_scheduler.state_dict(),
        # Replay buffers stored as raw deque contents
        "exp_replay_fast_mem": list(exp_replay_fast.memory),
        "exp_replay_fast2meta_mem": list(exp_replay_fast2meta.memory),
        "exp_replay_meta_mem": list(exp_replay_meta.memory),
        # Detector internal state (optional -- may not be serialisable for
        # all detector types; we catch exceptions and warn gracefully).
        "detector_state": None,
        # Accumulated wall-clock time from all previous sessions.
        "elapsed_time_so_far": elapsed_time_so_far,
    }
    # Attempt to serialise the detector's own state.
    if detector is not None:
        try:
            import io
            buf = io.BytesIO()
            pickle.dump(detector.core, buf)
            blob["detector_state"] = buf.getvalue()
        except Exception as exc:
            print(f"[checkpoint] warning: could not serialise detector state: {exc}")
    with open(tmp_path, "wb") as f:
        pickle.dump(blob, f)
    os.replace(tmp_path, path)  # atomic on POSIX
    print(f"[checkpoint] saved -> {path}  (step={step})")


def load_latest_checkpoint(checkpoint_dir: str, filename: str):
    """Return the checkpoint dict if one exists, else None."""
    path = _checkpoint_path(checkpoint_dir, filename)
    if not os.path.exists(path):
        return None, None
    print(f"[checkpoint] loading -> {path}")
    with open(path, "rb") as f:
        blob = pickle.load(f)
    return blob, path


def main():
    args = build_parser().parse_args()

    config = ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), "misc_params.cfg"))
    misc_param = config[str(args.env_name)]
    gamma = float(misc_param["gamma"])
    epsilon = float(misc_param["epsilon"])

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("device =", device)

    # Track wall-clock time; this accumulates across resumed sessions.
    _session_start_time = time.time()

    detector_kind = resolve_detector_kind(args)
    mode_tag = detector_kind

    # Optional: fill in unspecified detector defaults via heuristics.
    if args.auto_params:
        suggested = suggest_params(args.switch, mini=(args.switch < 50000))
        print(format_suggestion(suggested, args.switch))
        # Only override if the user left the flag at the argparse default.
        defaults = build_parser().parse_args([]).__dict__
        for k, v in suggested.as_dict().items():
            arg_key = {
                "L_D": "swoks_L_D", "L_W": "swoks_L_W",
                "detection_interval": "swoks_interval",
                "warmup": "swoks_warmup",
                "stable_phase": "swoks_stable_phase",
                "alpha": "swoks_alpha", "beta": "swoks_beta",
                "imp_alpha": "imp_alpha", "imp_lr": "imp_lr",
                "imp_buffer": "imp_replay",
                "imp_update_every": "imp_update_every",
                "hyb_tau_imp_loose": "hyb_tau_imp_loose",
                "hyb_tau_imp_strict": "hyb_tau_imp_strict",
                "hyb_tau_stat_strict": "hyb_tau_stat_strict",
                "hyb_tau_combined": "hyb_tau_combined",
                "hyb_horizon": "hyb_horizon",
                "hyb_persistence": "hyb_persistence",
            }.get(k)
            if arg_key and getattr(args, arg_key) == defaults.get(arg_key):
                setattr(args, arg_key, v)
                if arg_key.startswith("swoks_") and k != "alpha":
                    # Mirror into imp_* where appropriate.
                    mirror = {
                        "swoks_L_D": "imp_L_D",
                        "swoks_interval": "imp_interval",
                        "swoks_warmup": "imp_warmup",
                        "swoks_stable_phase": "imp_stable_phase",
                    }.get(arg_key)
                    if mirror and getattr(args, mirror) == defaults.get(mirror):
                        setattr(args, mirror, v)

    filename = (
        f"FAME_{mode_tag}_steps_{args.t_steps}_switch_{args.switch}"
        f"_seq_{args.seq}_warmstep_{args.warmstep}"
        f"_lambda_{args.lambda_reg}_seed_{args.seed}"
    )
    print(f"Run: {filename}")
    print(args)

    # ------------------------------------------------------------------
    # Try to resume from an existing checkpoint
    # ------------------------------------------------------------------
    _checkpoint_blob = None
    if args.checkpoint_dir:
        _checkpoint_blob, _ckpt_path = load_latest_checkpoint(
            args.checkpoint_dir, filename
        )

    # --- Initial env (always needed for shapes even during resume) ---
    gameid = 0
    env = CL_envs_func_replacement(seq=args.seq, game_id=gameid,
                                   seed=args.seed)
    Games = [env.game_name]
    in_channels = env.observation_space.shape[2]
    num_actions = env.action_space.n

    # --- Networks ---
    Fast_Learner = CNN(in_channels, num_actions).to(device)
    Fast_opt = optim.Adam(Fast_Learner.parameters(), lr=args.lr2)
    Fast_criterion = torch.nn.MSELoss()

    Random_Learner = CNN(in_channels, num_actions).to(device)
    Meta_Learner = CNN(in_channels, num_actions).to(device)
    Meta_opt = optim.Adam(Meta_Learner.parameters(), lr=args.lr1)
    Meta_scheduler = ExponentialLR(Meta_opt, gamma=0.95)
    Meta_criterion2 = torch.nn.NLLLoss()

    Target_net = CNN(in_channels, num_actions).to(device)
    Target_net.load_state_dict(Fast_Learner.state_dict())

    exp_replay_fast = expReplay(batch_size=args.batch_size, device=device)
    exp_replay_fast2meta = expReplay_Meta(
        max_size=args.size_fast2meta, batch_size=args.batch_size,
        device=device
    )
    exp_replay_meta = expReplay_Meta(
        max_size=args.size_meta, batch_size=args.batch_size, device=device
    )

    # --- Snapshot ring ---
    # Used by:
    #   * swoks:    periodic snapshots at --swoks_snapshot_interval; we pick
    #               the closest one to "stable_phase/2 steps ago" at fire time.
    #   * hybrid:   event-driven snapshot at SUSPECT entry (see on_suspect).
    snapshots = []              # list of (step, state_dict)
    snapshot_keep = 3

    def take_snapshot(step):
        if not args.swoks_snapshot:
            return
        snapshots.append((step, copy.deepcopy(Fast_Learner.state_dict())))
        if len(snapshots) > snapshot_keep:
            snapshots.pop(0)

    def pick_prebreak_snapshot(step):
        """Return a pre-contamination snapshot of the fast learner as a
        warm-up candidate.  Falls back to None when no snapshots exist."""
        if not args.swoks_snapshot or not snapshots:
            return None
        target_age = max(args.swoks_L_D, args.swoks_stable_phase // 2)
        target_step = step - target_age
        best = min(snapshots, key=lambda s: abs(s[0] - target_step))
        snap = CNN(in_channels, num_actions).to(device)
        snap.load_state_dict(best[1])
        snap.eval()
        return snap

    def on_suspect(_ts):
        """Hybrid's SUSPECT-entry hook: stash an event-driven snapshot."""
        take_snapshot(_ts)

    # --- Build detector ---
    latent_dim = Fast_Learner.latent_dim()
    detector = build_detector(detector_kind, args, latent_dim, num_actions,
                              device, on_suspect=on_suspect)

    trigger = FameBoundaryTrigger(
        args, CL_envs_func_replacement, device, in_channels, num_actions,
        epsilon, gamma
    )

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------
    returns_array = np.zeros(args.t_steps)
    oracle_boundaries = []     # true env switches
    detected_boundaries = []   # where the agent reacted
    detection_log = []          # rich log, per detection
    flag_history = []

    avg_return = 0.0
    epi_return = 0.0
    meta_warmup = 0
    step = 0
    _elapsed_time_prev_sessions = 0.0  # seconds accumulated before this session

    # ------------------------------------------------------------------
    # Restore checkpoint (if one was found)
    # ------------------------------------------------------------------
    if _checkpoint_blob is not None:
        ck = _checkpoint_blob
        step = ck["step"]
        gameid = ck["gameid"]
        avg_return = ck["avg_return"]
        epi_return = ck["epi_return"]
        meta_warmup = ck["meta_warmup"]
        returns_array = ck["returns_array"]
        oracle_boundaries = ck["oracle_boundaries"]
        detected_boundaries = ck["detected_boundaries"]
        detection_log = ck["detection_log"]
        flag_history = ck["flag_history"]
        Games = ck["Games"]
        snapshots[:] = ck["snapshots"]
        _elapsed_time_prev_sessions = ck.get("elapsed_time_so_far", 0.0)

        # Restore network weights
        Fast_Learner.load_state_dict(ck["Fast_Learner_sd"])
        Fast_opt.load_state_dict(ck["Fast_opt_sd"])
        Target_net.load_state_dict(ck["Target_net_sd"])
        Meta_Learner.load_state_dict(ck["Meta_Learner_sd"])
        Meta_opt.load_state_dict(ck["Meta_opt_sd"])
        Meta_scheduler.load_state_dict(ck["Meta_scheduler_sd"])

        # Restore replay buffers
        exp_replay_fast.memory.clear()
        exp_replay_fast.memory.extend(ck["exp_replay_fast_mem"])
        exp_replay_fast2meta.memory.clear()
        exp_replay_fast2meta.memory.extend(ck["exp_replay_fast2meta_mem"])
        exp_replay_meta.memory.clear()
        exp_replay_meta.memory.extend(ck["exp_replay_meta_mem"])

        # Restore detector internal state (best-effort)
        if detector is not None and ck.get("detector_state") is not None:
            try:
                import io
                restored_core = pickle.load(io.BytesIO(ck["detector_state"]))
                detector.core = restored_core
                print("[checkpoint] detector state restored.")
            except Exception as exc:
                print(f"[checkpoint] warning: could not restore detector state: {exc}")

        # Restore environment to the correct game
        env = CL_envs_func_replacement(seq=args.seq, game_id=gameid,
                                       seed=args.seed)
        print(f"[checkpoint] Resumed from step {step}, gameid={gameid}, "
              f"prev elapsed={_elapsed_time_prev_sessions:.1f}s")
    else:
        print(f"[checkpoint] No checkpoint found; starting fresh.")

    cs = env.reset()
    print(f"##### Env {gameid+1}: {env.game_name}")

    pbar = tqdm(total=args.t_steps, initial=step)

    num_envs = max(1, args.t_steps // args.switch)
    is_oracle = detector_kind == "oracle"
    is_boundary_free = not is_oracle
    _last_checkpoint_step = step  # track when we last saved

    # ------------------------------------------------------------------
    # Main interaction loop
    # ------------------------------------------------------------------
    while step < args.t_steps:

        # ---------- silent env switch (truth label for evaluation) ----------
        if step > 0 and step % args.switch == 0 and gameid + 1 < num_envs:
            gameid += 1
            env = CL_envs_func_replacement(seq=args.seq, game_id=gameid,
                                           seed=args.seed)
            Games.append(env.game_name)
            cs = env.reset()
            oracle_boundaries.append(step)
            print(f"[oracle switch @ step {step}] -> {env.game_name} "
                  f"(gameid={gameid})")
            if is_oracle:
                # Oracle mode: fire FAME immediately.
                avg_return = 0.0 if args.reset == 1 else avg_return
                meta_warmup = 0
                print(f"##### Env {gameid+1}/{num_envs}: {env.game_name} -- "
                      f"running oracle FAME boundary trigger")
                out = trigger.run(
                    env=env, gameid=gameid, step=step, cs=cs,
                    Fast_Learner=Fast_Learner, Meta_Learner=Meta_Learner,
                    Random_Learner=Random_Learner, Fast_opt=Fast_opt,
                    Target_net=Target_net,
                    exp_replay_fast=exp_replay_fast,
                    returns_array=returns_array, avg_return=avg_return,
                    pbar=pbar, fast_snapshot=None,
                )
                Fast_Learner = out["Fast_Learner"]
                Fast_opt = out["Fast_opt"]
                Target_net = out["Target_net"]
                cs = out["cs"]
                step = out["step"]
                avg_return = out["avg_return"]
                meta_warmup = out["meta_warmup"]
                flag_history.append(out["flag"])
                detected_boundaries.append(step)
                detection_log.append({
                    "step": step, "true_switch": oracle_boundaries[-1],
                    "source": "oracle", "flag": out["flag"],
                })

        # ---------- periodic snapshot for the detection-latency fix ----------
        # SWOKS uses periodic snapshots; hybrid uses event-driven ones
        # (via on_suspect); implicit currently reuses the periodic ring too.
        if is_boundary_free and args.swoks_snapshot \
                and detector_kind in ("swoks", "implicit") \
                and step > 0 and step % args.swoks_snapshot_interval == 0:
            take_snapshot(step)

        # ---------- act ----------
        c_action, phi = get_action_and_latent(
            cs, Fast_Learner, epsilon, env.action_space, device
        )
        ns, rew, done, _ = env.step(c_action)
        epi_return += rew
        exp_replay_fast.store(cs, c_action, ns, rew, done)

        # ---------- fast2meta streaming ----------
        # In oracle mode we match the original FAME schedule (last
        # size_fast2meta steps of each task).  In boundary-free modes we
        # stream continuously into a FIFO of capacity size_fast2meta so that
        # when a detection fires the recent window is already queued up.
        if is_boundary_free:
            exp_replay_fast2meta.store(cs, c_action)
        else:
            start = ((step // args.switch) + 1) * args.switch - args.size_fast2meta - 1
            end = ((step // args.switch) + 1) * args.switch - 1
            if start <= step <= end:
                exp_replay_fast2meta.store(cs, c_action)

        # ---------- sync target ----------
        if step > 0 and step % 1000 == 0:
            Target_net.load_state_dict(Fast_Learner.state_dict())

        # ---------- fast learner update ----------
        if exp_replay_fast.size() >= args.batch_size:
            do_bc = (meta_warmup == 1 and args.lambda_reg > 0
                     and (step % args.switch) < args.warmstep)
            if do_bc:
                train_fast(Fast_Learner, Fast_opt, Target_net, Fast_criterion,
                           exp_replay_fast, gamma, args.lambda_reg,
                           reg=Meta_Learner)
            else:
                train_fast(Fast_Learner, Fast_opt, Target_net, Fast_criterion,
                           exp_replay_fast, gamma, args.lambda_reg, reg=None)

        cs = ns
        if done:
            cs = env.reset()
            avg_return = 0.99 * avg_return + 0.01 * epi_return
            epi_return = 0.0
        returns_array[step] = avg_return

        # ---------- detection feed ----------
        # Compute next_phi lazily (a single extra forward pass on `ns`).
        # This is only done in boundary-free modes that need it.
        if is_boundary_free:
            if detector_kind == "swoks":
                next_phi = None  # ignored by the adapter for swoks
            else:
                with torch.no_grad():
                    _q, next_phi_t = Fast_Learner(
                        _obs_to_tensor(ns, device), return_latent=True
                    )
                next_phi = next_phi_t[0].detach().cpu().numpy()

            fired = detector.step(phi, c_action, rew, next_phi)
            if fired:
                true_switch = oracle_boundaries[-1] if oracle_boundaries else 0
                # Per-detector introspection for the log line.
                core = detector.core
                # Prefer the fire-time snapshot (preserved across reset).
                info = getattr(core, "last_fire_info", {}) or {}
                if detector_kind == "swoks":
                    pval = info.get("pval", core.last_pval)
                    swd = info.get("swd", core.last_swd)
                    detail = f"pval={pval:.2e} swd={swd:.4f}"
                    fire_meta = {"pval": float(pval), "swd": float(swd)}
                elif detector_kind == "implicit":
                    pval = info.get("pval", core.last_pval)
                    score = info.get("score", core.last_score)
                    err_r = info.get("err_r", core.last_err_r)
                    err_d = info.get("err_d", core.last_err_d)
                    detail = (f"pval={pval:.2e} score={score:.3f} "
                              f"err_r={err_r:.3e} err_d={err_d:.3e}")
                    fire_meta = {"pval": float(pval), "score": float(score),
                                 "err_r": float(err_r), "err_d": float(err_d)}
                else:  # hybrid
                    detail = (f"reason={core.last_reason} "
                              f"p_imp={core.last_p_imp:.2e} "
                              f"p_stat={core.last_p_stat:.2e} "
                              f"combined_nats={core.last_combined:.2f}")
                    fire_meta = {"reason": core.last_reason,
                                 "p_imp": float(core.last_p_imp),
                                 "p_stat": float(core.last_p_stat),
                                 "combined_nats": float(core.last_combined)}
                print(f"[{detector_kind} detected @ step {step}] "
                      f"delay={step - true_switch} {detail}")

                # FAME response: train meta on the accumulated fast2meta,
                # then run the detection/warm-up.
                if exp_replay_meta.size() > 0 or exp_replay_fast2meta.size() > 0:
                    print("  running meta update on streamed fast2meta")
                    Meta_opt_local = optim.Adam(Meta_Learner.parameters(),
                                                lr=args.lr1)
                    Meta_scheduler_local = ExponentialLR(Meta_opt_local,
                                                         gamma=0.95)
                    if exp_replay_meta.size() >= args.batch_size:
                        train_meta(Meta_Learner, Meta_opt_local,
                                   Meta_scheduler_local, Meta_criterion2,
                                   exp_replay_meta, exp_replay_fast2meta,
                                   args.epoch_meta, args.batch_size,
                                   max(1, len(oracle_boundaries)), device)
                    exp_replay_fast2meta.copy_to(exp_replay_meta)
                    exp_replay_fast2meta.delete()
                exp_replay_fast.delete()

                fast_snap = pick_prebreak_snapshot(step)
                avg_return = 0.0 if args.reset == 1 else avg_return
                meta_warmup = 0
                out = trigger.run(
                    env=env, gameid=max(1, len(detected_boundaries) + 1),
                    step=step, cs=cs,
                    Fast_Learner=Fast_Learner, Meta_Learner=Meta_Learner,
                    Random_Learner=Random_Learner, Fast_opt=Fast_opt,
                    Target_net=Target_net,
                    exp_replay_fast=exp_replay_fast,
                    returns_array=returns_array, avg_return=avg_return,
                    pbar=pbar, fast_snapshot=fast_snap,
                )
                Fast_Learner = out["Fast_Learner"]
                Fast_opt = out["Fast_opt"]
                Target_net = out["Target_net"]
                cs = out["cs"]
                step = out["step"]
                avg_return = out["avg_return"]
                meta_warmup = out["meta_warmup"]
                flag_history.append(out["flag"])
                detected_boundaries.append(step)
                entry = {
                    "step": step, "true_switch": true_switch,
                    "source": detector_kind,
                    "flag": out["flag"],
                    "delay": step - true_switch,
                }
                entry.update(fire_meta)
                detection_log.append(entry)

        # ---------- end of task in oracle mode ----------
        # (Boundary-free modes update the meta learner at each detected
        #  boundary instead of on a fixed schedule.)
        if is_oracle and (step + 1) % args.switch == 0:
            if step + 1 == args.switch:
                print("first task end: no meta update yet")
            else:
                print("##### meta update (oracle end of task)")
                Meta_opt = optim.Adam(Meta_Learner.parameters(), lr=args.lr1)
                Meta_scheduler = ExponentialLR(Meta_opt, gamma=0.95)
                train_meta(Meta_Learner, Meta_opt, Meta_scheduler,
                           Meta_criterion2, exp_replay_meta,
                           exp_replay_fast2meta, args.epoch_meta,
                           args.batch_size, gameid, device)
            exp_replay_fast2meta.copy_to(exp_replay_meta)
            exp_replay_fast2meta.delete()
            exp_replay_fast.delete()
            if args.save_model:
                os.makedirs(args.models_dir, exist_ok=True)
                torch.save(Meta_Learner.state_dict(),
                           os.path.join(args.models_dir,
                                        f"{filename}_Meta{gameid}.pt"))

        step += 1
        pbar.update(1)

        # ---------- periodic checkpoint ----------
        if (args.checkpoint_dir
                and step - _last_checkpoint_step >= args.checkpoint_interval):
            _elapsed_now = (_elapsed_time_prev_sessions
                            + time.time() - _session_start_time)
            save_checkpoint(
                _checkpoint_path(args.checkpoint_dir, filename),
                step=step,
                gameid=gameid,
                avg_return=avg_return,
                epi_return=epi_return,
                meta_warmup=meta_warmup,
                returns_array=returns_array,
                oracle_boundaries=oracle_boundaries,
                detected_boundaries=detected_boundaries,
                detection_log=detection_log,
                flag_history=flag_history,
                Games=Games,
                snapshots=snapshots,
                Fast_Learner=Fast_Learner,
                Fast_opt=Fast_opt,
                Target_net=Target_net,
                Meta_Learner=Meta_Learner,
                Meta_opt=Meta_opt,
                Meta_scheduler=Meta_scheduler,
                exp_replay_fast=exp_replay_fast,
                exp_replay_fast2meta=exp_replay_fast2meta,
                exp_replay_meta=exp_replay_meta,
                detector=detector,
                elapsed_time_so_far=_elapsed_now,
            )
            _last_checkpoint_step = step

    pbar.close()

    # Compute total training time (accumulated + this session)
    _total_training_time = (_elapsed_time_prev_sessions
                            + time.time() - _session_start_time)
    _total_training_time_str = time.strftime(
        "%H:%M:%S", time.gmtime(_total_training_time)
    )
    print(f"Total training time: {_total_training_time_str} "
          f"({_total_training_time:.1f}s)")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    if args.save:
        os.makedirs(args.results_dir, exist_ok=True)
        out = {
            "returns": returns_array,
            "oracle_boundaries": oracle_boundaries,
            "detected_boundaries": detected_boundaries,
            "detection_log": detection_log,
            "games": Games,
            "flag_history": flag_history,
            "args": vars(args),
            "mode": mode_tag,
            "detector_stats": detector.stats() if detector is not None else None,
            "total_training_time_seconds": _total_training_time,
            "total_training_time_hms": _total_training_time_str,
        }
        out_path = os.path.join(args.results_dir, f"{filename}_returns.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(out, f)
        print(f"Saved -> {out_path}")

    # Always persist the FINAL fast + meta weights when --save-model is set.
    # Experiment.py relies on these for post-hoc policy evaluation
    # (p_i(K*T) across all envs -> Avg. Perf + Forgetting a la FAME paper).
    if args.save_model:
        os.makedirs(args.models_dir, exist_ok=True)
        final_meta_path = os.path.join(args.models_dir,
                                       f"{filename}_MetaFinal.pt")
        final_fast_path = os.path.join(args.models_dir,
                                       f"{filename}_FastFinal.pt")
        torch.save(Meta_Learner.state_dict(), final_meta_path)
        torch.save(Fast_Learner.state_dict(), final_fast_path)
        print(f"Saved final meta -> {final_meta_path}")
        print(f"Saved final fast -> {final_fast_path}")

    print("Flag_Reg:", flag_history)
    print("Games:", Games)
    print("Oracle boundaries:", oracle_boundaries)
    print("Detected boundaries:", detected_boundaries)
    print(f"Total training time: {_total_training_time_str} "
          f"({_total_training_time:.1f}s)")
    print(args)


if __name__ == "__main__":
    main()
