# trainer.py
# PPO training with a custom CNN (no VisionNet shape fuss), tqdm progress bar,
# versioned checkpoints, PERIODIC AUTOSAVES, "best-so-far" saving,
# and ALWAYS save artifacts on finish/interrupt into models/version_n.

import json
import logging
import signal
import threading
from pathlib import Path

import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch
import torch.nn as nn

import chess_env  # our env with QUEUE injection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("trainer")

# ----------------------- JSON sanitize helper -----------------------
def _json_sanitize(obj):
    """Make RLlib configs JSON-serializable."""
    import numpy as np
    from pathlib import Path as _Path
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (list, tuple, set)):
        return [_json_sanitize(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, _Path):
        return str(obj)
    if isinstance(obj, type):  # classes (ABCMeta)
        return f"{obj.__module__}.{obj.__name__}"
    if callable(obj):
        name = getattr(obj, "__name__", obj.__class__.__name__)
        mod = getattr(obj, "__module__", "builtins")
        return f"{mod}.{name}"
    return str(obj)

# ----------------------- signal handling -----------------------
_STOP = threading.Event()
def _handle_signal(signum, _frame):
    logger.warning(f"Received signal {signum}; will stop after this iteration.")
    _STOP.set()
signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)

# ----------------------- tqdm (optional) -----------------------
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

# ----------------------- custom CNN model ----------------------
class ChessCNN(TorchModelV2, nn.Module):
    """
    Simple CNN for (H,W,C) inputs; uses AdaptiveAvgPool2d(1) so we never worry about
    spatial sizes. Outputs policy logits (action_space.n) and a scalar value head.
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        in_ch = int(obs_space.shape[-1])  # expect (10,10,111)
        self.feat = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),   nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # -> [B, 128, 1, 1]
            nn.Flatten(),                  # -> [B, 128]
        )
        self.policy = nn.Linear(128, action_space.n)
        self.value_branch = nn.Linear(128, 1)
        self._value = torch.tensor(0.0)

    def forward(self, input_dict, state, seq_lens):
        # obs comes in NHWC; convert to NCHW
        x = input_dict["obs"]
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.permute(0, 3, 1, 2).float()
        feats = self.feat(x)
        logits = self.policy(feats)
        self._value = self.value_branch(feats).squeeze(1)
        return logits, state

    def value_function(self):
        return self._value

# ----------------------- versioned saving ----------------------
def _next_version_dir(root: Path = Path("models")) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    n = 0
    while True:
        d = root / f"version_{n}"
        if not d.exists():
            d.mkdir(parents=True, exist_ok=False)
            return d
        n += 1

def _save_artifacts(algo: PPO, config_dict: dict | None, version_dir: Path) -> None:
    """
    Save:
      - RLlib checkpoint (resume-able)
      - lightweight policy weights (PyTorch state_dict)
      - training config JSON (sanitized; falls back to TXT on failure)
    """
    ckpt_dir = version_dir / "rllib_checkpoint"
    ckpt_dir.mkdir(exist_ok=True)
    res = algo.save(str(ckpt_dir))  # Ray may return str or a TrainingResult-like object
    # Extract a clean path string without huge metrics spam:
    try:
        if hasattr(res, "checkpoint"):
            cp = res.checkpoint
            ckpt_path_str = getattr(cp, "path", str(cp))
        elif isinstance(res, str):
            ckpt_path_str = res
        else:
            ckpt_path_str = str(res)
    except Exception:
        ckpt_path_str = str(res)
    logger.info(f"Saved checkpoint to {ckpt_path_str}")

    torch.save(
        algo.get_policy().model.state_dict(),        # lightweight inference weights
        version_dir / "policy_state_dict.pt"
    )
    logger.info(f"Saved weights to   {version_dir/'policy_state_dict.pt'}")

    if config_dict is not None:
        try:
            sanitized = _json_sanitize(config_dict)
            (version_dir / "train_config.json").write_text(
                json.dumps(sanitized, indent=2, ensure_ascii=False)
            )
            logger.info(f"Saved config to    {version_dir/'train_config.json'}")
        except Exception as e:
            (version_dir / "train_config.txt").write_text(repr(config_dict))
            logger.warning(f"Config JSON dump failed; wrote repr to train_config.txt: {e}")

# ----------------------- config builder (version-robust) ----------------------
def _build_cfg():
    """Build a PPOConfig that works across recent RLlib versions, on the classic stack."""
    base = (
        PPOConfig()
        .environment("chess_single")
        .framework("torch")
        .resources(num_gpus=1)
    )

    # Register and use our custom model; avoids VisionNet conv shape constraints.
    ModelCatalog.register_custom_model("chess_cnn", ChessCNN)
    model_cfg = {
        "custom_model": "chess_cnn",
        "vf_share_layers": False,
    }

    # Name differences across RLlib versions handled here:
    try:
        cfg = base.training(
            model=model_cfg,
            train_batch_size=2000,
            sgd_minibatch_size=256,
            num_sgd_iter=10,
            gamma=0.99,
        )
    except TypeError:
        cfg = base.training(
            model=model_cfg,
            train_batch_size=2000,
            minibatch_size=256,
            num_epochs=10,
            gamma=0.99,
        )

    # Prefer env_runners when present; else rollouts
    if hasattr(cfg, "env_runners"):
        cfg = cfg.env_runners(num_env_runners=0)
    else:
        cfg = cfg.rollouts(num_rollout_workers=0)

    # Try to disable the new stack if supported (quiet some warnings)
    try:
        cfg = cfg.api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
    except Exception:
        pass

    return cfg

# ----------------------- main entry ----------------------------
def train_ai(shared_queue):
    # 1) hook up viewer queue
    chess_env.QUEUE = shared_queue
    logger.info("Injected QUEUE into env module")

    # 2) Ray + env
    ray.init(ignore_reinit_error=True, log_to_driver=True)
    register_env("chess_single", lambda cfg: chess_env.ChessEnvSingle())
    logger.info("Registered env 'chess_single'")

    # 3) Build PPO (robust to RLlib version diffs)
    cfg = _build_cfg()
    algo = cfg.build_algo() if hasattr(cfg, "build_algo") else cfg.build()
    logger.info("Built PPO algorithm")

    # 4) versioned output dir
    version_dir = _next_version_dir(Path("models"))
    logger.info(f"Output version directory: {version_dir.resolve()}")

    # Prepare "best-so-far" and autosave dirs
    best_avg = None
    best_dir = version_dir / "best"
    best_dir.mkdir(exist_ok=True)
    autosave_every = 25  # iterations

    # 5) training loop with progress bar
    pbar = None
    if tqdm is not None:
        try:
            pbar = tqdm(
                desc=f"Training → {version_dir.name}",
                unit="iter",
                dynamic_ncols=True,
                mininterval=0.5,
                leave=True,
                disable=False,
            )
        except Exception:
            pbar = None

    i = 0
    try:
        while not _STOP.is_set():
            i += 1
            result = algo.train()
            avg = (
                result.get("episode_reward_mean")
                if result.get("episode_reward_mean") is not None
                else result.get("episode_return_mean", 0.0)
            )
            steps = (
                result.get("timesteps_total")
                or result.get("env_steps_sampled")
                or result.get("num_env_steps_sampled")
                or 0
            )

            # progress bar / logging
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(avg_reward=f"{float(avg):.3f}", steps=int(steps))
            elif i % 10 == 0:
                logger.info(f"Iter {i:>4}  avg_reward={float(avg):.3f}  steps={int(steps)}")

            # ---- Best-so-far saver ----
            try:
                favg = float(avg)
                if (best_avg is None) or (favg > best_avg):
                    best_avg = favg
                    _save_artifacts(
                        algo,
                        cfg.to_dict() if hasattr(cfg, "to_dict") else {},
                        best_dir
                    )
            except Exception as e:
                logger.warning(f"Best-save failed (ignored): {e}")

            # ---- Periodic autosave snapshots ----
            if i % autosave_every == 0:
                try:
                    snap_dir = version_dir / f"autosave_{i:05d}"
                    snap_dir.mkdir(exist_ok=True)
                    _save_artifacts(
                        algo,
                        cfg.to_dict() if hasattr(cfg, "to_dict") else {},
                        snap_dir
                    )
                except Exception as e:
                    logger.warning(f"Autosave at iter {i} failed (ignored): {e}")

    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt received.")
    finally:
        if pbar is not None:
            pbar.close()
        # always save on finish/interrupt
        try:
            config_dict = cfg.to_dict()
        except Exception:
            config_dict = {}
        _save_artifacts(algo, config_dict, version_dir)
        algo.stop()
        try:
            ray.shutdown()
        except Exception:
            pass
        logger.info(f"Training complete. Artifacts saved in: {version_dir.resolve()}")

if __name__ == "__main__":
    import queue
    q = queue.Queue()
    train_ai(q)
