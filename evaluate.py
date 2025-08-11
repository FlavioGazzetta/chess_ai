# evaluate.py
# Load a saved PPO checkpoint and run evaluation games in ChessEnvSingle.
# Works across RLlib versions; registers the custom model used in training.
# Usage examples:
#   python evaluate.py --version models/version_8 --episodes 20
#   python evaluate.py --checkpoint models/version_8/best/rllib_checkpoint/checkpoint_000001

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

import ray
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch
import torch.nn as nn

import chess_env  # same env as training


# ----------------------- custom model (must be registered before load) -------
class ChessCNN(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        in_ch = int(obs_space.shape[-1])  # (H,W,C)
        self.feat = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),   nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.policy = nn.Linear(128, action_space.n)
        self.value_branch = nn.Linear(128, 1)
        self._value = torch.tensor(0.0)

    def forward(self, input_dict, state, seq_lens):
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


# ----------------------- helpers --------------------------------------------
def _versions_sorted(models_root: Path) -> list[Path]:
    vers = []
    for p in models_root.iterdir():
        if p.is_dir() and p.name.startswith("version_"):
            try:
                n = int(p.name.split("_", 1)[1])
                vers.append((n, p))
            except Exception:
                pass
    vers.sort(key=lambda t: t[0])
    return [p for _, p in vers]

def _read_model_name(vdir: Path) -> str | None:
    for cfg_path in [vdir / "best" / "train_config.json", vdir / "train_config.json"]:
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text())
                model = cfg.get("model", {})
                return model.get("custom_model")
            except Exception:
                pass
    return None

def _auto_pick_version(models_root: Path) -> Path:
    versions = _versions_sorted(models_root)
    if not versions:
        raise SystemExit("No models found under ./models. Train first.")
    # Prefer runs that used our custom model; otherwise newest by index
    preferred = [v for v in versions if _read_model_name(v) == "chess_cnn"]
    return (preferred or versions)[-1]

def _find_checkpoint_dir(base: Path) -> Path:
    """Return an inner 'checkpoint_*' dir if present; else return base itself."""
    if not base.exists():
        raise FileNotFoundError(base)
    subs = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith("checkpoint")])
    return subs[-1] if subs else base

def _resolve_checkpoint_path(version_dir: Path) -> Path:
    candidates = [
        version_dir / "best" / "rllib_checkpoint",
        version_dir / "rllib_checkpoint",
    ]
    for c in candidates:
        if c.exists():
            return _find_checkpoint_dir(c)
    raise FileNotFoundError(f"No rllib_checkpoint found in {version_dir}")

def _load_algo_from_path(path: Path):
    """Load PPO from a local checkpoint path with a proper file:// URI."""
    # Register our custom model BEFORE loading so RLlib can reconstruct it
    ModelCatalog.register_custom_model("chess_cnn", ChessCNN)
    # Some Ray builds require a URI scheme; use absolute file://
    uri = path.resolve().as_uri()
    return PPO.from_checkpoint(uri)


# ----------------------- eval loop ------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=str, default=None,
                    help="Path to models/version_N directory (auto-picks latest chess_cnn if omitted).")
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="Direct path to a checkpoint directory/file. Overrides --version.")
    ap.add_argument("--episodes", type=int, default=20, help="Number of evaluation games.")
    ap.add_argument("--seed", type=int, default=None, help="Optional RNG seed.")
    args = ap.parse_args()

    # Ray headless init
    ray.init(ignore_reinit_error=True, log_to_driver=False)

    # Register env (no viewer queue in eval)
    chess_env.QUEUE = None
    register_env("chess_single", lambda cfg: chess_env.ChessEnvSingle())

    # Resolve which checkpoint to load
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        vdir = Path(args.version) if args.version else _auto_pick_version(Path("models"))
        ckpt_path = _resolve_checkpoint_path(vdir)

    print(f"[eval] Loading checkpoint: {ckpt_path.resolve()}")
    algo = _load_algo_from_path(ckpt_path)

    # Evaluate
    env = chess_env.ChessEnvSingle()
    if args.seed is not None and hasattr(env, "seed"):
        try:
            env.seed(args.seed)
        except Exception:
            pass

    returns = []
    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        done = False
        total = 0.0
        steps = 0
        while not done:
            action = algo.compute_single_action(obs, explore=False)
            obs, r, terminated, truncated, _ = env.step(action)
            total += float(r)
            steps += 1
            done = terminated or truncated
        returns.append(total)
        print(f"[eval] Episode {ep:03d}: return={total:+.1f} steps={steps}")

    mean = float(np.mean(returns)) if returns else 0.0
    std = float(np.std(returns)) if returns else 0.0
    wins = sum(1 for x in returns if x > 0)
    losses = sum(1 for x in returns if x < 0)
    draws = len(returns) - wins - losses
    print("\n[evaluate] --------------------------------")
    print(f"Episodes       : {len(returns)}")
    print(f"Mean return    : {mean:+.3f} ± {std:.3f}")
    print(f"Results (W/D/L): {wins}/{draws}/{losses}")

    # Cleanup
    try:
        algo.stop()
    except Exception:
        pass
    try:
        ray.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()
