# trainer.py  (use this to replace your current trainer.py)

import queue
import logging

import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPO

import chess_env  # our padded, float32 observation env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("trainer")


def train_ai(shared_queue: queue.Queue[str]):
    # 1) Inject shared queue for live streaming
    chess_env.QUEUE = shared_queue
    logger.info("Injected QUEUE into env module")

    # 2) Start (or re-use) local Ray
    ray.init(ignore_reinit_error=True, log_to_driver=True)

    # 3) Register single-agent chess environment
    register_env("chess_single", lambda cfg: chess_env.ChessEnvSingle())
    logger.info("Registered env 'chess_single'")

    # 4) Build PPO configuration (legacy stack, so no warnings)
    config = {
        "env": "chess_single",
        "framework": "torch",
        "num_gpus": 1,               # use your RTX 4070
        "num_workers": 0,            # single-process rollouts
        "model": {                   # legacy ModelV2 API
            "conv_filters": [
                [32, [8, 8], 4],
                [64, [4, 4], 2],
                [64, [3, 3], 1],
            ],
            "fcnet_hiddens": [128, 128],
        },
        "train_batch_size": 2000,
        "sgd_minibatch_size": 256,
        "num_sgd_iter": 10,
        "gamma": 0.99,
    }

    # ---- ONE extra line to silence the new-stack warnings ----
    config["api_stack"] = {
        "enable_rl_module_and_learner": False,
        "enable_env_runner_and_connector_v2": False,
    }
    # ----------------------------------------------------------

    # 5) Build PPO algorithm
    algo = PPO(config=config)
    logger.info("Built PPO algorithm")

    # 6) Simple training loop
    for i in range(1, 100000):
        result = algo.train()
        if i % 10 == 0:
            logger.info(
                f"Iter {i:>3}  avg_reward={result['episode_reward_mean']:.3f}  "
                f"timesteps_total={result['timesteps_total']}"
            )

    algo.stop()
    logger.info("Training complete")


if __name__ == "__main__":
    q = queue.Queue()
    train_ai(q)
