# chess_env.py
# ----------------------------------------------------------
import random
import chess
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import json

QUEUE = None  # will be injected by trainer.py


class ChessEnvSingle(gym.Env):
    """Discrete-action single-agent chess environment with padded planes."""

    def __init__(self):
        super().__init__()

        # Action space: 4672 = 64 from-squares × 73 to-square/under-promo options
        self.action_space = spaces.Discrete(4672)

        # Observation: 8×8×111 bool planes → float32 padded to 12×12 so RLlib is happy
        self.observation_space = spaces.Box(0.0, 1.0, (10, 10, 111), dtype=np.float32)

        self.board = chess.Board()

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _action_to_uci(action: int) -> str:
        """Very simple, slightly hacky mapping → UCI string."""
        from_sq = action // 73
        to_idx  = action % 73
        to_sq   = (to_idx % 64)
        promo   = ["", "q", "r", "b", "n"][to_idx // 64]  # 0-3 promotion pieces + ''
        return chess.square_name(from_sq) + chess.square_name(to_sq) + promo

    def _legal_random(self) -> chess.Move:
        """Choose a random legal move (board already checked to have moves)."""
        return random.choice(list(self.board.legal_moves))

    # ----------------------------------------------------------------- Gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.board.reset()
        return self._obs(), {}

    def step(self, action):
        # -------- Agent move --------------------------------------------------
        desired_uci = self._action_to_uci(int(action))
        try:
            move = chess.Move.from_uci(desired_uci)
            if move not in self.board.legal_moves:
                raise ValueError("illegal")  # force except path below
        except Exception:  # malformed or illegal
            move = self._legal_random()

        self.board.push(move)

        # -------- Optional “viewer” stream ------------------------------------
        if QUEUE is not None:
            QUEUE.put(json.dumps({
                "game": 0,                       # or self.game_id if you have one
                "fen": self.board.fen(),         # current board position
                "move": desired_uci,             # move just played
                "result": None                   # fill when game is over
            }))

        # -------- Opponent (very weak) ----------------------------------------
        if not self.board.is_game_over():
            self.board.push(self._legal_random())

        # -------- Compute reward & done flags ---------------------------------
        terminated = self.board.is_game_over()
        truncated  = False  # -- no max-turn cap yet

        # win=+1, loss=-1, draw=0
        result_reward = {
            "1-0": 1.0,
            "0-1": -1.0,
            "1/2-1/2": 0.0
        }
        reward = result_reward.get(self.board.result(), 0.0)

        return self._obs(), reward, terminated, truncated, {}

    # ----------------------------------------------------------------- internal
    def _obs(self):
        """Return 12×12×111 float32 planes (zero-padded from 8×8)."""
        planes = np.zeros((10, 10, 111), dtype=np.float32)
        # 👉 quick and dirty: just place 8×8 slice into the middle
        raw = self._raw_planes()  # 8×8×111 bool
        planes[2:10, 2:10, :] = raw
        return planes

    @staticmethod
    def _raw_planes():
        # minimal placeholder — implement your own board-to-planes function
        return np.zeros((8, 8, 111), dtype=np.float32)
