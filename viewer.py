# viewer.py
"""
Live, colorful chess-board viewer
─────────────────────────────────
Drop‐in replacement for your previous ``viewer.py``.

How it works
============
The training code should push either

• a *chess.Board* instance **or**
• a *uci-string* (e.g. ``"e2e4"``)

onto the shared queue.  
This viewer keeps its own local ``board``.  When it receives a move
(string) it plays the move on that board; when it receives a
``chess.Board`` object it resets its local copy to that position.

After every update a pretty, Unicode-based board is rendered with
ANSI colours plus the last move (in SAN) underneath.

If you run this in a Windows terminal you may need
``colorama.just_fix_windows_console()`` to get colours.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from typing import Any, Optional

import chess
from colorama import Fore, Back, Style, init as _colorama_init

# ── colour initialisation (no-op on real ANSI terminals) ───────────────
_colorama_init()

_WHITE_SQUARE = Back.LIGHTWHITE_EX + "  " + Style.RESET_ALL
_BLACK_SQUARE = Back.LIGHTBLACK_EX + "  " + Style.RESET_ALL

# Unicode pieces keyed by *piece.symbol()*
_PIECE_TO_UNICODE_COLOURED = {
    "P": Fore.BLACK + "♙ " + Style.RESET_ALL,
    "N": Fore.BLACK + "♘ " + Style.RESET_ALL,
    "B": Fore.BLACK + "♗ " + Style.RESET_ALL,
    "R": Fore.BLACK + "♖ " + Style.RESET_ALL,
    "Q": Fore.BLACK + "♕ " + Style.RESET_ALL,
    "K": Fore.BLACK + "♔ " + Style.RESET_ALL,
    "p": Fore.WHITE + "♟︎ " + Style.RESET_ALL,
    "n": Fore.WHITE + "♞ " + Style.RESET_ALL,
    "b": Fore.WHITE + "♝ " + Style.RESET_ALL,
    "r": Fore.WHITE + "♜ " + Style.RESET_ALL,
    "q": Fore.WHITE + "♛ " + Style.RESET_ALL,
    "k": Fore.WHITE + "♚ " + Style.RESET_ALL,
}


def _render_board(board: chess.Board, last_move: Optional[chess.Move] = None) -> str:
    """Return a coloured, Unicode drawing of *board*.  White at bottom."""
    ranks = []
    for rank in range(7, -1, -1):                       # 7 → 0
        squares = []
        for file in range(8):
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)

            # choose square colour
            base = _WHITE_SQUARE if (file + rank) % 2 == 0 else _BLACK_SQUARE
            if piece:
                sym = piece.symbol()
                piece_txt = _PIECE_TO_UNICODE_COLOURED[sym]
                cell = Back.RESET + piece_txt + Style.RESET_ALL
            else:
                cell = base
            squares.append(cell)
        ranks.append(f"{rank + 1} " + "".join(squares) + " ")
    files = "   a b c d e f g h"
    bar = "\n"
    board_str = bar.join(ranks) + bar + files

    if last_move:
        board_str += f"\nLast move: {board.san(last_move)}"
    return board_str


def view_moves(msg_queue: queue.Queue[Any], poll: float = 0.05) -> None:
    """Continuously read queue and render the board beautifully."""
    board = chess.Board()
    last_move: Optional[chess.Move] = None

    while True:
        try:
            msg = msg_queue.get_nowait()
        except queue.Empty:
            time.sleep(poll)
            continue

        # -------- interpret message --------------------------------------
        if isinstance(msg, chess.Board):
            board = msg.copy(stack=False)
            last_move = None

        elif isinstance(msg, (bytes, bytearray)):
            # assume bytes containing UCI
            try:
                move = board.parse_uci(msg.decode().strip())
            except ValueError:
                pass
            else:
                board.push(move)
                last_move = move

        elif isinstance(msg, str):
            try:
                content = json.loads(msg)
            except json.JSONDecodeError:
                # plain UCI?
                try:
                    move = board.parse_uci(msg.strip())
                except ValueError:
                    # ignore unknown message
                    continue
                else:
                    board.push(move)
                    last_move = move
            else:
                # JSON – if it contains FEN, sync board
                if "fen" in content:
                    board = chess.Board(content["fen"])
                    last_move = None

        # Anything else we ignore quietly.

        # -------- render -------------------------------------------------
        print("\033[H\033[J", end="")            # clear screen (ANSI)
        print(_render_board(board, last_move), flush=True)


def start_viewer_thread(msg_queue: queue.Queue[Any]) -> threading.Thread:
    """Start the viewer in a daemon thread so it terminates with main."""
    t = threading.Thread(target=view_moves, args=(msg_queue,), daemon=True)
    t.start()
    return t
