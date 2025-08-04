# main.py

import queue
import threading
import logging

from viewer import view_moves
from trainer import train_ai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

def main():
    logger.info("Creating shared queue")
    q = queue.Queue()

    logger.info("Launching viewer thread")
    t = threading.Thread(target=view_moves, args=(q,), daemon=True)
    t.start()

    logger.info("Launching training")
    train_ai(q)
    logger.info("Training finished")

if __name__ == "__main__":
    main()
