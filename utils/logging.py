from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Optional

def setup_logger(log_dir: str | os.PathLike, name: str = "mossgate", rank: int = 0) -> logging.Logger:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if rank == 0:
        fh = logging.FileHandler(log_dir / "train.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
