"""Legacy blueprint entrypoint. Use the root-level process_dataset.py instead."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from process_dataset import main as root_main


if __name__ == "__main__":
    Path("logs").mkdir(exist_ok=True)
    sys.exit(asyncio.run(root_main()))
