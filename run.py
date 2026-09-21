#!/usr/bin/env python3
"""Run the TeleJev CUDA scorer directly from a source checkout.

No package installation is required: this adds ``src/`` to the import path and
delegates to the command line scorer.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from telejev.cli import main  # noqa: E402

if __name__ == "__main__":
    main()