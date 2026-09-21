#!/usr/bin/env python3
"""Start the TeleJev HTTP interface directly from a source checkout."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from telejev.api import main  # noqa: E402

if __name__ == "__main__":
    main()