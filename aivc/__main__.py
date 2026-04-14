"""aivc.__main__ — enables `python -m aivc <command>`."""
import os
import sys

# Ensure repo root is importable so `import cli` resolves.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from cli import app  # noqa: E402

if __name__ == "__main__":
    app()
