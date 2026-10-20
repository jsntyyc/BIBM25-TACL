"""Compatibility entry point: install the namespace package from repository root."""
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    print(f"This task is part of BIBM25-TACL. Install from repository root: python -m pip install --no-deps -e \"{root}\"")
    print("Install the task environment as documented in tasks/icd/README.md; then use python -m tasks.icd.main --help.")
