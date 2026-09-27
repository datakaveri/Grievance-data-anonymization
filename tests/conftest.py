import sys
from pathlib import Path

# Add src/ directory to sys.path so grievance_anonymization package is importable in tests
src_path = str(Path(__file__).resolve().parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)
