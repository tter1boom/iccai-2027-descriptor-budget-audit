"""Replay cached-feature controls with locally supplied frozen text checkpoints."""
from pathlib import Path
import runpy,sys
s=Path(__file__).resolve().parents[1]/"03_code/experiments/additional_controls.py"
sys.argv=[str(s),*sys.argv[1:]]; runpy.run_path(str(s),run_name="__main__")
