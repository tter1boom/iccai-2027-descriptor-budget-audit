"""Reconcile stored exact count-based p-values and Holm adjustment.
This is an archived-output verification; it does not regenerate the null draws.
"""
from pathlib import Path
import runpy,sys
s=Path(__file__).resolve().parents[1]/"03_code/statistics/verify_adjacent_budget_tables.py"
sys.path.insert(0,str(s.parent)); runpy.run_path(str(s),run_name="__main__")
