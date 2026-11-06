"""List the available paper-facing CSV outputs."""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
for n in ["aircraft_budget_curves.csv","fixed_order_control.csv","random_ranking_control.csv","class_name_baseline.csv","normalized_ensemble.csv","paired_budget_table.csv","regret_distribution.csv","aircraft_classwise_deltas.csv","adjacent_inference.csv"]:
 p=ROOT/"05_results"/n
 if not p.is_file(): raise FileNotFoundError(p)
 print(f"{n}: {len(pd.read_csv(p))} rows")
