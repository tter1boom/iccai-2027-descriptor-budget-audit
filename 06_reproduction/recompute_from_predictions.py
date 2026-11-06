"""Recompute Aircraft macro-accuracy curves from per-image prediction records."""
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]
df=pd.read_csv(ROOT/"04_data/aircraft_prediction_records/predictions.csv")
ref=pd.read_csv(ROOT/"05_results/aircraft_budget_curves.csv")
def calc(g):
 correct=g.predicted_index.eq(g.true_index); by=g.assign(correct=correct).groupby("true_index").correct.mean()
 return pd.Series({"macro_accuracy":by.mean(),"top1_accuracy_secondary":correct.mean(),"n":len(g)})
out=df.groupby(["seed","backbone","scorer","split","k"],sort=True).apply(calc,include_groups=False).reset_index()
keys=["seed","backbone","scorer","split","k"]; j=ref.merge(out,on=keys,suffixes=("_archived","_recomputed"),validate="one_to_one")
assert len(j)==len(ref)==len(out) and np.allclose(j.macro_accuracy_archived,j.macro_accuracy_recomputed,rtol=0,atol=1e-12) and np.array_equal(j.n_archived,j.n_recomputed)
target=ROOT/"05_results/recomputed_aircraft_budget_curves.csv"; out.to_csv(target,index=False)
print(f"PASS: recomputed {len(out)} curve rows from per-image predictions; {target}")
