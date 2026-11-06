"""Reconcile archived paired effects, integer sign-flip counts, and Holm values."""
from pathlib import Path
import numpy as np, pandas as pd
from holm_adjustment import holm_adjust
ROOT=Path(__file__).resolve().parents[2]
c=pd.read_csv(ROOT/"05_results/adjacent_inference.csv"); e=pd.read_csv(ROOT/"05_results/paired_adjacent_effects.csv")
assert len(c)==16 and len(e)==80
p=(c.exceedance_count+1)/(c.n_permutations+1)
assert np.allclose(p,c.raw_p,rtol=0,atol=1e-15)
assert np.allclose(holm_adjust(p.tolist()),c.holm_p,rtol=0,atol=1e-15)
print("PASS: 80 seed effects, 16 exact count-based p-values, and Holm adjustment")
