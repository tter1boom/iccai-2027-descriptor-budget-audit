"""Nested bootstrap: outer support clusters, inner validation permutations."""
import numpy as np
def nested_support_validation_bootstrap(values,n_boot=10000,seed=20261104):
 x=np.asarray(values,dtype=float)
 if x.ndim!=2: raise ValueError('Expected support x nested validation-permutation matrix')
 n_support,n_perm=x.shape; rng=np.random.default_rng(seed); out=np.empty(n_boot)
 for b in range(n_boot):
  outer=rng.integers(0,n_support,size=n_support); sample=[]
  for s in outer:
   inner=rng.integers(0,n_perm,size=n_perm); sample.extend(x[s,inner])
  out[b]=np.mean(sample)
 return {'replicates':out,'ci95_percentile':np.quantile(out,[0.025,0.975]),'n_support':n_support,'n_permutations_per_support':n_perm}
