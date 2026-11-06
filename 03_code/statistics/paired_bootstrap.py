"""Paired class-stratified bootstrap for per-image budget effects."""
import numpy as np
def paired_class_stratified_bootstrap(delta_by_image_by_resample,labels,n_draws=20000,seed=20261103):
 d=np.asarray(delta_by_image_by_resample,dtype=float); y=np.asarray(labels)
 if d.ndim!=2 or d.shape[0]!=len(y): raise ValueError('Expected image x resample effects')
 classes=np.unique(y); groups=[np.flatnonzero(y==c) for c in classes]; rng=np.random.default_rng(seed); out=np.empty(n_draws)
 for b in range(n_draws):
  class_means=[]
  for ix in groups:
   pick=rng.choice(ix,size=len(ix),replace=True); class_means.append(d[pick].mean())
  out[b]=np.mean(class_means)
 return out
