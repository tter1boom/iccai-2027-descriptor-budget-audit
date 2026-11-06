"""Image-block sign-flip test preserving the five-resample block per image."""
import numpy as np
def class_balanced_statistic(delta_by_image,labels,signs=None):
 d=np.asarray(delta_by_image,dtype=float); y=np.asarray(labels); s=np.ones_like(d) if signs is None else np.asarray(signs,dtype=float)
 return float(np.mean([np.mean((d*s)[y==c]) for c in np.unique(y)]))
def signflip_test(delta_by_image_by_resample,labels,n_flips=100000,seed=11789901110311654668,chunk=256):
 d=np.asarray(delta_by_image_by_resample,dtype=float); y=np.asarray(labels)
 if d.ndim!=2 or d.shape[0]!=len(y): raise ValueError('Expected image x support-resample effects')
 mean_d=d.mean(axis=1); observed=class_balanced_statistic(mean_d,y); rng=np.random.default_rng(seed); exceed=0; done=0
 while done<n_flips:
  n=min(chunk,n_flips-done); signs=rng.integers(0,2,size=(n,len(y)),dtype=np.int8)*2-1
  stats=np.asarray([class_balanced_statistic(mean_d,y,row) for row in signs]); exceed+=int(np.sum(np.abs(stats)>=abs(observed))); done+=n
 return {'observed_statistic':observed,'n_permutations':int(n_flips),'exceedance_count':exceed,'raw_p':(exceed+1)/(n_flips+1)}
