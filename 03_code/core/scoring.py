"""Cosine score aggregation for joint and separate prompt interfaces."""
import numpy as np
def joint_scores(image_features,text_prototypes): return np.asarray(image_features)@np.asarray(text_prototypes).T
def mean_descriptor_scores(image_features,descriptor_features):
 sims=np.asarray(image_features)@np.asarray(descriptor_features).T
 return sims.mean(axis=-1)
def normalized_mean_prototype(descriptor_features):
 v=np.asarray(descriptor_features).mean(axis=0); n=np.linalg.norm(v,axis=-1,keepdims=True)
 return v/np.maximum(n,1e-12)
