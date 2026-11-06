"""Ranking functions for class-specific descriptor similarities."""
import numpy as np
def positive_mean(support_similarity):
 """Return mean support similarity for each descriptor."""
 return np.asarray(support_similarity,dtype=float).mean(axis=0)
def contrastive_margin(own_support_similarity, other_support_similarity):
 """Positive mean minus mean similarity to all other-class support images."""
 return positive_mean(own_support_similarity)-np.asarray(other_support_similarity,dtype=float).mean(axis=0)
def fixed_catalogue_order(n_descriptors=4): return list(range(n_descriptors))
def random_order(n_descriptors=4,seed=0): return np.random.default_rng(seed).permutation(n_descriptors).tolist()
def ranked_indices(scores): return np.argsort(-np.asarray(scores),kind="stable").tolist()
