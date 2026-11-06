"""Classification metrics used in the manuscript."""
import numpy as np
def macro_accuracy(truth,predicted,n_classes=100):
 y=np.asarray(truth); p=np.asarray(predicted)
 return float(np.mean([np.mean(p[y==c]==c) for c in range(n_classes) if np.any(y==c)]))
