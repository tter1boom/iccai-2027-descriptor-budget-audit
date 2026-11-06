"""Holm step-down adjustment for a finite family of raw p-values."""
def holm_adjust(p_values):
 values=[float(p) for p in p_values]; m=len(values); order=sorted(range(m),key=lambda i:values[i]); out=[0.0]*m; running=0.0
 for rank,idx in enumerate(order):
  running=max(running,min(1.0,(m-rank)*values[idx])); out[idx]=running
 return out
