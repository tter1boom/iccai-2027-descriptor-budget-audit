"""Experiment 3: Aircraft validation-size and oracle-regret analysis (CPU).

Reuses the corrected-clean image feature caches. Twenty support resamples and
50 nested validation permutations are frozen below. Test labels are used only
after each validation-selected k is fixed, to calculate evaluation metrics.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from aircraft_cached_core import (ROOT,BACKBONES,SCORERS,load_context,descriptor_features,
 rank_descriptors,build_recomposed_prompts,encode_text,predict,macro_accuracy,official_ids,sha256)

SUPPORT_SEEDS=tuple(range(20261011,20261031)); PERMUTATION_SEEDS=tuple(range(20261101,20261151))
VALIDATION_SIZES=(1,2,5,10,20,'all')

def parse():
 p=argparse.ArgumentParser(); p.add_argument('--output-dir',type=Path,default=ROOT/'revision_experiments_v13/exp3_validation_size'); p.add_argument('--hierarchical-bootstrap',type=int,default=10000); p.add_argument('--preflight',action='store_true'); p.add_argument('--overwrite',action='store_true'); return p.parse_args()

def class_sample(ids,labels,n,rng):
 return [x for c in range(100) for x in rng.choice(np.asarray(ids)[labels==c],size=n,replace=False).tolist()]

def hboot(frame,value,reps,seed):
 rng=np.random.default_rng(seed); supports=sorted(frame.support_seed.unique()); permutations=sorted(frame.validation_permutation.unique())
 table=frame.pivot(index='support_seed',columns='validation_permutation',values=value).loc[supports,permutations].to_numpy(dtype=np.float64)
 if table.shape!=(20,50) or np.isnan(table).any():raise RuntimeError(f'Incomplete hierarchical table for {value}: {table.shape}')
 out=np.empty(reps,dtype=np.float64);chunk=500
 for start in range(0,reps,chunk):
  n=min(chunk,reps-start);support_draw=rng.integers(0,20,size=(n,20));permutation_draw=rng.integers(0,50,size=(n,20,50));sampled=table[support_draw[:,:,None],permutation_draw];out[start:start+n]=sampled.mean(axis=(1,2))
 return np.quantile(out,[.025,.975])

def main():
 a=parse(); files=[ROOT/'external_aircraft_corrected_clean/feature_cache'/f'image_features_{b}.npz' for b in BACKBONES]
 if any(not p.is_file() for p in files):raise FileNotFoundError('Corrected-clean Aircraft feature cache missing')
 print('PASS: Experiment 3 preflight; 20 support seeds, 50 nested validation permutations, nV=1,2,5,10,20,all')
 if a.preflight:return
 outputs=[a.output_dir/x for x in ['all_selections.csv','regret_by_validation_size.csv','ranking_stability.csv','selected_k_distribution.csv','figure_regret_vs_nV.pdf']]
 if not a.overwrite and any(x.exists() for x in outputs):raise FileExistsError('Output exists')
 a.output_dir.mkdir(parents=True,exist_ok=True); selections=[]; rankrows=[]; curve_rows=[]
 for backbone in BACKBONES:
  ctx=load_context(backbone); attrs=descriptor_features(ctx); train_ids=official_ids('train'); val_ids=official_ids('val'); test_ids=official_ids('test')
  train_labels=np.asarray([ctx['labels'][ctx['positions'][i]] for i in train_ids]); val_labels=np.asarray([ctx['labels'][ctx['positions'][i]] for i in val_ids]); test_labels=np.asarray([ctx['labels'][ctx['positions'][i]] for i in test_ids])
  val_pos=torch.tensor([ctx['positions'][i] for i in val_ids]); test_pos=torch.tensor([ctx['positions'][i] for i in test_ids])
  for support_seed in SUPPORT_SEEDS:
   support=class_sample(train_ids,train_labels,5,np.random.default_rng(support_seed))
   for scorer in SCORERS:
    orders=rank_descriptors(ctx,support,scorer,attrs)
    for c,o in enumerate(orders):
     for rank,j in enumerate(o,1):rankrows.append(dict(backbone=backbone,ranker=scorer,support_seed=support_seed,class_id=c,rank=rank,descriptor_id=j+1))
    val_pred={}; test_acc={}
    for k in range(5):
     text=encode_text(ctx,build_recomposed_prompts(ctx['classes'],orders,k)); vp,_=predict(ctx['features'][val_pos],text); tp,_=predict(ctx['features'][test_pos],text)
     val_pred[k]=vp; test_acc[k]=macro_accuracy(test_labels,tp)
     curve_rows.append(dict(backbone=backbone,ranker=scorer,support_seed=support_seed,k=k,test_macro_accuracy=test_acc[k]))
    oracle_k=min(k for k,v in test_acc.items() if v==max(test_acc.values())); oracle_acc=test_acc[oracle_k]; second=sorted(test_acc.values(),reverse=True)[1]
    by_class={c:np.flatnonzero(val_labels==c) for c in range(100)}
    for perm_index,perm_seed in enumerate(PERMUTATION_SEEDS):
     rng=np.random.default_rng(support_seed*1000003+perm_seed); perms={c:rng.permutation(idx) for c,idx in by_class.items()}
     for nv in VALIDATION_SIZES:
      indices=np.concatenate([perms[c] if nv=='all' else perms[c][:int(nv)] for c in range(100)])
      va={k:macro_accuracy(val_labels[indices],p[indices]) for k,p in val_pred.items()}; selected=min(k for k,v in va.items() if v==max(va.values())); acc=test_acc[selected]; regret=oracle_acc-acc
      selections.append(dict(backbone=backbone,ranker=scorer,support_seed=support_seed,validation_permutation=perm_index,validation_permutation_seed=perm_seed,n_validation_per_class=str(nv),selected_k=selected,selected_validation_macro_accuracy=va[selected],selected_test_macro_accuracy=acc,fixed_k1_test_macro_accuracy=test_acc[1],fixed_k4_test_macro_accuracy=test_acc[4],oracle_k=oracle_k,oracle_test_macro_accuracy=oracle_acc,oracle_regret=regret,oracle_gap_best_minus_second=oracle_acc-second,exact_oracle_k=selected==oracle_k,regret_gt_0_5pp=regret>.005,regret_gt_1pp=regret>.01))
  del ctx
 allsel=pd.DataFrame(selections); allsel.to_csv(outputs[0],index=False); pd.DataFrame(rankrows).to_csv(a.output_dir/'rankings.csv',index=False); pd.DataFrame(curve_rows).to_csv(a.output_dir/'test_curves.csv',index=False)
 summary=[]
 for keys,g in allsel.groupby(['backbone','ranker','n_validation_per_class'],sort=False):
  row=dict(backbone=keys[0],ranker=keys[1],n_validation_per_class=keys[2],mean_regret=g.oracle_regret.mean(),median_regret=g.oracle_regret.median(),p95_regret=g.oracle_regret.quantile(.95),p_regret_zero=(g.oracle_regret==0).mean(),p_regret_gt_0_5pp=g.regret_gt_0_5pp.mean(),p_regret_gt_1pp=g.regret_gt_1pp.mean(),mean_selected_test_accuracy=g.selected_test_macro_accuracy.mean(),mean_fixed_k1=g.fixed_k1_test_macro_accuracy.mean(),mean_fixed_k4=g.fixed_k4_test_macro_accuracy.mean(),mean_oracle=g.oracle_test_macro_accuracy.mean(),p_exact_oracle_k=g.exact_oracle_k.mean())
  for metric in ['oracle_regret','selected_test_macro_accuracy']:
   ci=hboot(g,metric,a.hierarchical_bootstrap,20261004+sum(map(ord,''.join(keys)))); row[metric+'_ci95_low'],row[metric+'_ci95_high']=ci
  summary.append(row)
 sm=pd.DataFrame(summary); sm.to_csv(outputs[1],index=False)
 allsel.groupby(['backbone','ranker','n_validation_per_class','selected_k']).size().rename('count').reset_index().to_csv(outputs[3],index=False)
 ranks=pd.DataFrame(rankrows); stability=[]
 for (b,r,c),g in ranks.groupby(['backbone','ranker','class_id']):
  top=g[g['rank'].eq(1)].descriptor_id; stability.append(dict(backbone=b,ranker=r,class_id=c,top1_modal_agreement=top.value_counts().max()/len(top)))
 pd.DataFrame(stability).to_csv(outputs[2],index=False)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(6.5,4)); order=['1','2','5','10','20','all']
 for key,g in sm.groupby(['backbone','ranker']):g=g.set_index('n_validation_per_class').loc[order];ax.plot(order,100*g.mean_regret,marker='o',label=' | '.join(key))
 ax.set(xlabel='Validation images per class',ylabel='Mean oracle regret (pp)');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(outputs[4]);plt.close(fig);print(a.output_dir.resolve())
if __name__=='__main__':main()
