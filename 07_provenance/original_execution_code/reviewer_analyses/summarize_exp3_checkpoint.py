"""Resume Experiment 3 from completed selection records using vectorized bootstrap."""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ORDER=['1','2','5','10','20','all']
def stable_seed(label):return int.from_bytes(hashlib.sha256(('20261004:'+label).encode()).digest()[:8],'little')
def hboot(g,column,reps,label):
 supports=sorted(g.support_seed.unique());perms=sorted(g.validation_permutation.unique());table=g.pivot(index='support_seed',columns='validation_permutation',values=column).loc[supports,perms].to_numpy(float)
 if table.shape!=(20,50) or np.isnan(table).any():raise RuntimeError(f'Expected complete 20x50 table, got {table.shape}')
 rng=np.random.default_rng(stable_seed(label));out=np.empty(reps);chunk=500
 for start in range(0,reps,chunk):
  n=min(chunk,reps-start);s=rng.integers(0,20,(n,20));p=rng.integers(0,50,(n,20,50));out[start:start+n]=table[s[:,:,None],p].mean((1,2))
 return np.quantile(out,[.025,.975])
def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint-dir',type=Path,default=HERE/'checkpoint');p.add_argument('--output-dir',type=Path,default=HERE/'exp3_validation_size');p.add_argument('--replicates',type=int,default=10000);p.add_argument('--overwrite',action='store_true');a=p.parse_args()
 source=a.checkpoint_dir/'all_selections.csv';rankpath=a.checkpoint_dir/'rankings.csv';curvepath=a.checkpoint_dir/'test_curves.csv'
 for x in (source,rankpath,curvepath):
  if not x.is_file():raise FileNotFoundError(x)
 f=pd.read_csv(source);required={'backbone','ranker','support_seed','validation_permutation','n_validation_per_class','oracle_regret','selected_test_macro_accuracy'}
 if not required.issubset(f.columns) or len(f)!=24000:raise RuntimeError('Checkpoint is not the frozen 24,000-selection table')
 a.output_dir.mkdir(parents=True,exist_ok=True);targets=[a.output_dir/x for x in ['regret_by_validation_size.csv','ranking_stability.csv','selected_k_distribution.csv','figure_regret_vs_nV.pdf','summary_manifest.json','PROGRESS.txt']]
 if not a.overwrite and any(x.exists() for x in targets):raise FileExistsError('Summary output exists; use --overwrite or a new directory')
 progress=targets[-1];progress.write_text('stage=hierarchical_bootstrap status=running groups=0/24\n',encoding='utf-8');rows=[];start=time.time()
 groups=list(f.groupby(['backbone','ranker','n_validation_per_class'],sort=False))
 for index,(keys,g) in enumerate(groups,1):
  row=dict(backbone=keys[0],ranker=keys[1],n_validation_per_class=keys[2],mean_regret=g.oracle_regret.mean(),median_regret=g.oracle_regret.median(),p95_regret=g.oracle_regret.quantile(.95),p_regret_zero=(g.oracle_regret==0).mean(),p_regret_gt_0_5pp=g.regret_gt_0_5pp.mean(),p_regret_gt_1pp=g.regret_gt_1pp.mean(),mean_selected_test_accuracy=g.selected_test_macro_accuracy.mean(),mean_fixed_k1=g.fixed_k1_test_macro_accuracy.mean(),mean_fixed_k4=g.fixed_k4_test_macro_accuracy.mean(),mean_oracle=g.oracle_test_macro_accuracy.mean(),p_exact_oracle_k=g.exact_oracle_k.mean())
  for metric in ['oracle_regret','selected_test_macro_accuracy']:
   lo,hi=hboot(g,metric,a.replicates,'|'.join(map(str,keys))+'|'+metric);row[metric+'_ci95_low']=lo;row[metric+'_ci95_high']=hi
  rows.append(row);progress.write_text(f'stage=hierarchical_bootstrap status=running groups={index}/24 elapsed_seconds={time.time()-start:.1f}\n',encoding='utf-8')
 sm=pd.DataFrame(rows);sm.to_csv(targets[0],index=False);f.groupby(['backbone','ranker','n_validation_per_class','selected_k']).size().rename('count').reset_index().to_csv(targets[2],index=False)
 ranks=pd.read_csv(rankpath);st=[]
 for (b,r,c),g in ranks.groupby(['backbone','ranker','class_id']):
  top=g[g['rank'].eq(1)].descriptor_id;st.append(dict(backbone=b,ranker=r,class_id=c,top1_modal_agreement=top.value_counts().max()/len(top)))
 pd.DataFrame(st).to_csv(targets[1],index=False)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(6.5,4))
 for key,g in sm.groupby(['backbone','ranker']):g=g.assign(_order=g.n_validation_per_class.astype(str).map({v:i for i,v in enumerate(ORDER)})).sort_values('_order');ax.plot(ORDER,100*g.mean_regret,marker='o',label=' | '.join(key))
 ax.set(xlabel='Validation images per class',ylabel='Mean oracle regret (pp)');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(targets[3]);plt.close(fig)
 manifest={'status':'complete','checkpoint_rows':len(f),'hierarchical_bootstrap_replicates':a.replicates,'elapsed_seconds':time.time()-start,'method':'support-resample outer level; validation-permutation inner level; vectorized equivalent of frozen protocol'};targets[4].write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8');progress.write_text(f"stage=complete status=PASS elapsed_seconds={manifest['elapsed_seconds']:.1f}\n",encoding='utf-8');print(a.output_dir.resolve())
if __name__=='__main__':main()
