"""Experiment 5: bundled prompt/aggregation configurations and AttributeSelect reference.

The primary comparison fixes the clean Aircraft resource, frozen support IDs,
positive-mean ranking, k=1..4, backbone, and test set. Prompt context and score
aggregation jointly vary across the compared configurations, so the contrast
does not isolate a causal effect of one interface component. A separate
global-pool reference implements the AttributeSelect selection-and-voting
principle (AS-Mean and multinomial AS-Probe); it is not claimed as a reproduction
of the original large-pool ImageNet protocol.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from aircraft_cached_core import (ROOT,BACKBONES,load_context,descriptor_features,rank_descriptors,
 build_recomposed_prompts,encode_text,predict,macro_accuracy,official_ids,manifest_support)

SEEDS=(20261003,20261004,20261005,20261006,20261007)
def parse():
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=ROOT/'revision_experiments_v13/exp5_matched_prior');p.add_argument('--preflight',action='store_true');p.add_argument('--overwrite',action='store_true');return p.parse_args()
def vote(features,text_by_class,orders,k):
 scores=[]
 for c,(texts,order) in enumerate(zip(text_by_class,orders)):scores.append((features@texts[order[:k]].T).mean(1).numpy())
 matrix=np.stack(scores,1);return matrix.argmax(1),matrix
def global_vote(features,global_text,chosen):
 matrix=np.stack([(features@global_text[idx].T).mean(1).numpy() for idx in chosen],1);return matrix.argmax(1),matrix
def main():
 a=parse();required=[ROOT/'external_aircraft_corrected_clean/feature_cache'/f'image_features_{b}.npz' for b in BACKBONES]
 if any(not p.is_file() for p in required):raise FileNotFoundError('Aircraft cache missing')
 print('PASS: Experiment 5 preflight; matched interfaces and global-pool AS reference configured')
 if a.preflight:return
 names=['matched_interface_curves.csv','attributeselect_mean.csv','attributeselect_probe.csv','predictions.csv','rankings.csv','exact_prompts.csv','figure_interface_comparison.pdf'];paths=[a.output_dir/n for n in names]
 if not a.overwrite and any(p.exists() for p in paths):raise FileExistsError('Output exists')
 a.output_dir.mkdir(parents=True,exist_ok=True);matched=[];asm=[];asp=[];predrows=[];rankrows=[];promptrows=[]
 for b in BACKBONES:
  ctx=load_context(b);test_ids=official_ids('test');testpos=torch.tensor([ctx['positions'][i] for i in test_ids]);testx=ctx['features'][testpos];testy=ctx['labels'][testpos.numpy()]
  attrs=descriptor_features(ctx);anchored=descriptor_features(ctx,anchored=True)
  unique=[]
  for rec in ctx['classes']:
   for d in rec['descriptors']:
    if d not in unique:unique.append(d)
  global_text=encode_text(ctx,unique) # bare strings, as pre-specified
  for seed in SEEDS:
   support_ids=manifest_support(seed);spos=torch.tensor([ctx['positions'][i] for i in support_ids]);sx=ctx['features'][spos];sy=ctx['labels'][spos.numpy()]
   orders=rank_descriptors(ctx,support_ids,'positive_mean',attrs)
   for c,o in enumerate(orders):
    for rank,j in enumerate(o,1):rankrows.append(dict(analysis='matched',backbone=b,seed=seed,class_id=c,rank=rank,descriptor_id=j+1,descriptor_text=ctx['classes'][c]['descriptors'][j]))
   for k in range(1,5):
    prompts=build_recomposed_prompts(ctx['classes'],orders,k);text=encode_text(ctx,prompts);rp,_=predict(testx,text);vp,_=vote(testx,attrs,orders,k);ap,_=vote(testx,anchored,orders,k)
    for interface,pp in [('single_recomposed_prototype',rp),('independent_attribute_voting',vp),('class_anchored_descriptor_ensemble',ap)]:
     matched.append(dict(domain='aircraft',backbone=b,seed=seed,ranker='positive_mean',k=k,interface=interface,test_macro_accuracy=macro_accuracy(testy,pp)))
     predrows += [dict(analysis='matched',backbone=b,seed=seed,k=k,method=interface,image_id=i,true_index=int(t),predicted_index=int(p)) for i,t,p in zip(test_ids,testy,pp)]
    for c,prompt in enumerate(prompts):promptrows.append(dict(backbone=b,seed=seed,k=k,interface='single_recomposed_prototype',class_id=c,prompt=prompt))
   # AS-Mean: class-wise mean support similarity over the deduplicated global pool.
   support_attribute=(sx@global_text.T).numpy()*100.0;mean_scores=np.stack([support_attribute[sy==c].mean(0) for c in range(100)]);mean_order=np.argsort(-mean_scores,axis=1,kind='stable')
   # AS-Probe: multinomial logistic weights over the same support attribute vectors.
   probe=LogisticRegression(max_iter=3000,solver='lbfgs',random_state=seed).fit(support_attribute,sy);coef=np.empty((100,len(unique)));coef[probe.classes_.astype(int)]=probe.coef_;probe_order=np.argsort(-coef,axis=1,kind='stable')
   for method,order,target in [('AS-Mean',mean_order,asm),('AS-Probe',probe_order,asp)]:
    for k in range(1,5):
     pp,_=global_vote(testx,global_text,order[:,:k]);target.append(dict(backbone=b,seed=seed,k=k,method=method,test_macro_accuracy=macro_accuracy(testy,pp),global_pool_size=len(unique),encoding='bare descriptor string',classification='unweighted independent vote'))
     predrows += [dict(analysis='attributeselect_reference',backbone=b,seed=seed,k=k,method=method,image_id=i,true_index=int(t),predicted_index=int(p)) for i,t,p in zip(test_ids,testy,pp)]
     for c in range(100):
      for rank,j in enumerate(order[c,:k],1):rankrows.append(dict(analysis=method,backbone=b,seed=seed,class_id=c,rank=rank,descriptor_id=int(j),descriptor_text=unique[int(j)]))
 # Development-domain counterpart already exists in the frozen aggregation
 # diagnostic; copy its PM test curves rather than recomputing them.
 dev_path=ROOT/'experiments/results/experiment_41_novelty_rebuild_B/experiment_b_budget_curves.csv';dev=pd.read_csv(dev_path)
 dev=dev[(dev.selector=='positive_mean')&(dev.split=='test')&(dev.k.between(1,4))].copy();dev['domain']='development';dev['seed']=0;dev['ranker']='positive_mean';dev['interface']=dev.composition.map({'recompose':'single_recomposed_prototype','attribute_vote':'independent_attribute_voting','base_plus_vote':'base_plus_vote'});dev['test_macro_accuracy']=dev.accuracy
 matched_frame=pd.concat([pd.DataFrame(matched),dev[['domain','backbone','seed','ranker','k','interface','test_macro_accuracy']]],ignore_index=True)
 matched_frame.to_csv(paths[0],index=False);pd.DataFrame(asm).to_csv(paths[1],index=False);pd.DataFrame(asp).to_csv(paths[2],index=False);pd.DataFrame(predrows).to_csv(paths[3],index=False);pd.DataFrame(rankrows).to_csv(paths[4],index=False);pd.DataFrame(promptrows).to_csv(paths[5],index=False)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 frame=pd.concat([pd.DataFrame(matched).rename(columns={'interface':'method'}),pd.DataFrame(asm),pd.DataFrame(asp)],ignore_index=True);fig,axes=plt.subplots(1,2,figsize=(8,3.6),sharey=True)
 for ax,b in zip(axes,BACKBONES):
  for method,g in frame[frame.backbone.eq(b)].groupby('method'):
   curve=g.groupby('k').test_macro_accuracy.mean();ax.plot(curve.index,100*curve.values,marker='o',label=method)
  ax.set_title(b.replace('ViT-',''));ax.set_xlabel('k');ax.set_xticks(range(1,5));ax.grid(axis='y',color='.9')
 axes[0].set_ylabel('Macro accuracy (%)');axes[1].legend(fontsize=6,bbox_to_anchor=(1.02,1),loc='upper left');fig.tight_layout();fig.savefig(paths[6],bbox_inches='tight');plt.close(fig);print(a.output_dir.resolve())
if __name__=='__main__':main()
