"""Experiment 4: class-name and single-template baselines.

Runs new text-only encoding on frozen image feature caches and joins the result
to existing k=0/k=1/k=4 predictions. It never re-encodes an image.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score,f1_score,recall_score
from transformers import CLIPModel,CLIPTokenizer
from aircraft_cached_core import ROOT,BACKBONES,load_context,encode_text,official_ids

def parse():
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=ROOT/'revision_experiments_v13/exp4_prompt_baselines');p.add_argument('--preflight',action='store_true');p.add_argument('--overwrite',action='store_true');return p.parse_args()
def dev_encode(model,tok,prompts):
 x=tok(prompts,return_tensors='pt',padding=True,truncation=False)
 with torch.inference_mode():
  h=model.text_model(**x);last=x['attention_mask'].sum(1)-1;pooled=h.last_hidden_state[torch.arange(len(prompts)),last]
  return F.normalize(model.text_projection(pooled),dim=-1).cpu()
def rows(domain,b,r,prompt,y,pred,names):
 out=[]
 for c,n in enumerate(names):
  m=y==c;out.append(dict(domain=domain,backbone=b,ranker=r,prompt_condition=prompt,class_id=c,class_name=n,n=int(m.sum()),correct=int((pred[m]==c).sum()),recall=float((pred[m]==c).mean())))
 return out
def main():
 a=parse(); devpred=ROOT/'experiments/results/experiment_47_novelty_rebuild_D/experiment_d_per_image_metrics.csv'; airpred=ROOT/'external_aircraft_corrected_clean/predictions.csv'
 required=[devpred,airpred]+[ROOT/'experiments/results'/('visual_audit_crop4_features_b16.npz' if b=='ViT-B-16' else 'visual_audit_crop4_features.npz') for b in BACKBONES]
 if any(not p.is_file() for p in required):raise FileNotFoundError('Frozen prediction/feature input missing')
 print('PASS: Experiment 4 preflight; development and Aircraft caches/predictions present')
 if a.preflight:return
 outs=[a.output_dir/x for x in ['prompt_baselines.csv','classwise_prompt_baselines.csv','drugstore_prompt_audit.csv','predictions.csv']]
 if not a.overwrite and any(p.exists() for p in outs):raise FileExistsError('Output exists')
 a.output_dir.mkdir(parents=True,exist_ok=True); summary=[];classwise=[];predrows=[]
 # Development-domain A/B; C/k1/k4 are joined from frozen predictions for both rankers.
 meta=[json.loads(x) for x in (ROOT/'pest_vlm_dataset/annotations/metadata.jsonl').read_text(encoding='utf-8').splitlines() if x]; bystem={x['stem']:x for x in meta}; names=[x['name_en'] for x in sorted(json.loads((ROOT/'pest_vlm_dataset/annotations/class_knowledge.json').read_text(encoding='utf-8'))['classes'],key=lambda z:z['class_id'])]
 test=json.loads((ROOT/'pest_vlm_dataset/splits/test.json').read_text(encoding='utf-8')); teststems=[Path(x).stem for x in test['all_samples']]; class_ids=sorted({x['class_id'] for x in meta}); y=np.asarray([class_ids.index(bystem[s]['class_id']) for s in teststems])
 frozen=pd.read_csv(devpred)
 for b in BACKBONES:
  fp=ROOT/'experiments/results'/('visual_audit_crop4_features_b16.npz' if b=='ViT-B-16' else 'visual_audit_crop4_features.npz');z=np.load(fp);idx={str(s):i for i,s in enumerate(z['stems'])};image=torch.from_numpy(z['image_features'].astype('float32'))[torch.tensor([idx[s] for s in teststems])]
  model_path=ROOT/('models/clip-vit-base-patch16' if b=='ViT-B-16' else 'visual_knowledge_calibration_server_pack/models/clip-vit-base-patch32');model=CLIPModel.from_pretrained(model_path,local_files_only=True).eval().cpu();tok=CLIPTokenizer.from_pretrained(model_path,local_files_only=True)
  for condition,prompts in [('class_name_only',names),('standard_template',[f'a photo of a {n}.' for n in names])]:
   pred=(image@dev_encode(model,tok,prompts).T).numpy().argmax(1);summary.append(dict(domain='development',backbone=b,ranker='not_applicable',prompt_condition=condition,macro_accuracy=recall_score(y,pred,labels=range(10),average='macro',zero_division=0),top1_accuracy_secondary=accuracy_score(y,pred),macro_f1=f1_score(y,pred,average='macro')));classwise+=rows('development',b,'not_applicable',condition,y,pred,names);predrows += [dict(domain='development',backbone=b,ranker='not_applicable',prompt_condition=condition,image_id=s,true_index=int(t),predicted_index=int(p)) for s,t,p in zip(teststems,y,pred)]
  for ranker in ['positive_mean','contrastive_margin']:
   for condition,k in [('current_base',0),('fixed_k1',1),('full_k4',4)]:
    q=frozen[(frozen.domain=='main')&(frozen.backbone==b)&(frozen.selector==ranker)&(frozen.k==k)].sort_values('stem'); yy=q.true_index.to_numpy();pp=q.predicted_index.to_numpy();summary.append(dict(domain='development',backbone=b,ranker=ranker,prompt_condition=condition,macro_accuracy=recall_score(yy,pp,labels=range(10),average='macro',zero_division=0),top1_accuracy_secondary=accuracy_score(yy,pp),macro_f1=f1_score(yy,pp,labels=range(10),average='macro',zero_division=0)));classwise+=rows('development',b,ranker,condition,yy,pp,names)
 # Aircraft A/B and frozen C/k1/k4, averaged across five resamples for ranker-dependent conditions.
 af=pd.read_csv(airpred,dtype={'image_id':str})
 for b in BACKBONES:
  ctx=load_context(b);ids=official_ids('test');pos=torch.tensor([ctx['positions'][i] for i in ids]);yy=ctx['labels'][pos.numpy()];an=ctx['names']
  for condition,prompts in [('class_name_only',an),('standard_template',[f'a photo of a {n}.' for n in an])]:
   pp=(ctx['features'][pos]@encode_text(ctx,prompts).T).numpy().argmax(1);summary.append(dict(domain='aircraft',backbone=b,ranker='not_applicable',prompt_condition=condition,macro_accuracy=recall_score(yy,pp,labels=range(100),average='macro',zero_division=0),top1_accuracy_secondary=accuracy_score(yy,pp),macro_f1=f1_score(yy,pp,labels=range(100),average='macro',zero_division=0)));classwise+=rows('aircraft',b,'not_applicable',condition,yy,pp,an);predrows += [dict(domain='aircraft',backbone=b,ranker='not_applicable',prompt_condition=condition,image_id=i,true_index=int(t),predicted_index=int(p)) for i,t,p in zip(ids,yy,pp)]
  for ranker in ['positive_mean','contrastive_margin']:
   for condition,k in [('current_base',0),('fixed_k1',1),('full_k4',4)]:
    macro_vals=[];top1_vals=[];f1_vals=[]
    for seed,g in af[(af.backbone==b)&(af.scorer==ranker)&(af.split=='test')&(af.k==k)].groupby('seed'):
     macro_vals.append(recall_score(g.true_index,g.predicted_index,labels=range(100),average='macro',zero_division=0));top1_vals.append(accuracy_score(g.true_index,g.predicted_index));f1_vals.append(f1_score(g.true_index,g.predicted_index,labels=range(100),average='macro',zero_division=0));classwise+=rows('aircraft',b,ranker,condition,g.true_index.to_numpy(),g.predicted_index.to_numpy(),an)
    summary.append(dict(domain='aircraft',backbone=b,ranker=ranker,prompt_condition=condition,macro_accuracy=np.mean(macro_vals),top1_accuracy_secondary=np.mean(top1_vals),macro_f1=np.mean(f1_vals)))
 pd.DataFrame(summary).to_csv(outs[0],index=False);cw=pd.DataFrame(classwise);cw.to_csv(outs[1],index=False);pd.DataFrame(predrows).to_csv(outs[3],index=False);cw[cw.class_name.str.contains('drugstore',case=False,na=False)].to_csv(outs[2],index=False);print(a.output_dir.resolve())
if __name__=='__main__':main()
