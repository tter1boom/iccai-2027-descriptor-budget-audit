"""Experiment 2: development-domain class concentration and LOCO sensitivity.

Uses frozen per-image predictions and the frozen visual-component map. No CLIP
inference is performed. Intervals are descriptive 20,000-replicate
class-stratified visual-component bootstrap intervals; no new p-value family is
created.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

from canonical_macro_cluster_bootstrap import macro_cluster_bootstrap_draws

def find_root():
    for candidate in Path(__file__).resolve().parents:
        if (candidate/'machine_readable/validation_and_predictions/experiment_b_predictions.csv').is_file():
            return candidate
        if (candidate/'experiments/results/experiment_47_novelty_rebuild_D/experiment_d_per_image_metrics.csv').is_file():
            return candidate
    return Path(__file__).resolve().parents[1]

ROOT = find_root()

def args():
    p=argparse.ArgumentParser()
    packaged = (ROOT/'machine_readable/validation_and_predictions/experiment_b_predictions.csv').is_file()
    p.add_argument('--predictions',type=Path,default=(ROOT/'machine_readable/validation_and_predictions/experiment_b_predictions.csv' if packaged else ROOT/'experiments/results/experiment_47_novelty_rebuild_D/experiment_d_per_image_metrics.csv'))
    p.add_argument('--base-predictions',type=Path,default=(ROOT/'machine_readable/validation_and_predictions/main_base_k0_per_image_predictions.csv' if packaged else None))
    p.add_argument('--selected',type=Path,default=(ROOT/'machine_readable/validation_and_predictions/experiment_b_validation_selected.csv' if packaged else ROOT/'experiments/results/experiment_41_novelty_rebuild_B/experiment_b_validation_selected.csv'))
    p.add_argument('--components',type=Path,default=(ROOT/'machine_readable/group_clustered_inference/test_visual_component_map.csv' if packaged else ROOT/'experiments/results/experiment_57_group_clustered_statistics/test_visual_component_map.csv'))
    p.add_argument('--metadata',type=Path,default=(ROOT/'machine_readable/inputs/main_benchmark/annotations/metadata.jsonl' if packaged else ROOT/'pest_vlm_dataset/annotations/metadata.jsonl'))
    p.add_argument('--source-only-predictions',type=Path,default=None)
    p.add_argument('--output-dir',type=Path,default=(ROOT/'machine_readable/recomputed_from_raw/experiment_2_class_concentration' if packaged else ROOT/'revision_experiments_v13/exp2_class_concentration'))
    p.add_argument('--bootstrap-replicates',type=int,default=20000)
    p.add_argument('--random-seed',type=int,default=20261004)
    p.add_argument('--preflight',action='store_true'); p.add_argument('--overwrite',action='store_true')
    return p.parse_args()

def component_boot(delta, cls, comp, excluded, reps, rng):
    keep=cls!=excluded
    draws=macro_cluster_bootstrap_draws(delta[keep],cls[keep],comp[keep],reps,rng)
    return np.quantile(draws,[.025,.975])

def main():
    a=args(); required=[a.predictions,a.selected,a.components,a.metadata]
    if a.base_predictions is not None: required.append(a.base_predictions)
    if any(not p.is_file() for p in required): raise FileNotFoundError([str(p) for p in required if not p.is_file()])
    pred=pd.read_csv(a.predictions); sel=pd.read_csv(a.selected); cmap=pd.read_csv(a.components)
    if a.base_predictions is not None:
        base_input=pd.read_csv(a.base_predictions)
        if not {'backbone','selector','k','stem','true_index','predicted_index'}.issubset(base_input.columns):
            raise RuntimeError('Separate k=0 prediction file has an incompatible schema')
        base_input=base_input[base_input.k.eq(0)].copy()
        if 'composition' not in base_input.columns: base_input['composition']='recompose'
        if 'split' not in base_input.columns: base_input['split']='test'
        pred=pd.concat([pred,base_input],ignore_index=True,sort=False)
    names={r['class_id']:r['label_en'] for r in (json.loads(x) for x in a.metadata.read_text(encoding='utf-8').splitlines() if x)}
    if 'composition' in sel.columns: sel=sel[sel.composition.eq('recompose')]
    if 'component_id' not in cmap.columns and 'component' in cmap.columns: cmap=cmap.rename(columns={'component':'component_id'})
    if len(sel)!=4 or len(cmap)!=300 or 'component_id' not in cmap.columns: raise RuntimeError('Frozen development inputs do not match 4 conditions / 300 mapped test images')
    if 'domain' in pred.columns: pred=pred[pred.domain.eq('main')]
    if 'split' in pred.columns: pred=pred[pred.split.eq('test')]
    if 'composition' in pred.columns: pred=pred[pred.composition.eq('recompose')]
    print('PASS: Experiment 2 preflight (4 conditions, 10 classes, 300 test images, component map present)')
    if a.preflight:return
    targets=[a.output_dir/x for x in ['classwise_changes.csv','leave_one_class_out.csv','drugstore_confusion.csv','figure_class_concentration.pdf']]
    if not a.overwrite and any(x.exists() for x in targets): raise FileExistsError('Output exists; use --overwrite or another directory')
    a.output_dir.mkdir(parents=True,exist_ok=True); class_rows=[]; loco=[]; conf=[]
    for s in sel.itertuples(index=False):
        selected_k=int(getattr(s,'val_selected_k',getattr(s,'selected_k',-1)))
        base=pred[(pred.backbone==s.backbone)&(pred.selector==s.selector)&(pred.k==0)].sort_values('stem')
        chosen=pred[(pred.backbone==s.backbone)&(pred.selector==s.selector)&(pred.k==selected_k)].sort_values('stem')
        if len(base)!=300 or base.stem.tolist()!=chosen.stem.tolist():raise RuntimeError('Prediction alignment failed')
        delta=(chosen.predicted_index.to_numpy()==base.true_index.to_numpy()).astype(int)-(base.predicted_index.to_numpy()==base.true_index.to_numpy()).astype(int)
        positive=[]
        for c in sorted(names):
            mask=base.true_index.to_numpy()==c; change=int(delta[mask].sum()); positive.append(max(change,0))
            class_rows.append(dict(backbone=s.backbone,ranker=s.selector,selected_k=selected_k,class_id=c,class_name=names[c],n=int(mask.sum()),base_correct=int((base.predicted_index.to_numpy()[mask]==c).sum()),selected_correct=int((chosen.predicted_index.to_numpy()[mask]==c).sum()),delta_correct=change,delta_recall_pp=100*change/mask.sum()))
        total=sum(positive)
        for row in class_rows[-len(names):]: row['positive_gain_concentration']=max(positive)/total if total else np.nan
        for excluded in sorted(names):
            keep=base.true_index.to_numpy()!=excluded
            obs=np.mean([delta[(base.true_index.to_numpy()==c)].mean() for c in sorted(names) if c!=excluded])
            aligned=base[['stem','true_index']].merge(cmap[['stem','true_index','component_id']],on=['stem','true_index'],validate='one_to_one').sort_values('stem')
            ci=component_boot(delta,base.true_index.to_numpy(),aligned.component_id.to_numpy(),excluded,a.bootstrap_replicates,np.random.default_rng(a.random_seed+int(excluded)+sum(map(ord,s.backbone+s.selector))))
            loco.append(dict(backbone=s.backbone,ranker=s.selector,excluded_class_id=excluded,excluded_class_name=names[excluded],delta_selected_minus_base=obs,delta_pp=100*obs,ci95_low=ci[0],ci95_high=ci[1],ci95_low_pp=100*ci[0],ci95_high_pp=100*ci[1],interval_type='descriptive visual-component clustered bootstrap'))
        drug=[c for c,n in names.items() if 'drugstore' in n.lower()]
        if drug:
            c=drug[0]; b=base[base.true_index.eq(c)]; q=chosen[chosen.true_index.eq(c)]
            counts=b.predicted_index.value_counts()
            for rank,(pid,count) in enumerate(counts.head(3).items(),1):
                row=dict(backbone=s.backbone,ranker=s.selector,selected_k=selected_k,true_class=names[c],rank=rank,predicted_class_id=int(pid),predicted_class=names.get(int(pid),str(pid)),count=int(count),base_correct=int((b.predicted_index==c).sum()),selected_correct=int((q.predicted_index==c).sum()))
                for col,outcol in [('correct_class_alignment','base_correct_class_mean'),('strongest_competitor_alignment','base_strongest_competitor_mean'),('decision_margin','base_decision_margin_mean')]:
                    if col in base.columns: row[outcol]=float(b[col].mean())
                conf.append(row)
    cf=pd.DataFrame(class_rows); cf.to_csv(targets[0],index=False); pd.DataFrame(loco).to_csv(targets[1],index=False); pd.DataFrame(conf).to_csv(targets[2],index=False)
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    lf=pd.DataFrame(loco); fig,ax=plt.subplots(figsize=(8,4.5))
    for label,g in lf.groupby(['backbone','ranker']): ax.plot(g.excluded_class_name,g.delta_pp,marker='o',label=' | '.join(label))
    ax.axhline(0,color='.4',ls='--'); ax.set_ylabel('Selected - base (pp)'); ax.tick_params(axis='x',rotation=45,labelsize=7); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(targets[3]); plt.close(fig)
    (a.output_dir/'analysis_manifest.json').write_text(json.dumps({'analysis':'Experiment 2 class-concentration and leave-one-class-out intervals','bootstrap_replicates':a.bootstrap_replicates,'seed':a.random_seed,'bootstrap_estimand':'within each class, resample visual components with replacement; pool sampled image deltas and image counts; divide to obtain the class image-level mean; macro-average equally across non-excluded classes','model_inference':'none; frozen per-image predictions only'},indent=2)+'\n',encoding='utf-8')
    print(a.output_dir.resolve())
if __name__=='__main__':main()
