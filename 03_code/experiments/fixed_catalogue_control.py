"""Post-review exploratory fixed-catalogue Aircraft ranking control.

Uses the five frozen validation manifests, cached official train/validation/test
image features, the frozen local CLIP text encoders, the canonical recomposition
interface, and macro accuracy. Descriptor order is catalogue index 1-4 for every
class and does not depend on support images. No inferential test is performed.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from aircraft_core import (BACKBONES, FEATURE_CACHE, PACKAGE_ROOT, RESOURCE,
    SEEDS, encode_text, load_context, macro_accuracy, recomposed_prompt, sha256)

DEFAULT_OUTPUT = PACKAGE_ROOT / "05_results/replayed_fixed_order_control"


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, required=True,
        help="Project root containing checkpoints listed in 01_protocol/model_lock.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight", action="store_true")
    args=parser.parse_args()
    args.models_root=args.models_root.resolve()
    out=args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {out}")
    resource=json.loads(RESOURCE.read_text(encoding="utf-8"))
    classes=resource["classes"]
    class_to_id={c["name_en"]:i for i,c in enumerate(classes)}
    if len(classes)!=100 or any(len(c["descriptors"])!=4 for c in classes):
        raise RuntimeError("Expected 100 classes with four catalogue-ordered descriptors each")
    manifests={}
    split_dir=PACKAGE_ROOT/"01_protocol/split_manifests"
    for seed in SEEDS:
        path=split_dir/f"split_seed_{seed}.json"
        m=json.loads(path.read_text(encoding="utf-8"))
        if int(m["seed"])!=seed: raise RuntimeError(f"Seed mismatch in {path.name}")
        manifests[seed]=m
        if set(m["validation"])!=set(class_to_id):
            raise RuntimeError(f"Validation classes do not match the frozen catalogue for seed {seed}")
        if any(len(m["validation"][c])!=5 for c in class_to_id):
            raise RuntimeError(f"Expected 5 validation images per class for seed {seed}")
    if args.preflight:
        for backbone in BACKBONES:
            context=load_context(backbone,args.models_root)
            del context
        print("PASS: resource, five validation manifests, image-feature caches and checkpoint fingerprints")
        return

    curve=[]; selected=[]; test_reference=[]
    for backbone in BACKBONES:
        context=load_context(backbone,args.models_root)
        cache_path=FEATURE_CACHE/f"image_features_{backbone}.npz"
        cache=np.load(cache_path)
        cache_ids=cache["image_ids"].astype(str).tolist()
        all_features=F.normalize(torch.from_numpy(cache["features"].astype(np.float32)),dim=-1)
        id_pos={v:i for i,v in enumerate(cache_ids)}
        if len(id_pos)!=10000: raise RuntimeError("Expected 10,000 unique cached image IDs")
        test_features=context["features"]
        test_truth=context["labels"]
        test_ids=context["image_ids"]
        if test_ids != pd.read_csv(PACKAGE_ROOT/"01_protocol/aircraft_test_image_labels.csv",dtype={"image_id":str}).image_id.astype(str).tolist():
            raise RuntimeError("Test ordering differs from the frozen test-label map")
        fixed_test={}
        fixed_prompts={}
        fixed_text={}
        for k in range(5):
            prompts=[recomposed_prompt(record,record["descriptors"][:k]) for record in classes]
            fixed_prompts[k]=prompts
            text=encode_text(context,prompts)
            fixed_text[k]=text
            pred=(test_features@text.T).argmax(dim=1).numpy()
            acc=macro_accuracy(test_truth,pred,len(classes))
            fixed_test[k]=acc
            test_reference.append({"backbone":backbone,"k":k,"test_macro_accuracy":acc,
                "ranking":"fixed_catalogue_order","test_n":len(test_truth)})
        for seed in SEEDS:
            manifest=manifests[seed]
            val_ids=[]; val_truth=[]
            for cid,record in enumerate(classes):
                names=manifest["validation"][record["name_en"]]
                val_ids.extend(names); val_truth.extend([cid]*len(names))
            if len(val_ids)!=500: raise RuntimeError(f"Expected 500 validation IDs for {seed}")
            if any(x not in id_pos for x in val_ids): raise RuntimeError(f"Validation ID missing from cache for seed {seed}")
            val_features=all_features[torch.tensor([id_pos[x] for x in val_ids],dtype=torch.long)]
            val_truth_np=np.asarray(val_truth,dtype=np.int64)
            val_acc={}
            for k in range(5):
                text=fixed_text[k]
                pred=(val_features@text.T).argmax(dim=1).numpy()
                acc=macro_accuracy(val_truth_np,pred,len(classes))
                val_acc[k]=acc
                curve.append({"backbone":backbone,"seed":seed,"k":k,
                    "ranking":"fixed_catalogue_order","validation_macro_accuracy":acc,
                    "test_macro_accuracy":fixed_test[k],"validation_n":500,"test_n":3333,
                    "ranker_depends_on_support":False,
                    "analysis_status":"post_review_exploratory_ranking_control"})
            best_k=max(range(5),key=lambda k:(val_acc[k],-k))
            selected.append({"backbone":backbone,"seed":seed,"ranking":"fixed_catalogue_order",
                "selected_k":best_k,"validation_macro_accuracy":val_acc[best_k],
                "test_macro_accuracy":fixed_test[best_k],
                "analysis_status":"post_review_exploratory_ranking_control"})
        del context

    # Descriptive selected-operating-point comparison with the two existing
    # support-conditioned heuristics; no added inferential family.
    budget=pd.read_csv(PACKAGE_ROOT/"04_results/aircraft_main/budget_curves.csv")
    chosen=pd.read_csv(PACKAGE_ROOT/"04_results/aircraft_main/selected_budgets.csv")
    comparisons=[]
    fixed_lookup={(x["backbone"],x["seed"]):x for x in selected}
    for x in selected: comparisons.append(dict(x))
    for row in chosen.to_dict("records"):
        matches=budget[(budget.backbone==row["backbone"])&(budget.seed==row["seed"])&
            (budget.scorer==row["scorer"])&(budget.split=="test")&(budget.k==row["selected_k"])]
        if len(matches)!=1: raise RuntimeError("Cannot resolve selected test accuracy for an existing ranking")
        fixed=fixed_lookup[(row["backbone"],row["seed"])]
        comparisons.append({"backbone":row["backbone"],"seed":int(row["seed"]),
            "ranking":row["scorer"],"selected_k":int(row["selected_k"]),
            "validation_macro_accuracy":float(row["validation_macro_accuracy"]),
            "test_macro_accuracy":float(matches.iloc[0].macro_accuracy),
            "difference_pp_vs_fixed_catalogue":100*(float(matches.iloc[0].macro_accuracy)-fixed["test_macro_accuracy"]),
            "analysis_status":"existing_support_conditioned_curve_descriptive_reference"})
    fixed_rows=[x for x in comparisons if x["ranking"]=="fixed_catalogue_order"]
    for x in fixed_rows: x["difference_pp_vs_fixed_catalogue"]=0.0
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(curve).to_csv(out/"fixed_catalogue_budget_curves.csv",index=False)
    pd.DataFrame(selected).to_csv(out/"fixed_catalogue_validation_selected.csv",index=False)
    pd.DataFrame(test_reference).to_csv(out/"fixed_catalogue_test_curves.csv",index=False)
    pd.DataFrame(comparisons).to_csv(out/"selected_operating_point_descriptive_comparison.csv",index=False)
    manifest={"analysis_status":"post_review_exploratory_ranking_control",
      "question":"How does support-conditioned descriptor ranking compare descriptively with fixed catalogue order under the same recomposition, budgets, validation selection rule and official test?",
      "ranking":"Within each class, use descriptor catalogue index 1,2,3,4 for all seeds; no support-conditioned ordering or randomization.",
      "interface":"single-prototype recomposition via the archived canonical prompt template",
      "budget_values":[0,1,2,3,4],"validation_selection":"macro accuracy on the frozen 5-per-class validation sample; smallest k breaks ties",
      "test":"complete fixed official FGVC-Aircraft test split (n=3333); macro accuracy",
      "support_validation_seeds":list(SEEDS),"support_images_used_for_ranking":False,
      "inference":"cached image features and frozen local CLIP text encoders; no image-model inference; no hypothesis tests or multiplicity claims",
      "resource_sha256":sha256(RESOURCE),"feature_cache_sha256":{b:sha256(FEATURE_CACHE/f"image_features_{b}.npz") for b in BACKBONES},
      "inputs":["02_resource_audit/aircraft_descriptors_clean.json","01_protocol/split_manifests/","05_execution_records/feature_cache/","01_protocol/model_lock.json","04_results/aircraft_main/budget_curves.csv","04_results/aircraft_main/selected_budgets.csv"],
      "outputs":["fixed_catalogue_budget_curves.csv","fixed_catalogue_validation_selected.csv","fixed_catalogue_test_curves.csv","selected_operating_point_descriptive_comparison.csv"]}
    manifest["script_sha256"]=sha256(Path(__file__).resolve())
    (out/"run_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(f"Wrote exploratory fixed-order control: {out}")

if __name__=="__main__": main()
