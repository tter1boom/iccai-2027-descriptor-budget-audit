"""Verify required deliverables, one active README, path hygiene, and SHA-256 inventory."""
from pathlib import Path
import csv,hashlib,re,json
ROOT=Path(__file__).resolve().parents[1]; SUB=ROOT.parent
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
req=[ROOT/"README.md",ROOT/"environment.yml",ROOT/"ACTIVE_CODE_MANIFEST.csv",ROOT/"01_protocol/experiment_config.json",ROOT/"02_resources/aircraft_descriptors_frozen.json",ROOT/"05_results/aircraft_budget_curves.csv"]
missing=[p.as_posix() for p in req if not p.is_file()]; assert not missing,missing
readmes=[p for p in ROOT.rglob("README*") if p.is_file()]; assert readmes==[ROOT/"README.md"],readmes
colon,slash=chr(58),chr(92); drive=colon+slash
viol=[]
for d in ["01_protocol","02_resources","03_code","04_data","05_results","06_reproduction"]:
 for p in (ROOT/d).rglob("*"):
  if p.is_file() and p.suffix.lower() in {".json",".csv",".tsv",".txt",".md",".py",".yml",".yaml"}:
   t=p.read_text(encoding="utf-8-sig",errors="replace")
   if re.search(r"[A-Za-z]"+re.escape(drive),t) or ("/"+"Users/") in t or ("/"+"home/") in t: viol.append(p.relative_to(ROOT).as_posix())
assert not viol,viol
with (ROOT/"08_integrity/SHA256SUMS.tsv").open(newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f,delimiter="\t"))
bad=[r["path"] for r in rows if not (ROOT/r["path"]).is_file() or sha(ROOT/r["path"])!=r["sha256"]]
assert not bad,bad
manifest=json.loads((ROOT/"08_integrity/release_manifest.json").read_text(encoding="utf-8"))
submission=manifest.get("submission_files",{})
present=[name for name in submission if (SUB/name).is_file()]
assert not present or len(present)==len(submission), ("partial sibling submission files",present)
doc_bad=[]
for name,info in submission.items():
 p=SUB/name
 if p.is_file() and (p.stat().st_size!=info["bytes"] or sha(p)!=info["sha256"]): doc_bad.append(name)
assert not doc_bad,doc_bad
if present:
 print(f"PASS: release layout, one README, active-path hygiene, {len(rows)} package SHA-256 entries, and {len(present)} sibling submission-file hashes")
else:
 print(f"PASS: standalone package layout, one README, active-path hygiene, and {len(rows)} package SHA-256 entries; sibling submission files not present")
