from pathlib import Path
from zipfile import ZipFile
from pypdf import PdfReader
import re, sys

ROOT=Path(__file__).resolve().parents[1]
errors=[]
if list(ROOT.rglob("README*")) != [ROOT/"README.md"]:
    errors.append("expected exactly one active README.md")
if any(p.name==".git" for p in ROOT.rglob("*")):
    errors.append("Git metadata found")
drive=re.compile(r"(?i)\b[A-Z]:\\")
unix_home=re.compile(r"(?i)/(?:Users|home)/")
email=re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
text_ext={".py",".ps1",".md",".txt",".json",".csv",".tsv",".yml",".yaml",".log",".toml",".ini",".xml",".rst"}
for p in ROOT.rglob("*"):
    if not p.is_file(): continue
    if p.suffix.lower() in text_ext:
        text=p.read_text(encoding="utf-8-sig",errors="replace")
        if drive.search(text) or unix_home.search(text): errors.append(f"local path in {p.relative_to(ROOT)}")
        if email.search(text): errors.append(f"email address in {p.relative_to(ROOT)}")
    elif p.suffix.lower()==".pdf":
        try: meta=str(PdfReader(p).metadata or "")
        except Exception as exc:
            errors.append(f"unreadable PDF metadata in {p.relative_to(ROOT)}: {exc}")
            continue
        if drive.search(meta) or unix_home.search(meta) or email.search(meta):
            errors.append(f"identifying PDF metadata in {p.relative_to(ROOT)}")
    elif p.suffix.lower()==".docx":
        try:
            with ZipFile(p) as z: meta=z.read("docProps/core.xml").decode("utf-8","replace")
        except Exception as exc:
            errors.append(f"unreadable DOCX metadata in {p.relative_to(ROOT)}: {exc}")
            continue
        if drive.search(meta) or unix_home.search(meta) or email.search(meta):
            errors.append(f"identifying DOCX metadata in {p.relative_to(ROOT)}")
if errors:
    print("FAIL")
    for e in errors: print(e)
    sys.exit(1)
print(f"PASS: public-release content scan across {sum(1 for p in ROOT.rglob('*') if p.is_file())} files; one README; no Git metadata, local paths, email addresses, or identifying Office/PDF metadata")
