"""Compact redundant budget-level Table S26 without changing its evidence."""
from pathlib import Path

import pandas as pd
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "Supplementary_Information_v13.docx"
DATA = ROOT / "machine_readable" / "Figure5_matched_interface_summary.csv"
doc = Document(PATH)

# Table S25's complete 24-condition table is retained; modestly tighten only its
# typesetting so the adjacent descriptive summary is not stranded by a page break.
table_s25 = doc.tables[24]
for row in table_s25.rows:
    for cell in row.cells:
        for paragraph in cell.paragraphs:
            paragraph.paragraph_format.space_before = Pt(0)
            paragraph.paragraph_format.space_after = Pt(0)
            paragraph.paragraph_format.line_spacing = 0.9
            for run in paragraph.runs:
                run.font.size = Pt(7.5)

# Table S26 duplicates the full curves already shown in Figure 5 and supplied
# as CSV. Retain each configuration's best observed operating point; no data
# are discarded because the full k=1..4 table remains machine-readable.
old = doc.tables[25]
if len(old.rows) == 25:
    raw = pd.read_csv(DATA)
    labels = {
        "class_anchored_descriptor_ensemble": "Class-anchored ensemble",
        "independent_attribute_voting": "Independent vote",
        "single_recomposed_prototype": "Single recomposition",
    }
    rows = []
    for (backbone, interface), group in raw.groupby(["backbone", "interface"], sort=True):
        best = group.sort_values(["mean", "k"], ascending=[False, True]).iloc[0]
        rows.append((backbone.replace("ViT-", ""), labels[interface], int(best.k),
                     f"{100 * best['mean']:.2f} ± {100 * best['std']:.2f}"))
    parent = old._tbl.getparent()
    replacement = doc.add_table(rows=1, cols=4)
    replacement.style = "Table Grid"
    parent.remove(replacement._tbl)
    old._tbl.addprevious(replacement._tbl)
    parent.remove(old._tbl)
    headers = ["Backbone", "Consumption configuration", "Best k", "Mean macro accuracy (%) ± seed SD"]
    for cell, value in zip(replacement.rows[0].cells, headers):
        cell.text = value
    for values in rows:
        cells = replacement.add_row().cells
        for cell, value in zip(cells, values):
            cell.text = str(value)
    for row in replacement.rows:
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_after = Pt(0)
                for run in paragraph.runs:
                    run.font.size = Pt(7.5)
    caption = next(p for p in doc.paragraphs if p.text.startswith("Table S26."))
    caption.text = (
        "Table S26. Best observed budget within each matched Aircraft consumption configuration; "
        "macro accuracy is mean ± seed-level SD across five frozen resamples. Full k=1–4 curves "
        "are shown in Figure 5 and supplied in the machine-readable CSV."
    )
elif len(old.rows) == 7:
    pass  # Already summarized; make the operation safely repeatable.
else:
    raise RuntimeError(f"Unexpected Table S26 row count: {len(old.rows)}")

doc.save(PATH)
print("PASS: Table S26 now summarizes six best operating points; complete curves remain in CSV.")
