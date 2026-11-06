"""Integrity, protocol, schema, and numerical-claim audit for the v14 package."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import struct
import sys
import zipfile
from pathlib import Path, PurePosixPath

import pandas as pd
import numpy as np
from docx import Document

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "SHA256SUMS.tsv"


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_hash_inventory() -> None:
    if not MANIFEST.is_file():
        fail("SHA256SUMS.tsv is missing")
    declared: dict[str, str] = {}
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream, delimiter="\t")
        if rows.fieldnames != ["sha256", "path"]:
            fail("SHA256SUMS.tsv must have the header: sha256<TAB>path")
        for row in rows:
            rel = row["path"]
            pure = PurePosixPath(rel)
            if (not rel or pure.is_absolute() or ".." in pure.parts or "." in pure.parts or
                    "\\" in rel or rel.startswith("/") or rel == "SHA256SUMS.tsv"):
                fail(f"unsafe or invalid manifest path: {rel!r}")
            if rel in declared:
                fail(f"duplicate manifest path: {rel}")
            expected = row["sha256"].strip().lower()
            if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
                fail(f"invalid SHA-256 value for {rel}")
            target = (ROOT / Path(*pure.parts)).resolve()
            if ROOT.resolve() not in target.parents:
                fail(f"manifest path escapes package root: {rel}")
            if not target.is_file():
                fail(f"manifest file is missing: {rel}")
            actual = sha256(target)
            if actual != expected:
                fail(f"SHA-256 mismatch: {rel}")
            declared[rel] = expected
    actual_paths = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and path.resolve() != MANIFEST.resolve()
    }
    if actual_paths != set(declared):
        missing = sorted(set(declared) - actual_paths)
        unlisted = sorted(actual_paths - set(declared))
        fail(f"hash inventory differs from package files; missing={missing[:8]}, unlisted={unlisted[:8]}")
    print(f"PASS: verified {len(declared)} SHA-256 entries; every package file is listed exactly once.")


def read_csv(rel: str, required: tuple[str, ...], rows: int | None = None) -> pd.DataFrame:
    path = ROOT / rel
    if not path.is_file():
        fail(f"required CSV is missing: {rel}")
    frame = pd.read_csv(path)
    absent = [name for name in required if name not in frame.columns]
    if absent:
        fail(f"{rel} missing columns: {absent}")
    if rows is not None and len(frame) != rows:
        fail(f"{rel} expected {rows} rows, got {len(frame)}")
    return frame


def verify_protocol_and_schemas() -> None:
    required = (
        "Manuscript.pdf", "Supplementary_Information.pdf",
        "Manuscript.docx", "Supplementary_Information.docx",
        "machine_readable/paper_claims.json", "protocol/REVISION_EXPERIMENT_PROTOCOL_v13.json",
        "protocol/V13_REVIEWER_ANALYSIS_PROVENANCE.json",
        "scripts/reproduce_paper_outputs.py", "scripts/make_paper_figures.py",
        "scripts/recompute_revision_analyses_from_raw.py",
        "scripts/make_supplementary_figure_s4.py",
        "machine_readable/v13_analysis/experiment_3_validation_size/server_runs/exp3_validation_size/full_validation_support_selections.csv",
    )
    absent = [rel for rel in required if not (ROOT / rel).is_file()]
    if absent:
        fail("required release artifacts missing: " + ", ".join(absent))
    if not (ROOT / "figures" / "Interface_scoring_equations.png").is_file():
        fail("interface-scoring equation image is missing")
    protocol = json.loads((ROOT / "protocol" / "REVISION_EXPERIMENT_PROTOCOL_v13.json").read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != "REVISION_EXPERIMENT_PROTOCOL_v13":
        fail("wrong revision-experiment protocol version")
    if protocol.get("execution_order") != [1, 2, 4, 3, 5]:
        fail("frozen five-analysis execution order mismatch")
    if protocol.get("global_rules", {}).get("no_third_dataset") is not True or protocol.get("global_rules", {}).get("no_model_expansion") is not True:
        fail("protocol boundary fields are missing")
    exp3 = protocol.get("experiment_3", {})
    if len(exp3.get("support_seeds", [])) != 20 or len(exp3.get("validation_permutation_seeds", [])) != 50:
        fail("Experiment 3 seed dimensions do not match the frozen plan")
    if exp3.get("hierarchical_bootstrap_replicates") != 10000:
        fail("Experiment 3 hierarchical-bootstrap replicate count mismatch")
    if protocol.get("experiment_2", {}).get("loco_values") != 40:
        fail("Experiment 2 leave-one-class-out plan mismatch")
    if protocol.get("experiment_5", {}).get("k") != [1, 2, 3, 4]:
        fail("Experiment 5 matched-interface budget plan mismatch")
    provenance = json.loads((ROOT / "protocol" / "V13_REVIEWER_ANALYSIS_PROVENANCE.json").read_text(encoding="utf-8"))
    if provenance.get("status") != "documented_post_review_analysis_plan_not_preregistration":
        fail("analysis chronology/provenance status is missing or overstated")
    deviations = provenance.get("analysis_rules", "")
    if ("Experiment 3 text encoding was parallelized on CUDA" not in deviations or
            "Experiment 2 clustered-bootstrap implementation was subsequently corrected" not in deviations):
        fail("implementation deviations from the documented plan are not disclosed")
    archived_plan = ROOT / "provenance/original_protocols/REVISION_EXPERIMENT_PROTOCOL_v13_early_20261004_211619.json"
    if not archived_plan.is_file() or sha256(archived_plan) != provenance["chronology"][0]["sha256"]:
        fail("timestamped early protocol artifact does not match the provenance record")
    if (ROOT / "machine_readable/source_only_resource_v13").exists():
        fail("duplicate source-only resource directory remains in the full archive")
    if (ROOT / "machine_readable/source_only_resource/SOURCE_ONLY_AND_GROUP_CLUSTERED_RESULTS.md").exists():
        fail("stale source-only conclusion report remains in the current results path")

    read_csv("machine_readable/v13_analysis/experiment_1_aircraft_adjacent/adjacent_results.csv",
             ("backbone", "ranker", "transition", "delta_pp", "ci95_low_pp", "ci95_high_pp", "holm_p_16"), 16)
    read_csv("machine_readable/v13_analysis/experiment_2_class_concentration/classwise_changes.csv",
             ("backbone", "ranker", "class_name", "delta_correct"), 40)
    read_csv("machine_readable/v13_analysis/experiment_2_class_concentration/leave_one_class_out.csv",
             ("backbone", "ranker", "excluded_class_name", "delta_pp", "ci95_low_pp", "ci95_high_pp"), 40)
    read_csv("machine_readable/v13_analysis/experiment_3_validation_size/server_runs/exp3_validation_size/regret_by_validation_size.csv",
             ("backbone", "ranker", "n_validation_per_class", "mean_regret", "oracle_regret_ci95_low", "oracle_regret_ci95_high"), 24)
    distribution = read_csv("machine_readable/v13_analysis/experiment_3_validation_size/server_runs/exp3_validation_size/selected_k_distribution.csv",
                             ("backbone", "ranker", "n_validation_per_class", "selected_k", "count"))
    totals = distribution.groupby(["backbone", "ranker", "n_validation_per_class"])["count"].sum()
    if len(totals) != 24 or not totals.eq(1000).all():
        fail("selected-k distribution must contain 24 groups of 1,000 nested records")
    supports = read_csv("machine_readable/v13_analysis/experiment_3_validation_size/server_runs/exp3_validation_size/full_validation_support_selections.csv",
                        ("support_seed", "selected_k", "test_oracle_k", "permutation_records_with_all_validation", "distinct_selected_k_values"), 20)
    if supports.support_seed.nunique() != 20 or not supports.permutation_records_with_all_validation.eq(50).all() or not supports.distinct_selected_k_values.eq(1).all():
        fail("complete-validation support-level summary is inconsistent")
    read_csv("machine_readable/v13_analysis/experiment_3_validation_size/server_runs/merged/test_curves.csv",
             ("backbone", "ranker", "support_seed", "k", "test_macro_accuracy"), 400)
    read_csv("machine_readable/v13_analysis/experiment_4_prompt_baselines/prompt_baselines.csv",
             ("domain", "backbone", "ranker", "prompt_condition", "macro_accuracy",
              "top1_accuracy_secondary", "macro_f1"), 32)
    read_csv("machine_readable/v13_analysis/experiment_4_prompt_baselines/classwise_prompt_baselines.csv",
             ("domain", "backbone", "ranker", "prompt_condition", "class_id", "recall"), 6560)
    read_csv("machine_readable/v13_analysis/experiment_5_matched_prior/matched_interface_curves.csv",
             ("domain", "backbone", "ranker", "k", "interface", "test_macro_accuracy"), 144)
    corrected = read_csv("machine_readable/group_clustered_inference/group_clustered_paired_statistics.csv",
                         ("backbone", "selector", "comparison", "group_clustered_bootstrap_ci95_low", "group_clustered_bootstrap_ci95_high"), 8)
    if not corrected.group_clustered_bootstrap_ci95_low.notna().all() or not corrected.group_clustered_bootstrap_ci95_high.notna().all():
        fail("corrected historical clustered intervals contain missing endpoints")
    source_intervals = read_csv("machine_readable/source_only_exploratory_intervals/source_only_exploratory_paired_intervals.csv",
                                ("backbone", "selector", "comparison", "component_clustered_bootstrap_ci95_low", "component_clustered_bootstrap_ci95_high"), 8)
    if not source_intervals.analysis_status.eq("exploratory_post_hoc_no_confirmatory_p_value").all():
        fail("source-only intervals are not marked as exploratory post hoc")
    for number in range(1, 5):
        if not list(ROOT.glob(f"figures_v14/FigureS{number}_*.png")):
            fail(f"Supplementary Figure S{number} source image is missing")
    code_plan = (ROOT / "public_code_plan" / "PUBLIC_CODE_MANIFEST.txt").read_text(encoding="utf-8")
    if "run_exp5_base_matched_addendum.py" not in code_plan or "POST-HOC DIAGNOSTIC ADDENDUM" not in code_plan:
        fail("public code inventory omits the post hoc base-matched addendum")
    print("PASS: frozen protocol, expected analysis counts, CSV schemas, and nested-resampling dimensions.")


def verify_no_stale_claims_in_active_tree() -> None:
    forbidden = (
        "All eight pre-specified comparisons " + "retain",
        "before numerical " + "inspection",
        "no additional experiment was run " + "during manuscript assembly",
    )
    text_suffixes = {".md", ".txt", ".json", ".csv", ".py", ".ps1", ".yml", ".yaml"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or "provenance" in path.relative_to(ROOT).parts or path.suffix.lower() not in text_suffixes:
            continue
        body = path.read_text(encoding="utf-8", errors="ignore")
        for phrase in forbidden:
            if phrase.casefold() in body.casefold():
                fail(f"stale numerical/protocol claim found in active file {path.relative_to(ROOT)}: {phrase}")
    print("PASS: no stale protocol-frozen or 8/8 source-only claim appears outside provenance.")


def verify_document_boundaries() -> None:
    main = Document(ROOT / "Manuscript.docx")
    supplement = Document(ROOT / "Supplementary_Information.docx")
    main_text = "\n".join(p.text for p in main.paragraphs)
    supplement_text = "\n".join(p.text for p in supplement.paragraphs)
    if "v13" in (main_text + "\n" + supplement_text).casefold():
        fail("a package-version token remains in the manuscript or Supplement")
    for docx_path in (ROOT / "Manuscript.docx", ROOT / "Supplementary_Information.docx"):
        with zipfile.ZipFile(docx_path) as package:
            xml_text = "\n".join(package.read(n).decode("utf-8", "ignore")
                                    for n in package.namelist() if n.endswith(".xml"))
        if "v13" in xml_text.casefold():
            fail(f"version token remains in Word XML metadata: {docx_path.name}")
    banned = ("were protocol-frozen " + "before numerical " + "inspection",
              "All eight pre-specified comparisons " + "retain",
              "not a pre-registered external " + "replication")
    combined = main_text + "\n" + supplement_text
    for phrase in banned:
        if phrase.casefold() in combined.casefold():
            fail(f"stale reviewer-facing claim remains in DOCX text: {phrase}")
    if "TCA also evaluates one, two and three class attributes" not in main_text:
        fail("TCA's prior attribute-count study is not acknowledged in Related Work")
    design_text = "\n".join(cell.text for row in main.tables[1].rows for cell in row.cells)
    if ("prior test-image exposure during resource construction cannot be ruled out" not in design_text or
            "without benchmark images" not in design_text):
        fail("Table 2 does not distinguish historical and source-only resource boundaries")
    if ("jointly vary prompt context and score aggregation" not in main_text or
            "does not isolate a causal effect of g" not in main_text):
        fail("matched Aircraft analysis is still overstated as an isolated interface effect")
    if "conditional on the fixed official Aircraft test set" not in supplement_text:
        fail("validation-size intervals omit the fixed-test-set condition")
    if "primary visual-component clustered paired results" in supplement_text.casefold():
        fail("archived historical analysis is still labeled primary")
    table_s6 = supplement.tables[5]
    crossing = 0
    for row in table_s6.rows[1:]:
        low, high = row.cells[4].text.strip("[] ").split(",")
        crossing += float(low) < 0 < float(high)
    if crossing != 2:
        fail(f"source-only compact-vs-full intervals should show two B/16 zero crossings; got {crossing}")
    if (len(supplement.tables[25].rows) != 7 or len(supplement.tables[26].rows) != 5 or
            len(supplement.tables) < 28):
        fail("compact Tables S26/S27 or post hoc Table S28 are missing")
    for doc, prefixes in ((main, ("Figure 1.", "Figure 2.", "Figure 3.", "Figure 4.", "Figure 5.")),
                          (supplement, ("Figure S1.", "Figure S2.", "Figure S3.", "Figure S4."))):
        for prefix in prefixes:
            caption = next((p for p in doc.paragraphs if p.text.strip().startswith(prefix)), None)
            if caption is None:
                fail(f"figure caption is missing: {prefix}")
            previous = caption._p.getprevious()
            if previous is None or not any(node.tag.endswith("}drawing") for node in previous.iter()):
                fail(f"figure image is detached from its caption: {prefix}")
    if len(supplement.inline_shapes) != 4:
        fail(f"Supplement must contain four embedded figure images, got {len(supplement.inline_shapes)}")
    rel_ns = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"
    # Figure 1's PNG must retain its aspect ratio and stay with its caption across page breaks.
    figure1_caption = next(p for p in main.paragraphs if p.text.strip().startswith("Figure 1."))
    figure1_node = figure1_caption._p.getprevious()
    figure1_blips = figure1_node.xpath(".//a:blip") if figure1_node is not None else []
    figure1_extents = figure1_node.xpath(".//wp:inline/wp:extent") if figure1_node is not None else []
    if len(figure1_blips) != 1 or len(figure1_extents) != 1:
        fail("Figure 1 drawing or size metadata is missing before its caption")
    figure1_rid = figure1_blips[0].get(rel_ns)
    figure1_part = main.part.related_parts.get(figure1_rid)
    if figure1_part is None or figure1_part.blob[:8] != b"\x89PNG\r\n\x1a\n":
        fail("Figure 1 image relationship is broken or is not a PNG")
    pixel_width, pixel_height = struct.unpack(">II", figure1_part.blob[16:24])
    extent = figure1_extents[0]
    drawn_ratio = int(extent.get("cx")) / int(extent.get("cy"))
    source_ratio = pixel_width / pixel_height
    if abs(drawn_ratio - source_ratio) > 0.01:
        fail("Figure 1 is stretched from its source aspect ratio")
    if (not figure1_node.xpath("./w:pPr/w:keepNext") or
            not figure1_node.xpath("./w:pPr/w:keepLines") or
            not figure1_caption._p.xpath("./w:pPr/w:keepLines")):
        fail("Figure 1 is not bound to its caption as a complete visual block")
    # Figure 2 follows a dense Results paragraph; preserve a visible gap and the complete plot-caption block.
    figure2_caption = next(p for p in main.paragraphs if p.text.strip().startswith("Figure 2."))
    figure2_node = figure2_caption._p.getprevious()
    spacing_nodes = figure2_node.xpath("./w:pPr/w:spacing") if figure2_node is not None else []
    before_twips = int(spacing_nodes[0].get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}before", "0")) if spacing_nodes else 0
    if (before_twips < 120 or not figure2_node.xpath("./w:pPr/w:keepNext") or
            not figure2_node.xpath("./w:pPr/w:keepLines") or
            not figure2_caption._p.xpath("./w:pPr/w:keepLines")):
        fail("Figure 2 lacks space after the preceding prose or is split from its caption")
    # Figures 3-5 must be fresh inline objects from the complete source PNGs and stay with their captions.
    main_figure_sources = {
        3: "Figure3_aircraft_curves_and_adjacent_uncertainty.png",
        4: "Figure4_validation_size_and_operating_point_mismatch.png",
        5: "Figure5_matched_descriptor_consumption_interfaces.png",
    }
    for number, filename in main_figure_sources.items():
        caption = next(p for p in main.paragraphs if p.text.strip().startswith(f"Figure {number}."))
        image_node = caption._p.getprevious()
        blips = image_node.xpath(".//a:blip") if image_node is not None else []
        extents = image_node.xpath(".//wp:inline/wp:extent") if image_node is not None else []
        if len(blips) != 1 or len(extents) != 1:
            fail(f"Figure {number} lacks one independent inline image before its caption")
        image_part = main.part.related_parts.get(blips[0].get(rel_ns))
        source = ROOT / "figures" / filename
        if image_part is None or not source.is_file() or hashlib.sha256(image_part.blob).hexdigest() != sha256(source):
            fail(f"Figure {number} does not match its complete source PNG")
        if (not image_node.xpath("./w:pPr/w:keepNext") or
                not image_node.xpath("./w:pPr/w:keepLines") or
                not caption._p.xpath("./w:pPr/w:keepLines")):
            fail(f"Figure {number} image and caption are not protected from page splitting")
        spacing = image_node.xpath("./w:pPr/w:spacing")
        before = int(spacing[0].get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}before", "0")) if spacing else 0
        if before < 120:
            fail(f"Figure {number} lacks a clear paragraph break above the image")
        pixel_width, pixel_height = struct.unpack(">II", image_part.blob[16:24])
        extent = extents[0]
        if abs(int(extent.get("cx")) / int(extent.get("cy")) - pixel_width / pixel_height) > 0.01:
            fail(f"Figure {number} image aspect ratio differs from the source PNG")
    if "and the how validation sampling aligns with evaluation" in main_text or "and how well validation sampling aligns with evaluation" not in main_text:
        fail("Discussion sentence on validation sampling is malformed")
    table_s10 = supplement.tables[9]
    if int(table_s10._tbl.tblGrid.gridCol_lst[1].w) < 900000:
        fail("Table S10 resource column is too narrow to preserve complete resource names")
    # Release consistency checks added after review of the paired package contents.
    duplicate_assets = ROOT / "machine_readable/source_only_resource/figures"
    if duplicate_assets.exists():
        fail("duplicate source-only scientific figure assets remain beside the authoritative SI figure")

    expected_labels = (
        (14, 6, 0, "Archived historical paired inference"),
        (16, 8, 0, "Classes covered by two-pass model-assisted factual audit"),
        (16, 9, 0, "Unresolved conflicts flagged by the audit"),
    )
    for table_index, row_index, column_index, expected in expected_labels:
        if supplement.tables[table_index].cell(row_index, column_index).text.strip() != expected:
            fail(f"Supplement table label is missing or overstated: {expected}")

    manuscript_lower = main_text.casefold()
    if ("calibration distributions" in manuscript_lower or
            "calibration/evaluation distribution" in manuscript_lower or
            "validation-sampling conditions" not in manuscript_lower or
            "validation-test alignment" not in manuscript_lower):
        fail("calibration-distribution terminology exceeds the validation-sampling design")

    # Each SI image must be the authoritative PNG shipped in figures_v14, not a stale duplicate.
    for number in range(1, 5):
        sources = list(ROOT.glob(f"figures_v14/FigureS{number}_*.png"))
        if len(sources) != 1:
            fail(f"expected one authoritative source image for Figure S{number}, found {len(sources)}")
        caption = next((p for p in supplement.paragraphs
                        if p.text.strip().startswith(f"Figure S{number}.")), None)
        previous = caption._p.getprevious() if caption is not None else None
        blips = previous.xpath(".//a:blip") if previous is not None else []
        if len(blips) != 1:
            fail(f"Figure S{number} does not have exactly one embedded image before its caption")
        rel_id = blips[0].get(rel_ns)
        image_part = supplement.part.related_parts.get(rel_id)
        if image_part is None:
            fail(f"Figure S{number} has a broken image relationship")
        embedded_sha = hashlib.sha256(image_part.blob).hexdigest()
        if embedded_sha != sha256(sources[0]):
            fail(f"embedded Figure S{number} differs from its authoritative PNG")

    equation_order = ["Equation (3)", "Equation (4)", "Equation (5)", "Equations (6)-(8)"]
    equation_positions = {}
    for index, paragraph in enumerate(main.paragraphs):
        for node in paragraph._p.xpath(".//wp:docPr"):
            title = node.get("title")
            if title in equation_order:
                equation_positions[title] = index
    if set(equation_positions) != set(equation_order):
        fail("equation image labels are incomplete; cannot verify visual equation order")
    if [equation_positions[label] for label in equation_order] != sorted(
            equation_positions[label] for label in equation_order):
        fail("displayed equations are not ordered (3), (4), (5), then (6)-(8)")

    external_readme = (ROOT / "external_aircraft_clean/README.md").read_text(encoding="utf-8")
    if "used in v12" in external_readme.casefold():
        fail("Aircraft resource README still refers to an old package version")
    for release_note in ("V14_RELEASE_AUDIT.md", "V14_CHANGELOG.md", "FIGURE_QA.md"):
        note = (ROOT / release_note).read_text(encoding="utf-8")
        if any(char in note for char in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"):
            fail(f"release note contains a non-ASCII dash glyph: {release_note}")
        if any(pattern in note for pattern in ("?0.54", "?2.11", "1?5", "S1?S4")):
            fail(f"release note contains a corrupted range or minus sign: {release_note}")

    analysis_readme = ROOT / "revision_experiments/analysis_package_v13/README_ANALYSIS_PACKAGE.md"
    if analysis_readme.is_file():
        active_readme = analysis_readme.read_text(encoding="utf-8").casefold()
        if "frozen protocol" in active_readme or "not a preregistration" not in active_readme:
            fail("active analysis-package README uses inconsistent protocol chronology")
    replay = ROOT / "REPLAY_SMOKE_TEST.md"
    if replay.is_file():
        replay_text = replay.read_text(encoding="utf-8").casefold()
        if "current raw replay smoke test" not in replay_text or "historical v12" in replay_text:
            fail("top-level replay record is not the current raw replay smoke test")
    print("PASS: manuscript/Supplement claim boundaries, TCA citation, fixed-Q condition, and compact/addendum tables.")


def verify_source_only_claims() -> None:
    claims = json.loads((ROOT / "machine_readable" / "paper_claims.json").read_text(encoding="utf-8"))
    source = read_csv("machine_readable/source_only_exploratory_intervals/source_only_exploratory_paired_intervals.csv",
                      ("backbone", "selector", "comparison", "accuracy_difference_candidate_minus_comparator",
                       "component_clustered_bootstrap_ci95_low", "component_clustered_bootstrap_ci95_high"), 8)
    expected = claims.get("source_only_exploratory_ci_pp", {})
    if len(expected) != 4:
        fail("paper_claims.json does not contain the four source-only compact-versus-full intervals")
    for key, target in expected.items():
        backbone, selector = key.split("|")
        rows = source[(source.backbone == backbone) & (source.selector == selector) &
                      (source.comparison == target["contrast"])]
        if len(rows) != 1:
            fail(f"source-only claim row missing or duplicated: {key}")
        row = rows.iloc[0]
        observed = [
            round(100 * row.accuracy_difference_candidate_minus_comparator, 2),
            round(100 * row.component_clustered_bootstrap_ci95_low, 2),
            round(100 * row.component_clustered_bootstrap_ci95_high, 2),
        ]
        wanted = [target["delta_pp"], *target["ci95_pp"]]
        if observed != wanted:
            fail(f"source-only claim differs from the authoritative CSV for {key}: {observed} != {wanted}")
    manuscript_text = " ".join(p.text for p in Document(ROOT / "Manuscript.docx").paragraphs)
    expected_prose = (
        "+8.33 pp (95% CI [+4.41, +12.22])",
        "+11.33 pp ([+7.10, +15.40])",
        "+3.00 pp ([-0.54, +6.66])",
        "+1.67 pp ([-2.11, +5.31])",
    )
    if not all(value in manuscript_text for value in expected_prose):
        fail("main-text source-only intervals differ from the corrected estimator values")
    print("PASS: the four source-only bootstrap intervals match the claim ledger.")


def verify_base_matched_addendum() -> None:
    protocol_path = ROOT / "revision_experiments/analysis_package_v13/PROTOCOL/EXP5_BASE_MATCHED_ADDENDUM_PROTOCOL_v13.json"
    result_root = ROOT / "revision_experiments/analysis_package_v13/RESULTS/experiment_5_base_matched_addendum"
    code_path = ROOT / "revision_experiments/analysis_package_v13/CODE/run_exp5_base_matched_addendum.py"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    manifest = json.loads((result_root / "run_manifest.json").read_text(encoding="utf-8"))
    if protocol.get("analysis_status") != "post_hoc_addendum_not_in_original_five_analysis_protocol" or manifest.get("analysis_status") != protocol.get("analysis_status"):
        fail("base-matched addendum is not distinctly labeled as post hoc")
    if sha256(code_path) != protocol.get("script_sha256") or sha256(code_path) != manifest.get("script_sha256", protocol.get("script_sha256")):
        fail("base-matched addendum code hash differs from its frozen protocol")
    if manifest.get("test_images_opened") is not False or manifest.get("device") != "cpu":
        fail("base-matched addendum did not follow the cached-feature/CPU boundary")
    predictions = read_csv("revision_experiments/analysis_package_v13/RESULTS/experiment_5_base_matched_addendum/base_matched_predictions.csv",
                           ("analysis_status", "backbone", "seed", "k", "image_id", "true_index", "predicted_index"), 133320)
    curves = read_csv("revision_experiments/analysis_package_v13/RESULTS/experiment_5_base_matched_addendum/base_matched_interface_curves.csv",
                      ("backbone", "seed", "k", "test_macro_accuracy", "analysis_status"), 40)
    if predictions.image_id.nunique() != 3333 or predictions.analysis_status.nunique() != 1:
        fail("addendum per-image predictions do not match the frozen test split/status")
    rows = []
    for key, group in predictions.groupby(["backbone", "seed", "k"], sort=True):
        macro = group.assign(correct=group.true_index.eq(group.predicted_index)).groupby("true_index").correct.mean().mean()
        rows.append((*key, macro))
    replay = pd.DataFrame(rows, columns=["backbone", "seed", "k", "macro"])
    compare = curves.merge(replay, on=["backbone", "seed", "k"], validate="one_to_one")
    if not np.allclose(compare.test_macro_accuracy, compare.macro, atol=1e-12, rtol=0):
        fail("addendum curves do not reproduce from saved per-image predictions")
    print("PASS: post hoc base-matched addendum protocol/hash and 133,320 predictions reproduce 40 curves.")


def verify_active_run_paths() -> None:
    active = [p for p in ROOT.rglob("run_manifest.json")
              if "provenance" not in p.relative_to(ROOT).parts]
    for path in active:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        serialized = json.dumps(manifest, ensure_ascii=False)
        if (chr(68)+":"+chr(92)) in serialized or (chr(67)+":"+chr(92)) in serialized or (chr(68)+":/") in serialized or (chr(67)+":/") in serialized:
            fail(f"active run manifest contains a machine-local absolute path: {path.relative_to(ROOT)}")
    mapping = json.loads((ROOT / "provenance" / "release_mapping.json").read_text(encoding="utf-8"))
    records = mapping.get("preserved_original_manifests", [])
    for record in records:
        backup = ROOT / record["provenance_copy"]
        if not backup.is_file():
            fail(f"original run-manifest provenance copy is missing: {record['provenance_copy']}")
        if hashlib.sha256(backup.read_bytes()).hexdigest() != record["original_sha256"]:
            fail(f"original run-manifest provenance copy hash differs: {record['provenance_copy']}")
    if not records:
        fail("release mapping does not record original run-manifest copies")
    print(f"PASS: {len(active)} active run manifests are portable; {len(records)} original copies match their recorded hashes.")


def replay_claims() -> None:
    command = [sys.executable, "-B", str(ROOT / "scripts" / "reproduce_paper_outputs.py"), "--validate-only"]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        fail("numerical paper-claim replay failed")
    sys.stdout.write(result.stdout)
    print("PASS: verifier invoked numerical replay; package was not modified.")


def replay_raw_revision_analyses() -> None:
    command = [sys.executable, "-B", str(ROOT / "scripts" / "recompute_revision_analyses_from_raw.py"), "--validate-only"]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        fail("full raw reviewer-analysis replay failed")
    sys.stdout.write(result.stdout)


def main() -> None:
    verify_hash_inventory()
    verify_protocol_and_schemas()
    verify_no_stale_claims_in_active_tree()
    verify_document_boundaries()
    verify_source_only_claims()
    verify_base_matched_addendum()
    verify_active_run_paths()
    replay_claims()
    replay_raw_revision_analyses()
    print("PASS: v14 release integrity and reproducibility checks complete.")


if __name__ == "__main__":
    main()
