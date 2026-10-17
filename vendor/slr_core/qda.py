"""QDA export utilities."""
from __future__ import annotations

import csv
import json
import os
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

from .config import CONFIG, logger
from .schema import CODING_DEFAULTS
from .utils import _atomic_write_json, _atomic_write_text, enable_function_tracing


META_FIELDS = {
    "PUB_ID",
    "PUB_Author",
    "PUB_Year",
    "PUB_Title",
    "PUB_Journal",
    "PUB_DOI",
    "PUB_Type",
    "PUB_Domain",
    "PUB_Method",
    "PUB_Sample",
    "PUB_KeyFinding",
}


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _write_csv(path: str, rows: list[dict], fieldnames: list[str], *, encoding: str = "utf-8-sig") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding=encoding) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _safe_name(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "UNTITLED"
    safe = []
    for char in text:
        if char.isalnum() or char in ("_", "-", "."):
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe)[:120] or "UNTITLED"


def _short_var_name(name: str) -> str:
    cleaned = "".join(ch for ch in str(name or "") if ch.isalnum() or ch == "_")
    if not cleaned:
        cleaned = "VAR"
    if not cleaned[0].isalpha():
        cleaned = f"V_{cleaned}"
    return cleaned[:10]


def _coerce_float(value) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return 0.0
    text = text.replace(",", ".")
    # Try direct numeric parse
    try:
        return float(text)
    except ValueError:
        pass
    # Try first numeric token from free text
    import re

    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if match:
        try:
            return float(match.group(0))
        except ValueError:
            return 0.0
    return 0.0


def _build_numeric_qda_matrix(codings: list[dict]) -> tuple[list[list[float]], list[str], list[str]]:
    """Build numeric matrix suitable for qdafile serialization."""
    numeric_fields = [
        "PUB_Year",
        "DES_MethodRigor",
        "DES_Replicability",
        "AN_RQ1_Relevance",
        "AN_QFSR",
        "AN_SC",
        "AN_VF",
        "AN_RI",
        "AN_RQ2_Relevance",
        "AN_HR",
        "AN_SSS",
        "AN_CER",
        "AN_CopyTheory",
        "AN_Floridi",
        "AN_SemanticLaundering",
        "AN_Intermediation",
        "AN_KOS",
        "SYN_ProjectRelevance",
        "SYN_SynthesisPotential",
        "STR_CitationPriority",
        "STR_HS2_Framing",
        "_faithfulness_score",
        "_dual_agreement_score",
        "_needs_recoding",
        "_needs_manual_numeric_review",
        "_dual_disagreement",
    ]
    headers = ["doc_index", *numeric_fields]
    rows_labels: list[str] = []
    matrix: list[list[float]] = []

    for idx, coding in enumerate(codings, 1):
        rows_labels.append(str(coding.get("PUB_ID") or coding.get("_filename") or f"DOC_{idx:04d}"))
        row = [float(idx)]
        for field in numeric_fields:
            row.append(_coerce_float(coding.get(field)))
        matrix.append(row)
    return matrix, headers, rows_labels


def _export_qdafile(codings: list[dict], qda_dir: str) -> Optional[str]:
    if not CONFIG.get("qda_generate_legacy_qda", False):
        return None
    try:
        from qdafile import QDAfile  # type: ignore
    except Exception as exc:
        logger.warning("qdafile not available, .qda export skipped: %s", exc)
        return None

    matrix, headers, rows_labels = _build_numeric_qda_matrix(codings)
    qda_path = os.path.join(qda_dir, "kaleidagraph_numeric.qda")
    # qdafile expects list-of-columns. Transform row-major matrix to column-major.
    columns = list(map(list, zip(*matrix))) if matrix else [[] for _ in headers]
    qda_obj = QDAfile(columns, headers=headers, rows=[len(matrix)] * len(headers))
    qda_obj.write(qda_path)
    logger.info("Legacy QDA (KaleidaGraph) file exported with qdafile: %s", qda_path)
    return qda_path


def _export_xml(codings: list[dict], qda_dir: str, docs_dir: str) -> str:
    xml_path = os.path.join(qda_dir, "codings_project.xml")
    root = ET.Element("SLRQDAProject")
    root.set("generated_at", datetime.now().isoformat(timespec="seconds"))
    root.set("documents_count", str(len(codings)))

    documents_el = ET.SubElement(root, "Documents")
    codes_el = ET.SubElement(root, "Codes")
    applications_el = ET.SubElement(root, "CodeApplications")

    for field in CODING_DEFAULTS.keys():
        code_el = ET.SubElement(codes_el, "Code")
        code_el.set("name", str(field))

    for index, coding in enumerate(codings, 1):
        pub_id = str(coding.get("PUB_ID") or f"DOC_{index:04d}")
        title = str(coding.get("PUB_Title") or pub_id)
        filename = str(coding.get("_filename") or f"{pub_id}.md")
        doc_txt_name = f"{pub_id}.txt"
        source_text = str(coding.get("_source_text") or coding.get("PUB_KeyFinding") or "")
        doc_txt_path = os.path.join(docs_dir, doc_txt_name)
        if not os.path.exists(doc_txt_path):
            _atomic_write_text(doc_txt_path, source_text)

        doc_el = ET.SubElement(documents_el, "Document")
        doc_el.set("id", pub_id)
        doc_el.set("title", title)
        doc_el.set("filename", filename)
        doc_el.set("source_text_file", f"documents/{doc_txt_name}")
        doc_el.set("inclusion", str(coding.get("SLR_Inclusion", "")))

        for field in CODING_DEFAULTS.keys():
            if field in META_FIELDS:
                continue
            value = coding.get(field)
            if isinstance(value, list):
                serialized = "; ".join(str(x) for x in value)
            else:
                serialized = str(value) if value is not None else ""
            if not serialized.strip():
                continue
            app_el = ET.SubElement(applications_el, "Application")
            app_el.set("doc_id", pub_id)
            app_el.set("code", field)
            value_el = ET.SubElement(app_el, "Value")
            value_el.text = serialized
            quotes = _as_list(coding.get("evidence_quotes"))
            if quotes:
                quote_el = ET.SubElement(app_el, "EvidenceQuote")
                quote_el.text = quotes[0]

    tree = ET.ElementTree(root)
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    logger.info("QDA XML exported: %s", xml_path)
    return xml_path


def _export_qdpx(xml_path: str, qda_dir: str, docs_dir: str) -> str:
    qdpx_path = os.path.join(qda_dir, "codings_project.qdpx")
    with zipfile.ZipFile(qdpx_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(xml_path, arcname="project.xml")
        if os.path.isdir(docs_dir):
            for file_name in sorted(os.listdir(docs_dir)):
                full_path = os.path.join(docs_dir, file_name)
                if os.path.isfile(full_path):
                    archive.write(full_path, arcname=f"sources/{file_name}")
    logger.info("QDPX package exported: %s", qdpx_path)
    return qdpx_path


def _export_ris(codings: list[dict], qda_dir: str) -> str:
    ris_path = os.path.join(qda_dir, "references.ris")
    lines: list[str] = []
    for coding in codings:
        pub_type = str(coding.get("PUB_Type", "") or "").lower()
        ris_type = "JOUR"
        if "book" in pub_type:
            ris_type = "BOOK"
        elif "conference" in pub_type or "paper" in pub_type:
            ris_type = "CPAPER"
        lines.append(f"TY  - {ris_type}")
        title = str(coding.get("PUB_Title", "") or "").strip()
        if title:
            lines.append(f"TI  - {title}")
        author = str(coding.get("PUB_Author", "") or "").strip()
        if author:
            for a in [x.strip() for x in author.split(";") if x.strip()]:
                lines.append(f"AU  - {a}")
        year = str(coding.get("PUB_Year", "") or "").strip()
        if year:
            lines.append(f"PY  - {year}")
        journal = str(coding.get("PUB_Journal", "") or "").strip()
        if journal:
            lines.append(f"JO  - {journal}")
        doi = str(coding.get("PUB_DOI", "") or "").strip()
        if doi:
            lines.append(f"DO  - {doi}")
            lines.append(f"UR  - https://doi.org/{doi}")
        pub_id = str(coding.get("PUB_ID", "") or "").strip()
        if pub_id:
            lines.append(f"ID  - {pub_id}")
        lines.append("ER  - ")
        lines.append("")
    _atomic_write_text(ris_path, "\n".join(lines))
    logger.info("RIS references exported: %s", ris_path)
    return ris_path


def _zip_profile_dir(profile_dir: str) -> str:
    zip_path = f"{profile_dir}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for root, _, files in os.walk(profile_dir):
            for file_name in files:
                full_path = os.path.join(root, file_name)
                rel = os.path.relpath(full_path, profile_dir)
                archive.write(full_path, arcname=rel)
    return zip_path


def _write_vendor_readme(path: str, lines: list[str]) -> None:
    _atomic_write_text(path, "\n".join(lines))


def _export_vendor_profiles(
    *,
    qda_dir: str,
    docs_dir: str,
    documents_rows: list[dict],
    coding_wide_rows: list[dict],
    code_applications_rows: list[dict],
    xml_path: str,
    qdpx_path: str,
    ris_path: str,
) -> dict:
    profiles_root = os.path.join(qda_dir, "profiles")
    os.makedirs(profiles_root, exist_ok=True)

    # MAXQDA profile (REFI-QDA first)
    maxqda_dir = os.path.join(profiles_root, "MAXQDA")
    os.makedirs(maxqda_dir, exist_ok=True)
    shutil.copy2(qdpx_path, os.path.join(maxqda_dir, os.path.basename(qdpx_path)))
    shutil.copy2(xml_path, os.path.join(maxqda_dir, os.path.basename(xml_path)))
    shutil.copy2(ris_path, os.path.join(maxqda_dir, os.path.basename(ris_path)))
    _write_csv(
        os.path.join(maxqda_dir, "maxqda_document_variables.csv"),
        coding_wide_rows,
        ["doc_id", "title", *CODING_DEFAULTS.keys()],
    )
    codebook_rows = [{"code": key} for key in CODING_DEFAULTS.keys() if key not in META_FIELDS]
    _write_csv(os.path.join(maxqda_dir, "maxqda_codebook.csv"), codebook_rows, ["code"])
    _write_vendor_readme(
        os.path.join(maxqda_dir, "README_MAXQDA.md"),
        [
            "# MAXQDA export profile",
            "",
            "- Primary path: import `codings_project.qdpx` as REFI-QDA Project.",
            "- Backup path: import `maxqda_document_variables.csv` as document variables.",
            "- Optional bibliographic import: `references.ris`.",
            "- Source texts are in `../documents/`.",
        ],
    )

    # ATLAS.ti profile (REFI-QDA + coding table fallback)
    atlas_dir = os.path.join(profiles_root, "ATLAS_ti")
    os.makedirs(atlas_dir, exist_ok=True)
    shutil.copy2(qdpx_path, os.path.join(atlas_dir, os.path.basename(qdpx_path)))
    shutil.copy2(xml_path, os.path.join(atlas_dir, os.path.basename(xml_path)))
    shutil.copy2(ris_path, os.path.join(atlas_dir, os.path.basename(ris_path)))
    _write_csv(
        os.path.join(atlas_dir, "atlas_code_applications.csv"),
        code_applications_rows,
        ["doc_id", "code", "value", "evidence_quote"],
    )
    _write_vendor_readme(
        os.path.join(atlas_dir, "README_ATLAS_ti.md"),
        [
            "# ATLAS.ti export profile",
            "",
            "- Primary path: import `codings_project.qdpx` (REFI-QDA).",
            "- Fallback: import `atlas_code_applications.csv` for code/value matrix alignment.",
            "- Optional bibliographic import: `references.ris`.",
            "- Source texts are in `../documents/`.",
        ],
    )

    # NVivo profile (structured CSV + sources)
    nvivo_dir = os.path.join(profiles_root, "NVivo")
    os.makedirs(nvivo_dir, exist_ok=True)
    nvivo_cases_rows = []
    nvivo_codes_rows = []
    for row in documents_rows:
        nvivo_cases_rows.append(
            {
                "CaseName": row.get("doc_id", ""),
                "Title": row.get("title", ""),
                "SourceFile": row.get("source_text_file", ""),
                "Inclusion": row.get("inclusion", ""),
            }
        )
    for app in code_applications_rows:
        nvivo_codes_rows.append(
            {
                "CaseName": app.get("doc_id", ""),
                "NodeName": app.get("code", ""),
                "CodedValue": app.get("value", ""),
                "ReferenceText": app.get("evidence_quote", ""),
            }
        )
    _write_csv(
        os.path.join(nvivo_dir, "nvivo_cases.csv"),
        nvivo_cases_rows,
        ["CaseName", "Title", "SourceFile", "Inclusion"],
    )
    _write_csv(
        os.path.join(nvivo_dir, "nvivo_codes.csv"),
        nvivo_codes_rows,
        ["CaseName", "NodeName", "CodedValue", "ReferenceText"],
    )
    _write_vendor_readme(
        os.path.join(nvivo_dir, "README_NVivo.md"),
        [
            "# NVivo export profile",
            "",
            "- Import `../documents/` as sources.",
            "- Import `nvivo_cases.csv` as case classifications.",
            "- Import `nvivo_codes.csv` as coding helper matrix for node assignments.",
            "- Optional bibliographic import: `../references.ris` (if using external reference workflow).",
        ],
    )

    # QDA Miner profile (prefer REFI-QDA package, plus direct CSV import set)
    qdaminer_dir = os.path.join(profiles_root, "QDA_Miner")
    os.makedirs(qdaminer_dir, exist_ok=True)
    shutil.copy2(qdpx_path, os.path.join(qdaminer_dir, os.path.basename(qdpx_path)))
    shutil.copy2(xml_path, os.path.join(qdaminer_dir, os.path.basename(xml_path)))
    shutil.copy2(ris_path, os.path.join(qdaminer_dir, os.path.basename(ris_path)))
    _write_csv(
        os.path.join(qdaminer_dir, "qdaminer_documents.csv"),
        documents_rows,
        ["doc_id", "title", "filename", "source_text_file", "inclusion"],
    )
    _write_csv(
        os.path.join(qdaminer_dir, "qdaminer_codes.csv"),
        code_applications_rows,
        ["doc_id", "code", "value", "evidence_quote"],
    )
    qdm_short_rows = []
    for row in documents_rows:
        qdm_short_rows.append(
            {
                _short_var_name("doc_id"): row.get("doc_id", ""),
                _short_var_name("title"): row.get("title", ""),
                _short_var_name("source_text_file"): row.get("source_text_file", ""),
                _short_var_name("inclusion"): row.get("inclusion", ""),
            }
        )
    qdm_short_fields = [_short_var_name("doc_id"), _short_var_name("title"), _short_var_name("source_text_file"), _short_var_name("inclusion")]
    _write_csv(
        os.path.join(qdaminer_dir, "qdaminer_variables_short.csv"),
        qdm_short_rows,
        qdm_short_fields,
    )
    _write_vendor_readme(
        os.path.join(qdaminer_dir, "README_QDA_Miner.md"),
        [
            "# QDA Miner export profile",
            "",
            "- Prefer REFI-QDA import using `codings_project.qdpx`.",
            "- Do not open `.qda` legacy binary directly as a project.",
            "- Fallback A: import `qdaminer_documents.csv` + texts from `../documents/`.",
            "- Fallback B: import bibliographic data from `references.ris`.",
            "- `qdaminer_variables_short.csv` uses variable names <=10 chars for strict compatibility.",
        ],
    )

    for profile in (maxqda_dir, atlas_dir, nvivo_dir, qdaminer_dir):
        _zip_profile_dir(profile)

    return {
        "profiles_root": "profiles/",
        "maxqda_dir": "profiles/MAXQDA/",
        "maxqda_zip": "profiles/MAXQDA.zip",
        "atlas_ti_dir": "profiles/ATLAS_ti/",
        "atlas_ti_zip": "profiles/ATLAS_ti.zip",
        "nvivo_dir": "profiles/NVivo/",
        "nvivo_zip": "profiles/NVivo.zip",
        "qda_miner_dir": "profiles/QDA_Miner/",
        "qda_miner_zip": "profiles/QDA_Miner.zip",
    }


def export_qda_package(codings: list[dict], *, output_dir: str) -> str:
    qda_dir = os.path.join(output_dir, "QDA_EXPORT")
    docs_dir = os.path.join(qda_dir, "documents")
    os.makedirs(docs_dir, exist_ok=True)

    documents_rows: list[dict] = []
    coding_wide_rows: list[dict] = []
    code_applications_rows: list[dict] = []

    for index, coding in enumerate(codings, 1):
        pub_id = str(coding.get("PUB_ID") or f"DOC_{index:04d}")
        title = str(coding.get("PUB_Title") or pub_id)
        filename = str(coding.get("_filename") or f"{pub_id}.md")
        source_text = str(coding.get("_source_text") or coding.get("PUB_KeyFinding") or "")

        doc_txt_name = f"{pub_id}.txt"
        doc_txt_path = os.path.join(docs_dir, doc_txt_name)
        _atomic_write_text(doc_txt_path, source_text)

        documents_rows.append(
            {
                "doc_id": pub_id,
                "title": title,
                "filename": filename,
                "source_text_file": f"documents/{doc_txt_name}",
                "inclusion": coding.get("SLR_Inclusion", ""),
            }
        )

        # wide coding table for spreadsheet/QDA pre-import checks
        wide_row = {"doc_id": pub_id, "title": title}
        for field in CODING_DEFAULTS.keys():
            value = coding.get(field)
            if isinstance(value, list):
                wide_row[field] = "; ".join(str(x) for x in value)
            else:
                wide_row[field] = value
        coding_wide_rows.append(wide_row)

        # long code applications table
        for field in CODING_DEFAULTS.keys():
            value = coding.get(field)
            if field in META_FIELDS:
                continue
            if isinstance(value, list):
                serialized = "; ".join(str(x) for x in value)
            else:
                serialized = value
            if serialized in (None, "", []):
                continue
            code_applications_rows.append(
                {
                    "doc_id": pub_id,
                    "code": field,
                    "value": serialized,
                    "evidence_quote": (_as_list(coding.get("evidence_quotes"))[0] if _as_list(coding.get("evidence_quotes")) else ""),
                }
            )

    documents_csv = os.path.join(qda_dir, "documents.csv")
    codings_csv = os.path.join(qda_dir, "codings_wide.csv")
    applications_csv = os.path.join(qda_dir, "code_applications.csv")
    metadata_json = os.path.join(qda_dir, "export_meta.json")
    binary_qda_path = _export_qdafile(codings, qda_dir)

    _write_csv(documents_csv, documents_rows, ["doc_id", "title", "filename", "source_text_file", "inclusion"])
    _write_csv(codings_csv, coding_wide_rows, ["doc_id", "title", *CODING_DEFAULTS.keys()])
    _write_csv(applications_csv, code_applications_rows, ["doc_id", "code", "value", "evidence_quote"])
    xml_path = _export_xml(codings, qda_dir, docs_dir)
    qdpx_path = _export_qdpx(xml_path, qda_dir, docs_dir)
    ris_path = _export_ris(codings, qda_dir)
    vendor_profiles = _export_vendor_profiles(
        qda_dir=qda_dir,
        docs_dir=docs_dir,
        documents_rows=documents_rows,
        coding_wide_rows=coding_wide_rows,
        code_applications_rows=code_applications_rows,
        xml_path=xml_path,
        qdpx_path=qdpx_path,
        ris_path=ris_path,
    )

    _atomic_write_json(
        metadata_json,
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "documents": len(documents_rows),
            "codes_defined": len(CODING_DEFAULTS.keys()),
            "files": {
                "documents_csv": "documents.csv",
                "codings_wide_csv": "codings_wide.csv",
                "code_applications_csv": "code_applications.csv",
                "documents_dir": "documents/",
                "legacy_kaleidagraph_qda_file": os.path.basename(binary_qda_path) if binary_qda_path else None,
                "xml_file": os.path.basename(xml_path),
                "qdpx_file": os.path.basename(qdpx_path),
                "references_ris": os.path.basename(ris_path),
                **vendor_profiles,
            },
        },
    )

    readme_path = os.path.join(qda_dir, "README_QDA.md")
    _atomic_write_text(
        readme_path,
        "\n".join(
            [
                "# QDA export package",
                "",
                "Pakiet zawiera pliki gotowe do importu/analityki w narzędziach QDA (np. przez CSV):",
                "",
                "- `documents.csv` — lista dokumentów i ścieżki do tekstów źródłowych.",
                "- `codings_wide.csv` — szeroka tabela kodowania (1 wiersz = 1 dokument).",
                "- `code_applications.csv` — długa tabela aplikacji kodów (1 wiersz = 1 kod).",
                "- `documents/` — teksty źródłowe per dokument.",
                "- `codings_project.xml` — XML projektu QDA z dokumentami, kodami i aplikacjami kodów.",
                "- `codings_project.qdpx` — paczka QDPX (ZIP) zawierająca `project.xml` i `sources/`.",
                "- `references.ris` — bibliografia do importu przez menedżery referencji/QDA.",
                "- `kaleidagraph_numeric.qda` — opcjonalny plik legacy (KaleidaGraph), **nie** projekt QDA Miner.",
                "- `profiles/` — dedykowane profile importu: MAXQDA, ATLAS.ti, NVivo, QDA Miner.",
                "- `profiles/*.zip` — gotowe paczki per narzędzie.",
                "",
                "Rekomendacja importu:",
                "0. QDA Miner: importuj przez REFI-QDA (`codings_project.qdpx`), nie otwieraj pliku `.qda`.",
                "1. Zaimportuj `documents.csv` jako metadane dokumentów.",
                "2. Zaimportuj pliki z `documents/` jako korpus tekstowy.",
                "3. Zaimportuj `code_applications.csv` jako tabelę kodów pomocniczą do mapowania.",
                "",
                "Jeśli chcesz wygenerować legacy `kaleidagraph_numeric.qda`, ustaw `qda_generate_legacy_qda=true` i doinstaluj:",
                "- `pip install qdafile`",
            ]
        ),
    )
    logger.info("QDA export saved: %s", qda_dir)
    return qda_dir


def export_qda_from_existing() -> Optional[str]:
    output_dir = CONFIG.get("output_dir")
    codings_path = os.path.join(output_dir, "codings.json")
    if not os.path.exists(codings_path):
        logger.warning("Brak pliku %s", codings_path)
        return None
    try:
        with open(codings_path, "r", encoding="utf-8") as handle:
            codings = json.load(handle)
        if not isinstance(codings, list):
            logger.warning("codings.json ma niepoprawny format (oczekiwano listy).")
            return None
    except Exception as exc:
        logger.exception("Nie udało się wczytać codings.json: %s", exc)
        return None
    return export_qda_package(codings, output_dir=output_dir)


enable_function_tracing(globals(), __name__, include_private=True)

