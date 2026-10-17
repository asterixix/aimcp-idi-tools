import os
import json
import time
from pathlib import Path
from typing import Optional

from .config import CONFIG, logger
from .utils import _atomic_write_text, _atomic_write_json, _append_text, _safe_stem, enable_function_tracing


def load_documents(literature_dir: Optional[str] = None) -> list[dict]:
    lit_dir = literature_dir or CONFIG.get("literature_dir")
    lit_path = Path(lit_dir)
    documents = []
    if not lit_path.exists():
        logger.warning("Literature directory %s does not exist", lit_dir)
        return documents

    md_files = sorted(lit_path.glob("*.md"))
    for md_file in md_files:
        try:
            content = md_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                content = md_file.read_text(encoding="utf-8-sig")
            except Exception as e:
                logger.warning("Nie udało się wczytać %s: %s", md_file.name, e)
                continue

        if len(content.strip()) < CONFIG.get("min_content_chars", 100):
            logger.debug("Pominięto %s, za krótki dokument (%d chars)", md_file.name, len(content.strip()))
            continue
        if md_file.parent != lit_path:
            continue

        documents.append({
            "filename": md_file.name,
            "filepath": str(md_file),
            "content": content[: CONFIG.get("max_content_chars", 1000000)],
            "full_length": len(content),
            "truncated": len(content) > CONFIG.get("max_content_chars", 1000000),
        })

    logger.info("Wczytano %d dokumentów z %s", len(documents), lit_dir)
    return documents


def load_bibliografia(bib_path: str) -> Optional[str]:
    try:
        with open(bib_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.warning("Plik %s nie znaleziony — kontekst bibliograficzny niedostępny", bib_path)
        return None


def load_existing_codings(output_dir: str) -> list[dict]:
    aggregate_path = os.path.join(output_dir, "codings.json")
    loaded_codings: list[dict] = []
    unkeyed_codings: list[dict] = []
    by_filename: dict[str, dict] = {}

    if os.path.exists(aggregate_path):
        try:
            with open(aggregate_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                for item in loaded:
                    if not isinstance(item, dict):
                        continue
                    filename = item.get("_filename")
                    if filename:
                        by_filename[str(filename)] = item
                    else:
                        unkeyed_codings.append(item)
        except Exception as e:
            logger.warning("Nie udało się wczytać %s: %s", aggregate_path, e)

    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        if by_filename:
            return [*by_filename.values(), *unkeyed_codings]
        return unkeyed_codings

    for checkpoint_file in sorted(checkpoint_path.glob("*.json")):
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                filename = loaded.get("_filename")
                if filename:
                    by_filename[str(filename)] = loaded
                else:
                    unkeyed_codings.append(loaded)
        except Exception as e:
            logger.warning("Pomijam uszkodzony checkpoint %s: %s", checkpoint_file.name, e)

    if by_filename:
        return [*by_filename.values(), *unkeyed_codings]
    return unkeyed_codings


def save_coding_checkpoint(coding: dict, output_dir: str) -> str:
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    filename_part = _safe_stem(coding.get("_filename", "document"))
    pub_id_part = _safe_stem(str(coding.get("PUB_ID", "noid")))
    checkpoint_file = os.path.join(checkpoint_dir, f"{pub_id_part}__{filename_part}.json")
    _atomic_write_json(checkpoint_file, coding)
    return checkpoint_file


def save_memo_for_coding(coding: dict, output_dir: str) -> str:
    memos_dir = os.path.join(output_dir, "memos")
    os.makedirs(memos_dir, exist_ok=True)

    pub_id = _safe_stem(str(coding.get("PUB_ID", coding.get("_filename", "noid"))))
    filename_part = _safe_stem(coding.get("_filename", "document"))
    memo_file = os.path.join(memos_dir, f"{pub_id}__{filename_part}.md")

    title = coding.get("PUB_Title", "(brak tytułu)")
    keyfinding = coding.get("PUB_KeyFinding", "")
    focus = coding.get("DES_ResearchFocus", "")
    rq1 = coding.get("AN_RQ1_Relevance", "")
    rq2 = coding.get("AN_RQ2_Relevance", "")
    inclusion = coding.get("SLR_Inclusion", "")

    evidence = ""
    if "evidence_quotes" in coding and isinstance(coding["evidence_quotes"], list) and coding["evidence_quotes"]:
        evidence = coding["evidence_quotes"][0][:400]
    elif keyfinding:
        evidence = keyfinding[:400]

    memo_lines = [
        f"# Audit memo — {title}",
        "",
        f"- PUB_ID: {coding.get('PUB_ID', '')}",
        f"- Filename: {coding.get('_filename', '')}",
        f"- Chosen DES_ResearchFocus: {focus}",
        f"- AN_RQ1_Relevance: {rq1}",
        f"- AN_RQ2_Relevance: {rq2}",
        f"- SLR_Inclusion: {inclusion}",
        "",
        "## Auto-generated reason (short)",
    ]

    reason = ""
    if focus:
        reason = f"Focus '{focus}' selected based on evidence: {evidence[:200]}"
    else:
        reason = f"Brak jednoznacznego focusa; key finding: {evidence[:200]}"

    memo_lines.append(reason)
    memo_lines.append("")
    memo_lines.append("## Raw evidence (truncated)")
    memo_lines.append(evidence)

    _atomic_write_text(memo_file, "\n".join(memo_lines))
    return memo_file


def save_codings_aggregate(codings: list[dict], output_dir: str):
    aggregate_path = os.path.join(output_dir, "codings.json")
    _atomic_write_json(aggregate_path, codings)


def save_run_outputs(
    codings: list[dict],
    client,
    output_dir: str,
    bib_context: Optional[str],
    documents_total: int,
    failed: list[str],
    start_time: float,
    interrupted: bool = False,
    current_doc: Optional[str] = None,
):
    audit_result = None
    reliability_result = None

    if interrupted and current_doc:
        logger.warning("Przerwano przez Ctrl+C podczas: %s", current_doc)
    elif interrupted:
        logger.warning("Przerwano przez Ctrl+C.")

    save_codings_aggregate(codings, output_dir)

    if client is not None:
        log_path = CONFIG.get("llm_log_path") or os.path.join(output_dir, "llm_log.txt")
        try:
            _atomic_write_text(log_path, "\n".join(getattr(client, 'log_lines', [])))
            logger.info("Logi LLM zapisane: %s", log_path)
        except PermissionError as error:
            logger.warning("Nie udało się zapisać logów LLM do %s: %s", log_path, error)

    json_path = os.path.join(output_dir, "codings.json")
    logger.info("Kodowania JSON zapisane: %s", json_path)

    # QC skeleton
    if CONFIG.get("qc_integration", False):
        try:
            qc_dir = CONFIG.get("qc_project_dir")
            os.makedirs(qc_dir, exist_ok=True)
            settings_path = os.path.join(qc_dir, "settings.yaml")
            codebook_path = os.path.join(qc_dir, "codebook.yaml")
            corpus_dir = os.path.join(qc_dir, "corpus")
            os.makedirs(corpus_dir, exist_ok=True)
            settings = {
                "qc_version": "1.0",
                "corpus_dir": "corpus",
                "database": "qualitative_coding.sqlite3",
                "memos_dir": "memos",
                "codebook": "codebook.yaml",
                "editor": "code",
            }
            _atomic_write_text(settings_path, json.dumps(settings, ensure_ascii=False, indent=2))
            codes = "\n".join(sorted([k for k in []]))
            _atomic_write_text(codebook_path, codes)
            for c in codings:
                filename = c.get("_filename", c.get("PUB_ID", "noid"))
                safe = _safe_stem(filename) + ".txt"
                path = os.path.join(corpus_dir, safe)
                text = c.get("_source_text", "") or c.get("PUB_KeyFinding", "")
                _atomic_write_text(path, text)
            logger.info("qc project skeleton created: %s", qc_dir)
        except Exception as e:
            logger.exception("QC integration failed: %s", e)

    report_path = os.path.join(output_dir, "SLR_CODING_REPORT.md")
    # report generation may live in slr_core.report; keep call minimal here
    try:
        from . import report as _report

        _report.generate_report(codings, report_path, bib_context)
    except Exception:
        logger.debug("No report module available; skipping report generation")

    # quality audit
    try:
        from . import audit as _audit

        audit_result = _audit.build_quality_audit(codings, output_dir=output_dir)
    except Exception:
        logger.debug("No audit module available; skipping quality audit")

    # reliability layer from dual coding
    try:
        from . import reliability as _reliability

        reliability_result = _reliability.evaluate_reliability(codings, output_dir=output_dir)
    except Exception:
        logger.debug("No reliability module available; skipping reliability report")

    # review queue bootstrap for human verification workflow
    try:
        from . import review as _review

        _review.build_review_queue(force_refresh=False)
    except Exception:
        logger.debug("No review module available; skipping review queue sync")

    total_elapsed = time.time() - start_time
    logger.info("%s", "=" * 80)
    if interrupted:
        logger.info("KODOWANIE PRZERWANE")
    else:
        logger.info("KODOWANIE ZAKOŃCZONE")
    logger.info("  Zakodowano: %d/%d", len(codings), documents_total)
    logger.info("  Nieudane: %d/%d", len(failed), documents_total)
    if failed:
        logger.info("  Nieudane dokumenty: %s", ", ".join(failed[:5]) + ("..." if len(failed) > 5 else ""))
    logger.info("  Całkowity czas: %.1f min (%d s)", total_elapsed / 60, int(total_elapsed))
    if client is not None:
        try:
            logger.info("  Tokeny API: prompt=%s, completion=%s", f"{client.total_prompt_tokens:,}", f"{client.total_completion_tokens:,}")
            logger.info("  Żądania API: %d", getattr(client, "total_requests", 0))
        except Exception:
            logger.debug("No usage stats available on client")

    logger.info("%s", "=" * 80)
    return {"audit": audit_result, "reliability": reliability_result}


enable_function_tracing(globals(), __name__, include_private=True)
