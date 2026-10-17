"""Quality audit for coding outputs."""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional

from .config import CONFIG, logger
from .schema import CODING_DEFAULTS, missing_required_fields
from .utils import _atomic_write_json, _atomic_write_text, enable_function_tracing


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _coverage_metrics(coding: dict) -> tuple[int, int, float]:
    total = len(CODING_DEFAULTS.keys())
    filled = 0
    for key in CODING_DEFAULTS.keys():
        value = coding.get(key)
        if value not in (None, "", []):
            filled += 1
    pct = (filled / total * 100.0) if total else 0.0
    return filled, total, pct


def _uncertainty_score(coding: dict) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []

    missing = missing_required_fields(coding)
    if missing:
        penalty = min(40, len(missing) * 8)
        score += penalty
        reasons.append(f"braki pól wymaganych: {', '.join(missing)}")

    quotes = _as_list(coding.get("evidence_quotes"))
    min_quotes = int(CONFIG.get("min_evidence_quotes_critical", 2))
    if len(quotes) < min_quotes:
        score += 12
        reasons.append(f"mniej niż {min_quotes} cytaty dowodowe")

    notes = str(coding.get("notes", "")).lower()
    if "placeholder" in notes:
        score += 35
        reasons.append("kodowanie placeholder")
    elif "niepew" in notes or "uncertain" in notes:
        score += 15
        reasons.append("model sygnalizuje niepewność")

    if coding.get("SLR_Inclusion") == "wątpliwe":
        score += 10
        reasons.append("SLR_Inclusion = wątpliwe")

    if coding.get("_dual_disagreement"):
        if coding.get("_adjudication_fallback") or "adjudication_fallback" in notes:
            score += 20
            reasons.append("rozbieżność dual-coding (fallback adjudykacji)")
        else:
            score += 5
            reasons.append("rozbieżność dual-coding (rozwiązana adjudykacją)")

    faithfulness = coding.get("_faithfulness_score")
    try:
        if faithfulness is not None and float(faithfulness) < float(CONFIG.get("faithfulness_threshold", 0.7)):
            score += 25
            reasons.append("niski faithfulness")
    except Exception:
        pass

    if coding.get("_needs_manual_numeric_review"):
        score += 15
        reasons.append("high-risk numeric fields")

    try:
        rq1 = int(coding.get("AN_RQ1_Relevance", 0) or 0)
        rq2 = int(coding.get("AN_RQ2_Relevance", 0) or 0)
        if rq1 <= 1 and rq2 <= 1:
            score += 5
            reasons.append("niska relewancja dla obu pytań badawczych")
    except Exception:
        score += 5
        reasons.append("niepoprawne wartości AN_RQ1/AN_RQ2")

    return min(100, score), reasons


def build_quality_audit(codings: list[dict], *, output_dir: str) -> dict:
    rows = []
    threshold = int(CONFIG.get("quality_gate_uncertainty_threshold", 60))
    for coding in codings:
        filled, total, pct = _coverage_metrics(coding)
        uncertainty, reasons = _uncertainty_score(coding)
        rows.append(
            {
                "PUB_ID": coding.get("PUB_ID", ""),
                "PUB_Title": coding.get("PUB_Title", ""),
                "_filename": coding.get("_filename", ""),
                "fields_filled": filled,
                "fields_total": total,
                "coverage_pct": round(pct, 2),
                "uncertainty_score": uncertainty,
                "uncertainty_reasons": reasons,
                "inclusion": coding.get("SLR_Inclusion", ""),
                "_filepath": coding.get("_filepath", ""),
                "_needs_recoding": bool(coding.get("_needs_recoding", False)),
            }
        )

    rows_sorted = sorted(rows, key=lambda r: (-r["uncertainty_score"], r.get("PUB_ID", "")))
    avg_coverage = (sum(r["coverage_pct"] for r in rows_sorted) / len(rows_sorted)) if rows_sorted else 0.0
    avg_uncertainty = (sum(r["uncertainty_score"] for r in rows_sorted) / len(rows_sorted)) if rows_sorted else 0.0

    json_path = os.path.join(output_dir, "coding_quality_audit.json")
    _atomic_write_json(
        json_path,
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "documents": len(rows_sorted),
            "avg_coverage_pct": round(avg_coverage, 2),
            "avg_uncertainty_score": round(avg_uncertainty, 2),
            "items": rows_sorted,
        },
    )

    md_path = os.path.join(output_dir, "coding_quality_audit.md")
    lines = [
        "# Audyt jakości kodowania",
        "",
        f"- Dokumentów: **{len(rows_sorted)}**",
        f"- Średnie pokrycie pól: **{avg_coverage:.2f}%**",
        f"- Średni wskaźnik niepewności: **{avg_uncertainty:.2f}/100**",
        "",
        "## Dokumenty wymagające priorytetowego przeglądu",
    ]
    risky = [r for r in rows_sorted if r["uncertainty_score"] >= 40]
    if not risky:
        lines.append("- Brak dokumentów powyżej progu niepewności 40.")
    else:
        for row in risky[:30]:
            reasons = "; ".join(row["uncertainty_reasons"]) if row["uncertainty_reasons"] else "brak"
            lines.append(
                f"- {row['PUB_ID'] or row['_filename']}: niepewność={row['uncertainty_score']}, "
                f"pokrycie={row['coverage_pct']}%, powody: {reasons}"
            )
    _atomic_write_text(md_path, "\n".join(lines))

    queue_items = [
        {
            "PUB_ID": row["PUB_ID"],
            "PUB_Title": row["PUB_Title"],
            "_filename": row["_filename"],
            "_filepath": row["_filepath"],
            "uncertainty_score": row["uncertainty_score"],
            "uncertainty_reasons": row["uncertainty_reasons"],
        }
        for row in rows_sorted
        if row["uncertainty_score"] > threshold
        or (CONFIG.get("queue_include_needs_recoding_flag", False) and row.get("_needs_recoding", False))
    ]

    queue_json_path = CONFIG.get("quality_gate_queue_json") or os.path.join(output_dir, "recoding_queue.json")
    queue_txt_path = CONFIG.get("quality_gate_queue_txt") or os.path.join(output_dir, "recoding_queue.txt")

    _atomic_write_json(
        queue_json_path,
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "threshold": threshold,
            "count": len(queue_items),
            "items": queue_items,
        },
    )

    queue_lines = [
        f"# Rekodowanie: uncertainty_score > {threshold}",
        "",
    ]
    if not queue_items:
        queue_lines.append("Brak dokumentów wymagających rekodowania.")
    else:
        for item in queue_items:
            reasons = "; ".join(item["uncertainty_reasons"]) if item["uncertainty_reasons"] else "brak"
            queue_lines.append(
                f"- {item['PUB_ID'] or item['_filename']} | score={item['uncertainty_score']} | "
                f"file={item['_filename']} | reasons={reasons}"
            )
    _atomic_write_text(queue_txt_path, "\n".join(queue_lines))

    gate_failed = bool(queue_items) and bool(CONFIG.get("quality_gate_enabled", True))
    logger.info(
        "Quality audit saved: %s, %s; queue: %s (%d items, threshold=%d)",
        json_path,
        md_path,
        queue_json_path,
        len(queue_items),
        threshold,
    )
    return {
        "audit_json_path": json_path,
        "audit_md_path": md_path,
        "queue_json_path": queue_json_path,
        "queue_txt_path": queue_txt_path,
        "threshold": threshold,
        "queue_count": len(queue_items),
        "gate_failed": gate_failed,
    }


def run_quality_audit_from_existing() -> Optional[dict]:
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
    result = build_quality_audit(codings, output_dir=output_dir)
    try:
        from . import reliability as _reliability

        result["reliability"] = _reliability.evaluate_reliability(codings, output_dir=output_dir)
    except Exception:
        result["reliability"] = None
    return result


enable_function_tracing(globals(), __name__, include_private=True)

