"""Reliability layer (Krippendorff alpha) for dual coding outputs."""
from __future__ import annotations

import math
from datetime import datetime

from .config import CONFIG, logger
from .schema import normalize_inclusion_label
from .utils import _atomic_write_json, _atomic_write_text, enable_function_tracing


def _krippendorff_alpha_nominal(pairs: list[tuple[str, str]]) -> float:
    if not pairs:
        return float("nan")
    categories = sorted({a for a, _ in pairs} | {b for _, b in pairs})
    if len(categories) <= 1:
        return 1.0

    # observed disagreement Do: proportion of disagreements in coder pairs
    disagreements = sum(1 for a, b in pairs if a != b)
    do = disagreements / len(pairs)

    # expected disagreement De from pooled marginals
    counts = {cat: 0 for cat in categories}
    total = 0
    for a, b in pairs:
        counts[a] = counts.get(a, 0) + 1
        counts[b] = counts.get(b, 0) + 1
        total += 2
    if total <= 1:
        return float("nan")
    de = 1.0 - sum((counts[c] / total) ** 2 for c in categories)
    if de == 0:
        return 1.0
    return 1.0 - (do / de)


def evaluate_reliability(codings: list[dict], *, output_dir: str) -> dict:
    fields = CONFIG.get("dual_compare_fields") or []
    per_field: dict[str, dict] = {}
    pair_docs = [c for c in codings if isinstance(c.get("_coder_a"), dict) and isinstance(c.get("_coder_b"), dict)]

    for field in fields:
        pairs: list[tuple[str, str]] = []
        for coding in pair_docs:
            a = coding.get("_coder_a", {}).get(field)
            b = coding.get("_coder_b", {}).get(field)
            if a is None or b is None:
                continue
            if field == "SLR_Inclusion":
                a = normalize_inclusion_label(a)
                b = normalize_inclusion_label(b)
            pairs.append((str(a), str(b)))
        alpha = _krippendorff_alpha_nominal(pairs)
        per_field[field] = {
            "pairs": len(pairs),
            "krippendorff_alpha_nominal": None if math.isnan(alpha) else round(alpha, 4),
        }

    # global alpha using primary inclusion label for conservative gate
    inclusion_pairs = []
    for coding in pair_docs:
        a = coding.get("_coder_a", {}).get("SLR_Inclusion")
        b = coding.get("_coder_b", {}).get("SLR_Inclusion")
        if a is not None and b is not None:
            inclusion_pairs.append((normalize_inclusion_label(a), normalize_inclusion_label(b)))
    inclusion_alpha = _krippendorff_alpha_nominal(inclusion_pairs)
    threshold = float(CONFIG.get("reliability_threshold", 0.67))
    gate_failed = False
    if CONFIG.get("reliability_gate_enabled", True):
        if not math.isnan(inclusion_alpha) and inclusion_alpha < threshold:
            gate_failed = True

    out = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "sample_documents": len(pair_docs),
        "inclusion_pairs": len(inclusion_pairs),
        "inclusion_alpha": None if math.isnan(inclusion_alpha) else round(inclusion_alpha, 4),
        "threshold": threshold,
        "gate_failed": gate_failed,
        "per_field": per_field,
    }

    json_path = CONFIG.get("reliability_report_json")
    md_path = CONFIG.get("reliability_report_md")
    _atomic_write_json(json_path, out)
    lines = [
        "# Reliability layer (Krippendorff alpha)",
        "",
        f"- Sample documents (dual-coded): **{len(pair_docs)}**",
        f"- Inclusion alpha: **{out['inclusion_alpha']}**",
        f"- Threshold: **{threshold}**",
        f"- Gate failed: **{gate_failed}**",
        "",
        "## Per-field alpha",
    ]
    for field, details in per_field.items():
        lines.append(
            f"- {field}: alpha={details['krippendorff_alpha_nominal']} (pairs={details['pairs']})"
        )
    _atomic_write_text(md_path, "\n".join(lines))
    logger.info("Reliability report saved: %s, %s", json_path, md_path)
    return out


enable_function_tracing(globals(), __name__, include_private=True)

