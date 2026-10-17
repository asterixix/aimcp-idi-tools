"""Schema definitions and normalization for SLR coding outputs."""
from __future__ import annotations

from copy import deepcopy
from typing import Any
from .utils import enable_function_tracing


CODING_DEFAULTS: dict[str, Any] = {
    # publication metadata
    "PUB_ID": "",
    "PUB_Author": "",
    "PUB_Year": "",
    "PUB_Title": "",
    "PUB_Journal": "",
    "PUB_DOI": "",
    "PUB_Type": "teoretyczna",
    "PUB_Domain": "nauki_o_informacji",
    "PUB_Method": "",
    "PUB_Sample": "",
    "PUB_KeyFinding": "",
    # descriptive
    "DES_ResearchFocus": "inna",
    "DES_DataContext": "inny",
    "DES_MetadataSchema": [],
    "DES_LanguageContext": "nie_dotyczy",
    "DES_MethodRigor": 1,
    "DES_Replicability": 1,
    "DES_Comparative": "nie",
    "DES_Longitudinal": "nie",
    # analytical RQ1/RQ2
    "AN_RQ1_Relevance": 0,
    "AN_QFSR": 0,
    "AN_SC": 0,
    "AN_VF": 0,
    "AN_RI": 0,
    "AN_RQ2_Relevance": 0,
    "AN_HR": 0,
    "AN_SSS": 0,
    "AN_CER": 0,
    # theoretical mappings
    "AN_IQ_Framework": ["brak"],
    "AN_CopyTheory": 0,
    "AN_Floridi": 0,
    "AN_SemanticLaundering": 0,
    "AN_Intermediation": 0,
    "AN_KOS": 0,
    # synthesis
    "SYN_GapType": "brak_luki",
    "SYN_GapDescription": "",
    "SYN_ProjectRelevance": 0,
    "SYN_Contradiction": "brak",
    "SYN_ContradictionRefs": [],
    "SYN_Cluster": "metodologia",
    "SYN_Theme": "",
    "SYN_SynthesisPotential": 0,
    # SLR decision
    "SLR_Cluster": [],
    "SLR_Inclusion": "wątpliwe",
    "SLR_InclusionReason": "",
    # results
    "RES_DV": "",
    "RES_IV": "",
    "RES_N": "",
    "RES_EffectSize": "",
    "RES_KeyStat": "",
    "RES_Limitation": "",
    "RES_Framework": "",
    "RES_Proposition": "",
    "RES_Definition": "",
    # strategic
    "STR_CitationPriority": 2,
    "STR_SectionUsage": [],
    "STR_HS2_Framing": 2,
    "STR_RiskNote": "brak",
    # explainability
    "evidence_quotes": [],
    "notes": "",
}


INT_FIELDS = {
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
}

BOUNDED_INT_FIELDS: dict[str, tuple[int, int]] = {
    "AN_RQ1_Relevance": (0, 3),
    "AN_RQ2_Relevance": (0, 3),
    "AN_QFSR": (0, 2),
    "AN_SC": (0, 2),
    "AN_VF": (0, 2),
    "AN_RI": (0, 2),
    "AN_HR": (0, 2),
    "AN_SSS": (0, 2),
    "AN_CER": (0, 2),
}

LIST_FIELDS = {
    "DES_MetadataSchema",
    "AN_IQ_Framework",
    "SYN_ContradictionRefs",
    "SLR_Cluster",
    "STR_SectionUsage",
    "evidence_quotes",
}


def _clamp_int(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def normalize_inclusion_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return "wątpliwe"
    if text in {"włączyć", "wlaczyc", "include", "included", "tak", "yes"}:
        return "włączyć"
    if text in {"wykluczyć", "wykluczyc", "exclude", "excluded", "nie", "no"}:
        return "wykluczyć"
    if text in {"wątpliwe", "watpliwe", "uncertain", "maybe"}:
        return "wątpliwe"
    return "wątpliwe"


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        value = value.strip()
        if value.isdigit():
            return int(value)
    return default


def normalize_coding(raw: dict[str, Any], *, pub_id: str, title: str) -> dict[str, Any]:
    """Coerce LLM output into a stable schema-friendly coding payload."""
    data = deepcopy(CODING_DEFAULTS)
    data.update(raw or {})
    data["PUB_ID"] = data.get("PUB_ID") or pub_id
    data["PUB_Title"] = data.get("PUB_Title") or title

    for field in INT_FIELDS:
        data[field] = _as_int(data.get(field), _as_int(CODING_DEFAULTS.get(field), 0))
    for field, (lower, upper) in BOUNDED_INT_FIELDS.items():
        data[field] = _clamp_int(_as_int(data.get(field), 0), lower, upper)

    for field in LIST_FIELDS:
        value = data.get(field)
        if value is None:
            data[field] = []
        elif isinstance(value, list):
            data[field] = [str(x).strip() for x in value if str(x).strip()]
        else:
            # Accept semicolon/comma-delimited strings from weaker models.
            text = str(value)
            split = text.replace(";", ",").split(",")
            data[field] = [chunk.strip() for chunk in split if chunk.strip()]

    # Keep short scalar fields tidy.
    for key, value in list(data.items()):
        if key in LIST_FIELDS:
            continue
        if isinstance(value, str):
            data[key] = value.strip()

    data["SLR_Inclusion"] = normalize_inclusion_label(data.get("SLR_Inclusion"))

    return data


def missing_required_fields(coding: dict[str, Any]) -> list[str]:
    required = [
        "PUB_ID",
        "PUB_Title",
        "PUB_KeyFinding",
        "DES_ResearchFocus",
        "AN_RQ1_Relevance",
        "AN_RQ2_Relevance",
        "SLR_Inclusion",
    ]
    missing = []
    for field in required:
        value = coding.get(field)
        if value in (None, "", []):
            missing.append(field)
    return missing


def build_coding_json_schema() -> dict[str, Any]:
    """Return a strict JSON schema for structured output-capable providers."""
    properties: dict[str, Any] = {}
    required: list[str] = []
    for key, default in CODING_DEFAULTS.items():
        required.append(key)
        if isinstance(default, list):
            properties[key] = {"type": "array", "items": {"type": "string"}}
        elif isinstance(default, int):
            prop: dict[str, Any] = {"type": "integer"}
            if key in BOUNDED_INT_FIELDS:
                lower, upper = BOUNDED_INT_FIELDS[key]
                prop["minimum"] = lower
                prop["maximum"] = upper
            properties[key] = prop
        else:
            if key == "SLR_Inclusion":
                properties[key] = {"type": "string", "enum": ["włączyć", "wykluczyć", "wątpliwe"]}
            else:
                properties[key] = {"type": "string"}

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }


enable_function_tracing(globals(), __name__, include_private=True)

