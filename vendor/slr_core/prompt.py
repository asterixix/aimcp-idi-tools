"""Prompt templates and few-shot examples used by the LLM client."""
from __future__ import annotations

from .schema import CODING_DEFAULTS
from .utils import enable_function_tracing

# faithfulness i evidence-first porównują słowa tych pól z tekstem źródła, więc tłumaczenie zeruje wynik.
LANGUAGE_RULE = (
    "LANGUAGE RULE: write PUB_KeyFinding, SLR_InclusionReason and RES_KeyStat in the language of the source document "
    "(English source: in English), reusing its terminology; never translate them into Polish. "
    "RES_KeyStat without a reported statistic is an empty string. "
)


def coding_json_schema_hint() -> str:
    keys = ", ".join(sorted(CODING_DEFAULTS.keys()))
    return (
        "Zwróć WYŁĄCZNIE pojedynczy obiekt JSON, bez markdown i bez komentarzy. "
        "JSON musi zawierać pola: "
        f"{keys}. "
        "Dla pól liczbowych używaj liczb całkowitych. Dla pól listowych zwracaj listę. "
        "SLR_Inclusion dozwolone tylko: włączyć | wykluczyć | wątpliwe. "
        f"{LANGUAGE_RULE}"
        "Zakresy: AN_RQ1_Relevance i AN_RQ2_Relevance = 0-3; AN_QFSR/AN_SC/AN_VF/AN_RI/AN_HR/AN_SSS/AN_CER = 0-2."
    )


def few_shot_chunk_examples() -> str:
    return (
        "EXAMPLE 1:\n"
        "CHUNK: 'W badaniu autorzy analizują jakość metadanych w repozytoriach uczelnianych, identyfikując braki w polach DOI i autorów.'\n"
        "OUTPUT_JSON:\n"
        "{\n  \"chunk_focus\": \"Analiza braków metadanych (DOI, autor)\",\n  \"section_type\": \"wyniki\",\n  \"evidence_quotes\": [\"missing DOI field in record\"],\n  \"confidence\": 2\n}\n"
        "\n"
        "EXAMPLE 2:\n"
        "CHUNK: 'Autorka omawia mechanizmy pośrednictwa protokołu MCP oraz api-tool-calling.'\n"
        "OUTPUT_JSON:\n"
        "{\n  \"chunk_focus\": \"Opis mechanizmu MCP i tool-calling\",\n  \"section_type\": \"wprowadzenie\",\n  \"evidence_quotes\": [\"MCP structures the semantic transaction\"],\n  \"confidence\": 3\n}\n"
    )


def few_shot_coding_examples() -> str:
    return (
        "EXAMPLE 1:\n"
        "INPUT: 'Empiryczne badanie jakości metadanych w polskich repozytoriach, analiza pól DOI i autorów.'\n"
        "OUTPUT_JSON:\n"
        "{\n  \"PUB_Title\": \"Quality of metadata in Polish repositories\",\n  \"PUB_Type\": \"empiryczna\",\n  \"DES_ResearchFocus\": \"IQ_metadanych\",\n  \"AN_RQ1_Relevance\": 3,\n  \"AN_VF\": 1,\n  \"SLR_Inclusion\": \"włączyć\",\n  \"PUB_KeyFinding\": \"Braki w polach DOI i autorów wpływają na odnajdywalność rekordów.\"\n}\n"
        "\n"
        "EXAMPLE 2:\n"
        "INPUT: 'Artykuł koncepcyjny opisujący MCP jako protokół narzędziowy.'\n"
        "OUTPUT_JSON:\n"
        "{\n  \"PUB_Title\": \"Model Context Protocol as an information intermediary\",\n  \"PUB_Type\": \"teoretyczna\",\n  \"DES_ResearchFocus\": \"pośrednictwo_informacyjne\",\n  \"AN_RQ1_Relevance\": 3,\n  \"AN_RQ2_Relevance\": 2,\n  \"SLR_Inclusion\": \"włączyć\",\n  \"PUB_KeyFinding\": \"MCP structures semantic transactions between users, models and repositories.\"\n}\n"
    )


def build_chunk_prompt(chunk_text: str, index: int, total: int, doc_filename: str) -> str:
    header = (
        "You are a careful reviewer. Extract short JSON about this chunk.\n"
        "Return only a JSON object with keys: chunk_focus, section_type, evidence_quotes (list), confidence (0-3).\n"
    )
    examples = few_shot_chunk_examples()
    body = f"CHUNK_INDEX: {index}/{total}\nFILENAME: {doc_filename}\nCHUNK_TEXT:\n{chunk_text}\n"
    return header + "\n" + examples + "\n" + body


def build_synthesis_prompt(chunk_evidences: list[dict], doc_title: str) -> str:
    header = (
        "You are an expert SLR reviewer. Given chunk-level JSON evidences, synthesize a single JSON coding for the whole document.\n"
        "Return only a single JSON object with fields: PUB_ID, PUB_Title, PUB_KeyFinding, DES_ResearchFocus, AN_RQ1_Relevance, AN_RQ2_Relevance, SLR_Inclusion, notes.\n"
        f"{LANGUAGE_RULE}\n"
    )
    examples = few_shot_coding_examples()
    body = "CHUNK_EVIDENCES:\n"
    for i, e in enumerate(chunk_evidences, 1):
        body += f"EVIDENCE_{i}: {json_safe(e)}\n"
    body += f"\nDOCUMENT TITLE: {doc_title}\n"
    return header + "\n" + examples + "\n" + body


def build_single_shot_prompt(
    *,
    title: str,
    excerpt: str,
    coding_scheme_excerpt: str,
    doc_filename: str,
    coder_role: str = "coder_a",
) -> str:
    return (
        "Jesteś badaczem SLR i kodujesz publikację zgodnie ze schematem.\n"
        "Najpierw wybierz cytaty-dowody, potem mapuj je do kodów. Nie zgaduj.\n"
        f"{coding_json_schema_hint()}\n\n"
        f"<role>{coder_role}</role>\n"
        "<rules>\n"
        "- Jeśli brak pewności: wybierz wartości konserwatywne, a w notes opisz niepewność.\n"
        "- PUB_KeyFinding: maks 2 zdania.\n"
        "- evidence_quotes: 4-8 krótkich cytatów z dokumentu (dokładne fragmenty).\n"
        "- AN_* i skale: tylko wartości dozwolone przez schemat.\n"
        "- Dla pól krytycznych (PUB_KeyFinding, AN_*, SLR_InclusionReason) użyj cytatów źródłowych.\n"
        "</rules>\n\n"
        "<coding_scheme_excerpt>\n"
        f"{coding_scheme_excerpt}\n"
        "</coding_scheme_excerpt>\n\n"
        "<few_shot>\n"
        f"{few_shot_coding_examples()}\n"
        "</few_shot>\n\n"
        "<document_meta>\n"
        f"filename: {doc_filename}\n"
        f"title: {title}\n"
        "</document_meta>\n\n"
        "<document_text>\n"
        f"{excerpt}\n"
        "</document_text>"
    )


def build_repair_prompt(
    *,
    previous_json: str,
    missing_fields: list[str],
    title: str,
    excerpt: str,
) -> str:
    missing = ", ".join(missing_fields)
    return (
        "Popraw poprzedni wynik tak, aby był kompletnym i poprawnym JSON.\n"
        f"Brakujące lub błędne pola: {missing}\n"
        f"{coding_json_schema_hint()}\n\n"
        "<previous_output>\n"
        f"{previous_json}\n"
        "</previous_output>\n\n"
        "<document_title>\n"
        f"{title}\n"
        "</document_title>\n\n"
        "<document_excerpt>\n"
        f"{excerpt}\n"
        "</document_excerpt>"
    )


def build_evidence_repair_prompt(*, previous_json: str, issues: list[str], excerpt: str) -> str:
    return (
        "Popraw wynik JSON tak, aby był evidence-first i audytowalny.\n"
        f"Wykryte problemy: {'; '.join(issues)}\n"
        f"{coding_json_schema_hint()}\n"
        "Wymagane: minimum 2 dokładne cytaty wspierające pola krytyczne.\n\n"
        "<previous_output>\n"
        f"{previous_json}\n"
        "</previous_output>\n\n"
        "<document_excerpt>\n"
        f"{excerpt}\n"
        "</document_excerpt>"
    )


def build_adjudication_prompt(
    *,
    title: str,
    excerpt: str,
    coder_a_json: str,
    coder_b_json: str,
) -> str:
    return (
        "Jesteś adjudykatorem. Masz 2 niezależne kodowania tego samego dokumentu.\n"
        "Wybierz i uzgodnij finalne kodowanie zgodne ze schematem. Nie zgaduj.\n"
        "Zachowaj konserwatywność i wymagaj cytatów.\n"
        f"{coding_json_schema_hint()}\n\n"
        "<coder_a>\n"
        f"{coder_a_json}\n"
        "</coder_a>\n\n"
        "<coder_b>\n"
        f"{coder_b_json}\n"
        "</coder_b>\n\n"
        "<document_title>\n"
        f"{title}\n"
        "</document_title>\n\n"
        "<document_excerpt>\n"
        f"{excerpt}\n"
        "</document_excerpt>"
    )


def json_safe(obj) -> str:
    try:
        import json

        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)


enable_function_tracing(globals(), __name__, include_private=True)
