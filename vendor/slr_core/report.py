"""Report generation for SLR coding outputs."""
from __future__ import annotations

import json
import os
import re
import socket
from collections import Counter
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse

from .config import CONFIG, logger
from .schema import normalize_inclusion_label
from .utils import _atomic_write_text, enable_function_tracing
from .web_research import build_web_research

_REPORT_LLM_HOST_OK: Optional[bool] = None


def _to_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _top(counter: Counter, limit: int = 10) -> list[tuple[str, int]]:
    return counter.most_common(limit)


def _label(value, default: str = "nieznany") -> str:
    if value is None:
        return default
    if isinstance(value, list):
        items = [str(x).strip() for x in value if str(x).strip()]
        return ", ".join(items) if items else default
    if isinstance(value, dict):
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except Exception:
            return default
    text = str(value).strip()
    return text if text else default


def _as_int(value, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            return int(text)
    return default


def _as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _source_bucket(path_value) -> str:
    path = str(path_value or "").replace("\\", "/")
    if "/Literature_Ready/" in path or "/literatura/md/" in path:
        return "Literature_Ready"
    if "/Literature/" in path or "/literatura/pdf/" in path:
        return "Literature"
    if not path:
        return "brak_źródła"
    parts = [p for p in path.split("/") if p]
    if len(parts) >= 2:
        return parts[-2]
    return "inne"


def _extract_geo_hints(codings: list[dict], limit: int = 10) -> list[tuple[str, int]]:
    markers = [
        "poland",
        "polish",
        "nigeria",
        "europe",
        "global",
        "usa",
        "uk",
        "italy",
        "italian",
        "pakistan",
    ]
    counts: Counter = Counter()
    for c in codings:
        blob = " ".join(
            [
                str(c.get("PUB_Sample", "") or ""),
                str(c.get("DES_DataContext", "") or ""),
                str(c.get("DES_LanguageContext", "") or ""),
                str(c.get("notes", "") or ""),
            ]
        ).lower()
        for marker in markers:
            if marker in blob:
                counts[marker] += 1
    return counts.most_common(limit)


def _build_project_context(codings: list[dict]) -> dict:
    total = len(codings)
    by_inclusion = Counter(normalize_inclusion_label(c.get("SLR_Inclusion")) for c in codings)
    by_focus = Counter(_label(c.get("DES_ResearchFocus"), "inna") for c in codings)
    by_pub_type = Counter(_label(c.get("PUB_Type"), "nieznany") for c in codings)
    by_domain = Counter(_label(c.get("PUB_Domain"), "inna") for c in codings)
    by_method = Counter(_label(c.get("PUB_Method"), "nieokreślona") for c in codings)
    by_gap = Counter(_label(c.get("SYN_GapType"), "brak_luki") for c in codings)
    by_source = Counter(_source_bucket(c.get("_filepath")) for c in codings)
    by_year = Counter(_as_int(c.get("PUB_Year"), 0) for c in codings if _as_int(c.get("PUB_Year"), 0) > 0)
    rq1_avg = sum(_as_int(c.get("AN_RQ1_Relevance", 0), 0) for c in codings) / total if total else 0.0
    rq2_avg = sum(_as_int(c.get("AN_RQ2_Relevance", 0), 0) for c in codings) / total if total else 0.0
    avg_faithfulness = (
        sum(_as_float(c.get("_faithfulness_score", 1.0), 0.0) for c in codings) / total if total else 0.0
    )
    gap_desc = Counter(_label(c.get("SYN_GapDescription"), "") for c in codings)
    geo_hints = _extract_geo_hints(codings)
    web_research = build_web_research(codings)
    return {
        "total": total,
        "by_inclusion": by_inclusion,
        "by_focus": by_focus,
        "by_pub_type": by_pub_type,
        "by_domain": by_domain,
        "by_method": by_method,
        "by_gap": by_gap,
        "by_source": by_source,
        "by_year": by_year,
        "rq1_avg": rq1_avg,
        "rq2_avg": rq2_avg,
        "avg_faithfulness": avg_faithfulness,
        "gap_desc": gap_desc,
        "geo_hints": geo_hints,
        "web_research": web_research,
    }


def _generate_llm_synthesis(context: dict) -> Optional[str]:
    global _REPORT_LLM_HOST_OK
    if not CONFIG.get("report_llm_enabled", True):
        return None
    if not str(CONFIG.get("api_key", "")).strip():
        return None
    if _REPORT_LLM_HOST_OK is None:
        try:
            host = urlparse(str(CONFIG.get("api_base", "") or "")).hostname or ""
            if not host:
                _REPORT_LLM_HOST_OK = False
            else:
                socket.getaddrinfo(host, 443)
                _REPORT_LLM_HOST_OK = True
        except Exception:
            _REPORT_LLM_HOST_OK = False
            logger.warning("Report LLM synthesis skipped (API host unresolved)")
    if not _REPORT_LLM_HOST_OK:
        return None
    try:
        from .llm import LLMClient
    except Exception:
        return None
    try:
        client = LLMClient(CONFIG)
        model = CONFIG.get("report_llm_model") or CONFIG.get("model")
        compact_context = {
            "total": context.get("total"),
            "by_inclusion": context.get("by_inclusion", {}),
            "by_focus_top": context.get("by_focus", {}).most_common(10),
            "by_pub_type_top": context.get("by_pub_type", {}).most_common(10),
            "by_domain_top": context.get("by_domain", {}).most_common(10),
            "by_method_top": context.get("by_method", {}).most_common(10),
            "by_gap_top": context.get("by_gap", {}).most_common(10),
            "by_source_top": context.get("by_source", {}).most_common(10),
            "by_year_top": context.get("by_year", {}).most_common(10),
            "rq1_avg": round(float(context.get("rq1_avg", 0.0)), 3),
            "rq2_avg": round(float(context.get("rq2_avg", 0.0)), 3),
            "avg_faithfulness": round(float(context.get("avg_faithfulness", 0.0)), 3),
            "geo_hints_top": context.get("geo_hints", [])[:10],
            "gap_desc_top": context.get("gap_desc", {}).most_common(8),
            "web_queries": (context.get("web_research") or {}).get("queries", []),
            "web_openalex_top": (context.get("web_research") or {}).get("openalex", [])[:8],
            "web_crossref_verified": (context.get("web_research") or {}).get("crossref_verified", [])[:8],
        }
        prompt = (
            "Przygotuj zwięzłą syntezę badawczą po polsku na podstawie danych SLR.\n"
            "Zwróć WYŁĄCZNIE pojedynczy obiekt JSON (bez markdown i bez komentarzy) o strukturze:\n"
            "{\n"
            "  \"stan_badan\": [\"...\"],\n"
            "  \"glowne_nurty\": [\"...\"],\n"
            "  \"luki_badawcze\": [\"...\"],\n"
            "  \"synteza_i_uzasadnienie\": [\"...\"]\n"
            "}\n"
            "W każdej liście podaj 3-6 krótkich punktów. Nie wymyślaj faktów spoza danych.\n\n"
            f"DANE:\n{json.dumps(compact_context, ensure_ascii=False)}"
        )
        response = client._request_text(
            prompt,
            model_override=model,
            structured=False,
            expect_json=False,
            prefer_stream=False,
            max_tokens_override=int(CONFIG.get("report_llm_max_tokens", 1800)),
            timeout_read_override=int(CONFIG.get("report_llm_timeout_seconds", 120)),
        )
        if not response:
            return None
        text = str(response).strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        match = re.search(r"(\{.*\})", text, re.S)
        if not match:
            return None
        parsed = json.loads(match.group(1))
        if not isinstance(parsed, dict):
            return None
        sections = [
            ("Stan badań", parsed.get("stan_badan", [])),
            ("Główne nurty", parsed.get("glowne_nurty", [])),
            ("Luki badawcze", parsed.get("luki_badawcze", [])),
            ("Synteza i uzasadnienie własnego projektu", parsed.get("synteza_i_uzasadnienie", [])),
        ]
        lines: list[str] = []
        for title, items in sections:
            lines.append(f"### {title}")
            if isinstance(items, list):
                cleaned = [str(x).strip() for x in items if str(x).strip()]
            else:
                cleaned = []
            if not cleaned:
                lines.append("- Brak danych do syntezy.")
            else:
                for item in cleaned[:6]:
                    lines.append(f"- {item}")
            lines.append("")
        return "\n".join(lines).strip() or None
    except Exception:
        logger.exception("LLM synthesis for report failed")
        return None


def generate_report(codings: list[dict], report_path: str, bib_context: Optional[str] = None) -> str:
    """Generate markdown report from coding records."""
    context = _build_project_context(codings)
    total = len(codings)
    by_inclusion = Counter(normalize_inclusion_label(c.get("SLR_Inclusion")) for c in codings)
    by_focus = Counter(_label(c.get("DES_ResearchFocus"), "inna") for c in codings)
    by_cluster = Counter()
    by_pub_type = Counter(_label(c.get("PUB_Type"), "nieznany") for c in codings)
    by_gap = Counter(_label(c.get("SYN_GapType"), "brak_luki") for c in codings)
    by_domain = Counter(_label(c.get("PUB_Domain"), "inna") for c in codings)

    for coding in codings:
        for cluster in _to_list(coding.get("SLR_Cluster")):
            by_cluster[cluster] += 1

    rq1_avg = sum(_as_int(c.get("AN_RQ1_Relevance", 0), 0) for c in codings) / total if total else 0.0
    rq2_avg = sum(_as_int(c.get("AN_RQ2_Relevance", 0), 0) for c in codings) / total if total else 0.0
    dual_disagreements = sum(1 for c in codings if c.get("_dual_disagreement"))
    avg_faithfulness = (
        sum(float(c.get("_faithfulness_score", 1.0) or 0.0) for c in codings) / total if total else 0.0
    )
    recoding_flags = sum(1 for c in codings if c.get("_needs_recoding"))
    numeric_flags = sum(1 for c in codings if c.get("_needs_manual_numeric_review"))

    report_lines = [
        "# Raport kodowania SLR",
        "",
        f"_Wygenerowano: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_",
        "",
        "## 1. Podsumowanie zbioru",
        f"- Liczba zakodowanych publikacji: **{total}**",
        f"- Włączyć: **{by_inclusion.get('włączyć', 0)}**",
        f"- Wykluczyć: **{by_inclusion.get('wykluczyć', 0)}**",
        f"- Wątpliwe: **{by_inclusion.get('wątpliwe', 0)}**",
        f"- Średnia AN_RQ1_Relevance: **{rq1_avg:.2f}**",
        f"- Średnia AN_RQ2_Relevance: **{rq2_avg:.2f}**",
        "",
        "## 2. Rozkład typów publikacji",
    ]

    for name, count in _top(by_pub_type, 10):
        report_lines.append(f"- {name}: {count}")

    report_lines.extend(["", "## 3. Główne osie badawcze"])
    for name, count in _top(by_focus, 15):
        report_lines.append(f"- {name}: {count}")

    report_lines.extend(["", "## 4. Klastry SLR"])
    for name, count in _top(by_cluster, 10):
        report_lines.append(f"- {name}: {count}")

    report_lines.extend(["", "## 5. Domeny publikacji"])
    for name, count in _top(by_domain, 10):
        report_lines.append(f"- {name}: {count}")

    report_lines.extend(["", "## 6. Najczęściej sygnalizowane luki"])
    for name, count in _top(by_gap, 10):
        report_lines.append(f"- {name}: {count}")

    report_lines.extend(["", "## 7. Publikacje kluczowe (priorytet cytowania = 3)"])
    strategic = [c for c in codings if _as_int(c.get("STR_CitationPriority", 0), 0) == 3]
    for coding in strategic[:30]:
        report_lines.append(
            f"- {coding.get('PUB_ID', 'NOID')}: {coding.get('PUB_Title', '(brak tytułu)')} "
            f"({coding.get('PUB_Year', 'brak roku')})"
        )
    if not strategic:
        report_lines.append("- Brak rekordów o priorytecie 3.")

    report_lines.extend(["", "## 8. Notatki metodologiczne"])
    report_lines.append(
        "- Raport jest syntetycznym podsumowaniem automatycznego kodowania. "
        "Wyniki wymagają audytu badacza przed publikacją."
    )
    if bib_context:
        report_lines.append("- Kontekst bibliograficzny był dostępny podczas generowania raportu.")
    else:
        report_lines.append("- Kontekst bibliograficzny nie był dostępny.")

    report_lines.extend(
        [
            "",
            "## 9. AI-assisted extraction protocol + quality controls (PRISMA 2020 aligned)",
            "- Pipeline używa dual independent coding (2 niezależne przebiegi modeli) oraz adjudication mode przy rozbieżnościach.",
            "- Struktura danych wymuszana przez schema-first JSON; dodatkowo stosowana jest lokalna walidacja semantyczna.",
            "- Evidence-first: pola krytyczne muszą mieć cytaty źródłowe (provenance).",
            f"- Średni faithfulness score: **{avg_faithfulness:.3f}** (zakres 0-1).",
            f"- Dokumenty z rozbieżnością dual-coding: **{dual_disagreements}**.",
            f"- Dokumenty oznaczone do rekodowania: **{recoding_flags}**.",
            f"- Dokumenty high-risk (numeric review): **{numeric_flags}**.",
            "- Wyniki wymagają ostatecznej walidacji badacza przed syntezą i publikacją.",
            "",
            "## 10. Governance / TEVV monitoring",
            "- Testy jakości wykonywane automatycznie po każdym przebiegu (audit, reliability, queue).",
            "- Raporty kontroli: `coding_quality_audit.*`, `recoding_queue.*`, `reliability_report.*`.",
            "- Ograniczenia uogólniania i ryzyka transparentności powinny być dokumentowane razem z wynikami."
        ]
    )

    report_lines.extend(["", "## 11. Temat pracy / zagadnienie badawcze / pomysł projektu"])
    top_focus = ", ".join(f for f, _ in context["by_focus"].most_common(3)) or "brak dominującego nurtu"
    top_domain = ", ".join(f for f, _ in context["by_domain"].most_common(3)) or "brak"
    report_lines.append(
        f"- Rdzeń tematyczny korpusu: **{top_focus}**; dominujące domeny: **{top_domain}**."
    )
    report_lines.append(
        "- Proponowany kierunek projektu: analiza jakości pośrednictwa informacyjnego LLM/MCP "
        "z naciskiem na wierność semantyczną, zniekształcenia epistemiczne i kontrolę jakości danych."
    )

    report_lines.extend(["", "## 12. Ile artykułów wstępnie wybrano i skąd pochodziły"])
    report_lines.append(f"- Wstępnie wybrane artykuły (w aktualnym zbiorze kodowań): **{context['total']}**.")
    report_lines.append(f"- Źródła plików: {', '.join(f'{k}: {v}' for k, v in context['by_source'].most_common(5)) or 'brak'}")
    report_lines.append(
        f"- Typy publikacji: {', '.join(f'{k}: {v}' for k, v in context['by_pub_type'].most_common(8)) or 'brak'}."
    )
    report_lines.append(
        f"- Decyzje włączenia: włączyć={context['by_inclusion'].get('włączyć', 0)}, "
        f"wykluczyć={context['by_inclusion'].get('wykluczyć', 0)}, "
        f"wątpliwe={context['by_inclusion'].get('wątpliwe', 0)}."
    )
    report_lines.append(
        "- Brane pod uwagę: relevance dla RQ1/RQ2, typ luki, jakość dowodów i spójność semantyczna; "
        "niebranie pod uwagę: rekordy bez użytecznej treści i wyniki nieprzechodzące minimalnych kontroli jakości."
    )

    report_lines.extend(["", "## 13. Stan badań w temacie"])
    if context["by_year"]:
        year_min = min(context["by_year"].keys())
        year_max = max(context["by_year"].keys())
        report_lines.append(f"- Zakres czasowy publikacji: **{year_min}-{year_max}**.")
    report_lines.append(
        f"- Średnia relewancja: AN_RQ1={context['rq1_avg']:.2f}, AN_RQ2={context['rq2_avg']:.2f}."
    )
    report_lines.append(
        f"- Średni faithfulness zakodowań: **{context['avg_faithfulness']:.3f}**."
    )
    report_lines.append(
        f"- Dominujące obszary badań: {', '.join(f'{k} ({v})' for k, v in context['by_focus'].most_common(5)) or 'brak'}."
    )

    report_lines.extend(["", "## 14. Główne nurty (kąty badania)"])
    for name, count in context["by_focus"].most_common(10):
        report_lines.append(f"- {name}: {count}")
    if not context["by_focus"]:
        report_lines.append("- Brak danych o nurtach.")

    report_lines.extend(["", "## 15. Metody – jak badano"])
    for name, count in context["by_method"].most_common(10):
        report_lines.append(f"- {name}: {count}")
    if not context["by_method"]:
        report_lines.append("- Brak danych o metodach.")

    report_lines.extend(["", "## 16. Populacja / kontekst / demografia"])
    report_lines.append(
        f"- Konteksty domenowe: {', '.join(f'{k} ({v})' for k, v in context['by_domain'].most_common(8)) or 'brak'}."
    )
    geo = context["geo_hints"]
    if geo:
        report_lines.append(
            f"- Sygnały geograficzne/demograficzne w próbie: {', '.join(f'{k} ({v})' for k, v in geo)}."
        )
    else:
        report_lines.append("- Brak jednoznacznych sygnałów geograficznych/demograficznych w metadanych.")

    report_lines.extend(["", "## 17. Luki badawcze"])
    for name, count in context["by_gap"].most_common(8):
        report_lines.append(f"- Typ luki: {name} ({count})")
    for desc, count in context["gap_desc"].most_common(6):
        if not desc or desc == "brak":
            continue
        report_lines.append(f"- {desc} ({count})")
    report_lines.append(
        "- Priorytet nowych pytań: gdzie LLM/MCP traci wierność semantyczną, "
        "jakie KOS i domeny generują największe zniekształcenia oraz jakie warunki ograniczają halucynacje."
    )

    report_lines.extend(["", "## 18. Plan poprawy procesu kodowania (na podstawie outputów i logów)"])
    needs_recode = sum(1 for c in codings if c.get("_needs_recoding"))
    low_faith = sum(
        1
        for c in codings
        if _as_float(c.get("_faithfulness_score", 1.0), 1.0) < float(CONFIG.get("faithfulness_threshold", 0.7))
    )
    low_quotes = sum(
        1
        for c in codings
        if len(_to_list(c.get("evidence_quotes"))) < int(CONFIG.get("min_evidence_quotes_critical", 2))
    )
    dual_dis = sum(1 for c in codings if c.get("_dual_disagreement"))
    fallback_dis = sum(1 for c in codings if c.get("_adjudication_fallback"))
    report_lines.append(
        f"- Priorytet 1 (rekodowanie): `_needs_recoding`={needs_recode}, "
        f"low_faithfulness={low_faith}, low_evidence_quotes={low_quotes}."
    )
    report_lines.append(
        f"- Priorytet 2 (stabilność dual coding): dual_disagreement={dual_dis}, adjudication_fallback={fallback_dis}. "
        "Dla fallback uruchamiać recoding celowany."
    )
    report_lines.append(
        "- Priorytet 3 (spójność etykiet): utrzymywać normalizację `SLR_Inclusion` i typów publikacji, "
        "aby poprawić porównywalność wyników i raportowanie."
    )
    report_lines.append(
        "- Priorytet 4 (odporność sieciowa): przy problemach DNS przechodzić automatycznie w tryb offline "
        "dla web-research/LLM-synthesis raportu."
    )

    report_lines.extend(["", "## 19. Synteza i uzasadnienie własnego projektu"])
    report_lines.append(
        "- Temat warto podjąć, ponieważ bieżące badania są heterogeniczne metodologicznie i często mają ograniczoną walidację "
        "wierności semantycznej na poziomie metadanych."
    )
    report_lines.append(
        "- Projekt wnosi wartość przez połączenie: dual coding, adjudication, evidence-first, faithfulness i quality gates, "
        "co zwiększa audytowalność i replikowalność."
    )
    report_lines.append(
        "- Wyniki mogą dostarczyć modelu konceptualnego jakości pośrednictwa informacyjnego LLM-MCP-repozytorium "
        "oraz wskazań dla praktyki bibliotek i repozytoriów."
    )

    web = context.get("web_research") or {}
    openalex = web.get("openalex") or []
    verified = web.get("crossref_verified") or []
    queries = web.get("queries") or []
    report_lines.extend(["", "## 20. Internetowy przegląd stanu badań (OpenAlex + Crossref)"])
    if queries:
        report_lines.append(f"- Zapytania badawcze: {', '.join(queries)}.")
    if openalex:
        report_lines.append("- Najczęściej cytowane prace (OpenAlex, od 2020):")
        for item in openalex[:10]:
            title = _label(item.get("title"), "(bez tytułu)")
            year = item.get("year") or "brak roku"
            cited = int(item.get("cited_by_count") or 0)
            url = _label(item.get("url"), "")
            if url:
                report_lines.append(f"- [{title}]({url}) ({year}), citations={cited}")
            else:
                report_lines.append(f"- {title} ({year}), citations={cited}")
    else:
        report_lines.append("- Brak danych z OpenAlex (sprawdź łączność internetową/API).")

    if verified:
        report_lines.append("- Zweryfikowane DOI z korpusu (Crossref):")
        for item in verified[:12]:
            title = _label(item.get("title"), "(bez tytułu)")
            year = item.get("year") or "brak roku"
            doi_url = _label(item.get("doi"), "")
            if doi_url:
                report_lines.append(f"- [{title}]({doi_url}) ({year})")
            else:
                report_lines.append(f"- {title} ({year})")
    else:
        report_lines.append("- Brak zweryfikowanych DOI w Crossref lub brak DOI w rekordach.")

    llm_md = _generate_llm_synthesis(context)
    if llm_md:
        report_lines.extend(["", "## 21. Synteza ekspercka LLM (wspomaganie raportu)", llm_md])

    _atomic_write_text(report_path, "\n".join(report_lines))
    logger.info("Report saved: %s", report_path)
    return report_path


def generate_report_from_existing() -> Optional[str]:
    output_dir = CONFIG.get("output_dir")
    codings_path = os.path.join(output_dir, "codings.json")
    if not os.path.exists(codings_path):
        logger.warning("Brak pliku %s", codings_path)
        return None

    try:
        with open(codings_path, "r", encoding="utf-8") as handle:
            codings = json.load(handle)
        if not isinstance(codings, list):
            logger.warning("Niepoprawny format codings.json (oczekiwano listy).")
            return None
    except Exception as exc:
        logger.exception("Nie udało się wczytać codings.json: %s", exc)
        return None

    bib_context = None
    try:
        with open(CONFIG.get("bibliografia_path"), "r", encoding="utf-8") as handle:
            bib_context = handle.read()
    except Exception:
        bib_context = None

    report_path = os.path.join(output_dir, "SLR_CODING_REPORT.md")
    return generate_report(codings, report_path, bib_context)


enable_function_tracing(globals(), __name__, include_private=True)

