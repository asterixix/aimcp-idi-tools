"""Lightweight web research helpers for report enrichment."""
from __future__ import annotations

import re
import socket
from collections import Counter
from typing import Optional
from urllib.parse import quote, unquote, urlparse

import requests

from .config import CONFIG, logger
from .utils import enable_function_tracing

_HOST_STATUS: dict[str, bool] = {}


def _host_from_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme and parsed.netloc:
        return parsed.netloc.split("@")[-1].split(":")[0].lower()
    return str(value or "").split(":")[0].strip().lower()


def _is_host_reachable(value: str) -> bool:
    host = _host_from_url(value)
    if not host:
        return False
    cached = _HOST_STATUS.get(host)
    if cached is not None:
        return cached
    try:
        socket.getaddrinfo(host, 443)
        _HOST_STATUS[host] = True
        return True
    except Exception:
        _HOST_STATUS[host] = False
        logger.warning("Web research host unresolved, skipping requests: %s", host)
        return False


def _request_json(url: str, *, params: Optional[dict] = None, timeout: Optional[int] = None) -> Optional[dict]:
    if not _is_host_reachable(url):
        return None
    try:
        response = requests.get(
            url,
            params=params or {},
            timeout=timeout or int(CONFIG.get("report_web_timeout_seconds", 20)),
        )
        if response.status_code != 200:
            if "api.crossref.org" in url and response.status_code == 404:
                logger.info("Crossref: record not found for query: %s", url)
            else:
                logger.warning("Web research request failed (%s): %s", response.status_code, url)
            return None
        return response.json()
    except Exception as exc:
        msg = str(exc)
        if "NameResolutionError" in msg or "getaddrinfo failed" in msg or "Failed to resolve" in msg:
            _HOST_STATUS[_host_from_url(url)] = False
            logger.warning("Web research DNS error for %s; host temporarily disabled", url)
            return None
        logger.warning("Web research request error for %s: %s", url, exc)
        return None


def _looks_like_doi(value: str) -> bool:
    return bool(re.match(r"^10\.\d{4,9}/\S+$", value, flags=re.I))


def _normalize_doi(doi: str) -> str:
    text = str(doi or "").strip()
    if not text:
        return ""
    text = unquote(text)
    text = text.split("#", 1)[0].split("?", 1)[0].strip()
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text, flags=re.I)
    text = re.sub(r"^doi:\s*", "", text, flags=re.I).strip()
    if text.lower().startswith("arxiv:"):
        return ""
    if text.lower().startswith("http://") or text.lower().startswith("https://"):
        return ""
    # Common language suffix accidentally appended to DOI URL paths.
    text = re.sub(r"/(en|pl|de|fr|es|it)$", "", text, flags=re.I).strip()
    if not _looks_like_doi(text):
        return ""
    return text


def build_queries(codings: list[dict]) -> list[str]:
    terms: list[str] = []
    for coding in codings:
        focus = str(coding.get("DES_ResearchFocus", "") or "").strip()
        if focus and focus not in {"inna", "nieokreślona"}:
            terms.append(focus.replace("_", " "))
        title = str(coding.get("PUB_Title", "") or "").strip()
        if title:
            terms.append(title)

    # Prefer frequent and short/high-signal phrases.
    counts = Counter(terms)
    ranked = [term for term, _ in counts.most_common(20)]
    max_queries = int(CONFIG.get("report_web_max_queries", 4))
    return ranked[:max_queries]


def _is_relevant_openalex_title(title: str) -> bool:
    text = str(title or "").lower()
    if not text:
        return False
    strong_markers = [
        "library",
        "libraries",
        "metadata",
        "bibliograph",
        "catalog",
        "catalogu",
        "repository",
        "repositories",
        "information science",
        "knowledge organization",
        "mcp",
    ]
    support_markers = [
        "llm",
        "language model",
        "generative ai",
        "hallucination",
        "information retrieval",
        "semantic",
        "intermediation",
    ]
    strong_hits = sum(1 for marker in strong_markers if marker in text)
    support_hits = sum(1 for marker in support_markers if marker in text)
    if strong_hits >= 1:
        return True
    # Allow some narrower LLM papers if they are clearly in IR/semantic context.
    return support_hits >= 2


def fetch_openalex_overview(queries: list[str]) -> list[dict]:
    per_query = int(CONFIG.get("report_web_results_per_query", 5))
    email = str(CONFIG.get("report_web_email", "") or "").strip()
    unique: dict[str, dict] = {}

    for query in queries:
        params = {
            "search": query,
            "per-page": per_query,
            "sort": "cited_by_count:desc",
            "filter": "from_publication_date:2020-01-01",
        }
        if email:
            params["mailto"] = email
        payload = _request_json("https://api.openalex.org/works", params=params)
        if not payload:
            continue
        for work in payload.get("results", []):
            work_id = str(work.get("id") or "").strip()
            if not work_id:
                continue
            title = str(work.get("display_name") or "").strip()
            if not _is_relevant_openalex_title(title):
                continue
            year = work.get("publication_year")
            cited = int(work.get("cited_by_count") or 0)
            doi = str(work.get("doi") or "").strip()
            landing = str(work.get("primary_location", {}).get("landing_page_url") or "").strip()
            source = str(work.get("primary_location", {}).get("source", {}).get("display_name") or "").strip()
            unique[work_id] = {
                "id": work_id,
                "title": title,
                "year": year,
                "cited_by_count": cited,
                "doi": doi,
                "source": source,
                "url": landing or doi or work_id,
                "query": query,
            }

    results = sorted(unique.values(), key=lambda x: x.get("cited_by_count", 0), reverse=True)
    return results[: max(10, int(CONFIG.get("report_web_results_per_query", 5)) * 2)]


def verify_dois_with_crossref(codings: list[dict]) -> list[dict]:
    verified = []
    seen = set()
    for coding in codings:
        doi = _normalize_doi(str(coding.get("PUB_DOI", "") or ""))
        if not doi or doi in seen:
            continue
        seen.add(doi)
        payload = _request_json(f"https://api.crossref.org/works/{quote(doi, safe='')}")
        if not payload:
            continue
        msg = payload.get("message") or {}
        title = ""
        title_list = msg.get("title") or []
        if isinstance(title_list, list) and title_list:
            title = str(title_list[0]).strip()
        year = None
        issued = msg.get("issued", {}).get("date-parts") or []
        if issued and isinstance(issued, list) and issued[0]:
            try:
                year = int(issued[0][0])
            except Exception:
                year = None
        verified.append(
            {
                "doi": f"https://doi.org/{doi}",
                "title": title,
                "year": year,
                "publisher": str(msg.get("publisher") or "").strip(),
                "type": str(msg.get("type") or "").strip(),
            }
        )
    return verified


def build_web_research(codings: list[dict]) -> dict:
    if not CONFIG.get("report_web_research_enabled", True):
        return {"enabled": False, "queries": [], "openalex": [], "crossref_verified": []}
    queries = build_queries(codings)
    openalex = fetch_openalex_overview(queries)
    crossref_verified = verify_dois_with_crossref(codings)
    return {
        "enabled": True,
        "queries": queries,
        "openalex": openalex,
        "crossref_verified": crossref_verified,
    }


enable_function_tracing(globals(), __name__, include_private=True)

