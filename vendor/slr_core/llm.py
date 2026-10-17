import json
import re
import time
from difflib import SequenceMatcher
from typing import Callable, Optional

import requests

from . import chunks as chunks_module
from . import prompt as prompt_module
from .config import logger
from .schema import CODING_DEFAULTS, build_coding_json_schema, missing_required_fields, normalize_coding
from .utils import _safe_stem, enable_function_tracing

SYSTEM_PROMPT = "You are a meticulous qualitative coding assistant that outputs valid JSON only."


class LLMClient:
    """LLM client with prompt hardening and schema-driven retries."""

    def __init__(self, config: dict):
        self.config = config
        self.api_base = config.get("api_base")
        self.api_key = config.get("api_key")
        self.model = config.get("model")
        self.max_retries = int(config.get("max_retries", 3))
        self.retry_delay = float(config.get("retry_delay", 5))
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_requests = 0
        self.total_time = 0.0
        self.log_lines: list[str] = []
        self.coding_scheme_excerpt = self._load_coding_scheme_excerpt()
        self._json_schema = build_coding_json_schema()
        self._request_seq = 0
        self._anthropic = None

    def _log(self, msg: str) -> None:
        logger.info(msg)
        self.log_lines.append(msg)

    def _load_coding_scheme_excerpt(self) -> str:
        path = self.config.get("coding_scheme_path")
        if not path:
            return ""
        try:
            with open(path, "r", encoding="utf-8") as handle:
                text = handle.read()
            return text[:8000]
        except Exception as exc:
            self._log(f"Nie udało się wczytać schematu kodowania: {exc}")
            return ""

    def _extract_json(self, text: str) -> Optional[dict]:
        try:
            cleaned = re.sub(r"```(?:json)?\n", "", text or "")
            cleaned = re.sub(r"```", "", cleaned)
            match = re.search(r"(\{.*\})", cleaned, re.S)
            if match:
                return json.loads(match.group(1))
            return json.loads(cleaned)
        except Exception:
            return None

    def _structured_output_payload(self) -> dict:
        provider = str(self.config.get("api_provider", "openai_compatible")).lower()
        if provider in {"openai", "openai_compatible", "openrouter", "generic"}:
            return {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "slr_coding",
                        "schema": self._json_schema,
                        "strict": True,
                    },
                }
            }
        if provider in {"anthropic", "claude"}:
            return {
                "output_config": {
                    "format": {
                        "type": "json_schema",
                        "name": "slr_coding",
                        "schema": self._json_schema,
                    }
                }
            }
        return {}

    def _request_body(
        self,
        prompt: str,
        *,
        stream: bool,
        model_override: Optional[str] = None,
        structured: bool = False,
        max_tokens_override: Optional[int] = None,
    ) -> dict:
        body = {
            "model": model_override or self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": float(self.config.get("temperature", 0.15)),
            "max_tokens": int(max_tokens_override or self.config.get("max_tokens_per_doc", 8192)),
        }
        if self.config.get("disable_thinking_mode", True):
            # Best-effort hints; unknown fields are usually ignored by OpenAI-compatible routers.
            # "none" wyłącza myślenie DeepSeek V4.1 Flash w OpenCode Go; przy "low" rozumowanie potrafiło zająć cały limit max_tokens.
            body["reasoning_effort"] = "none"
            body["include_reasoning"] = False
        if structured and self.config.get("structured_output_enabled", True):
            body.update(self._structured_output_payload())
        if stream:
            body["stream"] = True
        return body

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json", **(self.config.get("extra_headers") or {})}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _is_messages_model(self, model: Optional[str]) -> bool:
        return (model or self.model) in (self.config.get("messages_api_models") or [])

    def _make_messages_request(
        self,
        prompt: str,
        *,
        model: str,
        structured: bool,
        max_tokens_override: Optional[int],
        timeout_read_override: Optional[int],
    ) -> Optional[str]:
        """Anthropic Messages API (np. Claude przez OpenCode Go); SDK sam ponawia 429/5xx i błędy połączenia."""
        import anthropic

        if self._anthropic is None:
            self._anthropic = anthropic.Anthropic(
                api_key=self.api_key,
                base_url=self.api_base.rstrip("/").removesuffix("/v1"),  # SDK dopisuje /v1/messages
                max_retries=self.max_retries,
                default_headers=self.config.get("extra_headers") or {},
            )
        kwargs = {
            "model": model,
            "max_tokens": int(max_tokens_override or self.config.get("max_tokens_per_doc", 8192)),
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
            "timeout": float(timeout_read_override or self.config.get("request_read_timeout_seconds", 600)),
        }
        # Bez temperature: Claude Haiku 5.5 odrzuca wartości inne niż domyślna (HTTP 400).
        if self.config.get("disable_thinking_mode", True):
            # Limity max_tokens w CONFIG liczono bez myślenia; Haiku 5.5 przyjmuje "disabled" przy domyślnym effort.
            kwargs["thinking"] = {"type": "disabled"}
        if structured and self.config.get("structured_output_enabled", True):
            # Messages API nie przyjmuje minimum/maximum dla integer (HTTP 400); zakresy sprawdza _numeric_validation_issues.
            properties = {
                key: {k: v for k, v in prop.items() if k not in ("minimum", "maximum")}
                for key, prop in self._json_schema["properties"].items()
            }
            kwargs["output_config"] = {"format": {"type": "json_schema", "schema": {**self._json_schema, "properties": properties}}}
        start = time.time()
        try:
            response = self._anthropic.messages.create(**kwargs)
        except anthropic.APIStatusError as exc:
            if not (structured and exc.status_code == 400):
                self._log(f"LLM HTTP {exc.status_code}: {exc.message}")
                return None
            self._log(f"Structured output request failed ({exc.message}); retrying once without structured mode.")
            response = None
        except anthropic.APIConnectionError as exc:
            self._log(f"LLM request exception: {exc}")
            return None
        finally:
            self.total_time += time.time() - start
        if response is None:
            return self._make_messages_request(
                prompt,
                model=model,
                structured=False,
                max_tokens_override=max_tokens_override,
                timeout_read_override=timeout_read_override,
            )
        self.total_requests += 1
        self.total_prompt_tokens += response.usage.input_tokens
        self.total_completion_tokens += response.usage.output_tokens
        if response.stop_reason != "end_turn":  # max_tokens: JSON ucięty; refusal: brak treści
            self._log(f"LLM stop_reason={response.stop_reason} {getattr(response, 'stop_details', None)}")
            return None
        return "".join(block.text for block in response.content if block.type == "text")

    def _make_request(
        self,
        prompt: str,
        *,
        model_override: Optional[str] = None,
        structured: bool = False,
        max_tokens_override: Optional[int] = None,
        timeout_read_override: Optional[int] = None,
    ) -> Optional[str]:
        if self._is_messages_model(model_override):
            return self._make_messages_request(
                prompt,
                model=model_override or self.model,
                structured=structured,
                max_tokens_override=max_tokens_override,
                timeout_read_override=timeout_read_override,
            )
        url = self.api_base.rstrip("/") + "/chat/completions"
        headers = self._headers()

        attempt = 0
        while attempt < self.max_retries:
            attempt += 1
            start = time.time()
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=self._request_body(
                        prompt,
                        stream=False,
                        model_override=model_override,
                        structured=structured,
                        max_tokens_override=max_tokens_override,
                    ),
                    timeout=(
                        int(self.config.get("request_connect_timeout_seconds", 30)),
                        int(timeout_read_override or self.config.get("request_read_timeout_seconds", 600)),
                    ),
                )
                elapsed = time.time() - start
                self.total_time += elapsed
                if response.status_code != 200:
                    if structured:
                        self._log("Structured output request failed; retrying once without structured mode.")
                        return self._make_request(prompt, model_override=model_override, structured=False)
                    self._log(f"LLM HTTP {response.status_code}: {response.text[:240]}")
                    time.sleep(self.retry_delay * attempt)
                    continue
                payload = response.json()
                content = self._extract_content(payload)
                if not content:
                    payload_preview = json.dumps(payload, ensure_ascii=False)[:800]
                    self._log(f"LLM empty content (attempt {attempt}); payload_preview={payload_preview}")
                    time.sleep(self.retry_delay * attempt)
                    continue
                self._update_usage(payload)
                return content
            except requests.exceptions.RequestException as exc:
                self._log(f"LLM request exception: {exc} (attempt {attempt})")
                time.sleep(self.retry_delay * attempt)
            except Exception as exc:
                self._log(f"LLM parse exception: {exc} (attempt {attempt})")
                time.sleep(self.retry_delay * attempt)
        return None

    def _make_request_stream(
        self,
        prompt: str,
        *,
        model_override: Optional[str] = None,
        structured: bool = False,
        max_tokens_override: Optional[int] = None,
        timeout_read_override: Optional[int] = None,
    ) -> Optional[str]:
        if self._is_messages_model(model_override):
            return None  # shortcut: brak parsera SSE formatu Messages, _request_text przechodzi na zapytanie bez strumienia
        url = self.api_base.rstrip("/") + "/chat/completions"
        headers = self._headers()

        try:
            response = requests.post(
                url,
                headers=headers,
                json=self._request_body(
                    prompt,
                    stream=True,
                    model_override=model_override,
                    structured=structured,
                    max_tokens_override=max_tokens_override,
                ),
                timeout=(
                    int(self.config.get("request_connect_timeout_seconds", 30)),
                    int(timeout_read_override or self.config.get("request_read_timeout_seconds", 600)),
                ),
                stream=True,
            )
        except Exception as exc:
            self._log(f"Streaming request failed to start: {exc}")
            return None

        if response.status_code != 200:
            if structured:
                self._log("Structured streaming request failed; retrying without structured mode.")
                return self._make_request_stream(prompt, model_override=model_override, structured=False)
            self._log(f"Streaming HTTP {response.status_code}; fallback to non-streaming")
            return None

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        events_count = 0
        try:
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                text = line.strip()
                if text.startswith("data:"):
                    text = text[len("data:") :].strip()
                if text == "[DONE]":
                    break
                events_count += 1
                # Try to parse OpenAI-compatible stream chunk JSON.
                try:
                    payload = json.loads(text)
                except Exception:
                    # Some providers may return plain text fragments.
                    content_parts.append(text)
                    continue

                choices = payload.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    content_piece = delta.get("content")
                    if isinstance(content_piece, str) and content_piece:
                        content_parts.append(content_piece)
                    reasoning_piece = delta.get("reasoning_content")
                    if isinstance(reasoning_piece, str) and reasoning_piece:
                        reasoning_parts.append(reasoning_piece)

                usage = payload.get("usage") or {}
                self.total_prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
                self.total_completion_tokens += int(usage.get("completion_tokens", 0) or 0)

            self.total_requests += 1
            if content_parts:
                assembled_content = "".join(content_parts).strip()
                self._log(
                    f"LLM_STREAM parsed events={events_count} content_chars={len(assembled_content)} "
                    f"reasoning_chars={sum(len(x) for x in reasoning_parts)}"
                )
                return assembled_content

            # If provider returned only reasoning stream without assistant content,
            # signal caller to fallback to non-streaming request.
            if reasoning_parts:
                self._log(
                    f"LLM_STREAM received reasoning-only output (events={events_count}); "
                    "fallback to non-streaming."
                )
                return None

            # Last-resort fallback for odd providers.
            if events_count > 0:
                self._log(f"LLM_STREAM no parseable content from {events_count} events; fallback.")
            return None
        except Exception as exc:
            self._log(f"Error while reading stream: {exc}")
            return None

    def _extract_content(self, payload: dict) -> str:
        def _flatten_content(value) -> str:
            if value is None:
                return ""
            if isinstance(value, str):
                return value
            if isinstance(value, list):
                parts = []
                for item in value:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict):
                        text = item.get("text")
                        if isinstance(text, str) and text:
                            parts.append(text)
                            continue
                        for key in ("content", "value", "output_text"):
                            v = item.get(key)
                            if isinstance(v, str) and v:
                                parts.append(v)
                                break
                return "".join(parts).strip()
            if isinstance(value, dict):
                for key in ("text", "content", "value", "output_text"):
                    v = value.get(key)
                    if isinstance(v, str) and v:
                        return v
                return ""
            return str(value)

        choices = payload.get("choices") or []
        if choices and isinstance(choices[0], dict):
            first = choices[0]
            message = first.get("message") or {}
            content = _flatten_content(message.get("content")) if isinstance(message, dict) else ""
            if content:
                return content

            text = _flatten_content(first.get("text"))
            if text:
                return text

            # Recovery path for providers that put JSON in reasoning field.
            reasoning = _flatten_content(message.get("reasoning_content")) if isinstance(message, dict) else ""
            if reasoning:
                parsed = self._extract_json(reasoning)
                if parsed is not None:
                    return json.dumps(parsed, ensure_ascii=False)
                return reasoning

        for key in ("content", "output_text", "text"):
            root_text = _flatten_content(payload.get(key))
            if root_text:
                return root_text
        return ""

    def _update_usage(self, payload: dict) -> None:
        usage = payload.get("usage") or {}
        self.total_prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        self.total_completion_tokens += int(usage.get("completion_tokens", 0) or 0)
        self.total_requests += 1

    def _request_text(
        self,
        prompt: str,
        *,
        model_override: Optional[str] = None,
        structured: bool = False,
        expect_json: bool = True,
        prefer_stream: Optional[bool] = None,
        max_tokens_override: Optional[int] = None,
        timeout_read_override: Optional[int] = None,
    ) -> Optional[str]:
        self._request_seq += 1
        req_id = self._request_seq
        model_name = model_override or self.model
        prompt_preview_limit = int(self.config.get("llm_log_prompt_preview_chars", 1500))
        response_preview_limit = int(self.config.get("llm_log_response_preview_chars", 2000))
        prompt_preview = (prompt or "")[:prompt_preview_limit]
        self._log(
            f"LLM_REQ[{req_id}] model={model_name} structured={structured} prompt_preview={json.dumps(prompt_preview, ensure_ascii=False)}"
        )
        if self.config.get("llm_log_full_prompt", False):
            self._log(f"LLM_REQ_FULL[{req_id}] {prompt}")

        text: Optional[str] = None
        stream_enabled = self.config.get("enable_streaming_response", False)
        if prefer_stream is not None:
            stream_enabled = prefer_stream
        if expect_json and self.config.get("force_non_stream_for_json", True):
            stream_enabled = False
        if stream_enabled:
            text = self._make_request_stream(
                prompt,
                model_override=model_override,
                structured=structured,
                max_tokens_override=max_tokens_override,
                timeout_read_override=timeout_read_override,
            )
            # If stream returns something that doesn't parse to JSON where JSON is expected,
            # retry once with non-streaming to improve robustness.
            if text and expect_json and self._extract_json(text) is None:
                self._log("LLM_STREAM output not parseable JSON; retrying non-stream.")
                text = None
        if not text:
            text = self._make_request(
                prompt,
                model_override=model_override,
                structured=structured,
                max_tokens_override=max_tokens_override,
                timeout_read_override=timeout_read_override,
            )
        if text is not None and not str(text).strip():
            self._log("LLM returned blank text payload.")
            text = None

        if text is None:
            self._log(f"LLM_RESP[{req_id}] empty_response")
            return None

        response_preview = text[:response_preview_limit]
        self._log(
            f"LLM_RESP[{req_id}] response_preview={json.dumps(response_preview, ensure_ascii=False)}"
        )
        if self.config.get("llm_log_full_response", False):
            self._log(f"LLM_RESP_FULL[{req_id}] {text}")
        return text

    def _build_excerpt(self, doc_content: str) -> str:
        head_chars = int(self.config.get("coding_excerpt_chars", 22000))
        tail_chars = int(self.config.get("coding_excerpt_tail_chars", 7000))
        if len(doc_content) <= head_chars + tail_chars:
            return doc_content
        head = doc_content[:head_chars]
        tail = doc_content[-tail_chars:]
        return f"{head}\n\n[... POMINIETO FRAGMENT ...]\n\n{tail}"

    def _placeholder(self, doc_filename: str, doc_content: str) -> dict:
        placeholder = dict(CODING_DEFAULTS)
        placeholder["PUB_ID"] = _safe_stem(doc_filename)
        placeholder["PUB_Title"] = _safe_stem(doc_filename)
        placeholder["PUB_KeyFinding"] = (doc_content or "")[:300]
        placeholder["notes"] = "Placeholder coding produced because LLM call failed or returned invalid output."
        placeholder["SLR_Inclusion"] = "wątpliwe"
        return placeholder

    def _supports_structured_mode(self) -> bool:
        mode = str(self.config.get("structured_output_mode", "auto")).lower()
        if mode == "off":
            return False
        if mode == "on":
            return True
        model = str(self.model or "").lower()
        provider = str(self.config.get("api_provider", "openai_compatible")).lower()
        if provider in {"openai", "openai_compatible", "openrouter"}:
            return any(token in model for token in ("gpt-4o", "gpt-5", "o3", "o4"))
        if provider in {"anthropic", "claude"}:
            return any(token in model for token in ("claude", "sonnet", "opus", "haiku"))
        return False

    def _normalize_and_validate(
        self,
        raw_result: dict,
        *,
        pub_id: str,
        title: str,
        excerpt: str,
    ) -> dict:
        coding = normalize_coding(raw_result, pub_id=pub_id, title=title)
        missing = missing_required_fields(coding)
        if not missing:
            return coding

        self._log(f"Detected missing fields: {missing}; attempting repair.")
        current_json = json.dumps(coding, ensure_ascii=False)
        attempts = int(self.config.get("repair_max_attempts", 2))

        for _ in range(attempts):
            repair_prompt = prompt_module.build_repair_prompt(
                previous_json=current_json,
                missing_fields=missing,
                title=title,
                excerpt=excerpt[:12000],
            )
            repair_text = self._request_text(repair_prompt)
            if not repair_text:
                continue
            parsed = self._extract_json(repair_text)
            if not isinstance(parsed, dict):
                continue
            coding = normalize_coding(parsed, pub_id=pub_id, title=title)
            missing = missing_required_fields(coding)
            if not missing:
                return coding
            current_json = json.dumps(coding, ensure_ascii=False)
        return coding

    def _keyword_supported(self, claim: str, source: str) -> bool:
        claim_words = [w for w in re.findall(r"[A-Za-zÀ-ÿ0-9_/-]{5,}", claim.lower())]
        if not claim_words:
            return True
        src = source.lower()
        hits = sum(1 for word in set(claim_words) if word in src)
        return hits / max(1, len(set(claim_words))) >= 0.5

    def _compute_faithfulness(self, coding: dict, excerpt: str) -> float:
        claims: list[str] = []
        # notes pominięte (decyzja autora 2026-10-14): to komentarz kodera o kodach i dopiski programu, nie twierdzenia o źródle.
        for field in ("PUB_KeyFinding", "SLR_InclusionReason", "RES_KeyStat"):
            text = str(coding.get(field, "") or "").strip()
            if text:
                claims.extend([piece.strip() for piece in re.split(r"[.!?]\s+", text) if piece.strip()])
        if not claims:
            return 1.0
        supported = 0
        source = excerpt.lower()
        source_tokens = set(re.findall(r"[A-Za-zÀ-ÿ0-9_/-]{4,}", source))

        def _best_local_ratio(claim_text: str, source_text: str) -> float:
            claim_piece = claim_text[:280]
            if not claim_piece or not source_text:
                return 0.0
            window = max(320, min(1200, len(claim_piece) * 4))
            step = max(120, window // 3)
            best = 0.0
            for start in range(0, max(1, len(source_text) - window + 1), step):
                chunk = source_text[start : start + window]
                ratio = SequenceMatcher(None, claim_piece, chunk).ratio()
                if ratio > best:
                    best = ratio
                    if best >= 0.62:
                        break
            return best

        for claim in claims:
            low = claim.lower()
            if low in source:
                supported += 1
                continue
            claim_tokens = set(re.findall(r"[A-Za-zÀ-ÿ0-9_/-]{4,}", low))
            overlap_ratio = 0.0
            if claim_tokens:
                overlap_ratio = len(claim_tokens & source_tokens) / max(1, len(claim_tokens))
            ratio = _best_local_ratio(low, source)
            if ratio >= 0.44 or overlap_ratio >= 0.5 or self._keyword_supported(claim, excerpt):
                supported += 1
        return round(supported / max(1, len(claims)), 4)

    def _numeric_validation_issues(self, coding: dict) -> list[str]:
        if not self.config.get("numeric_validation_enabled", True):
            return []
        issues: list[str] = []
        try:
            year = int(coding.get("PUB_Year", 0) or 0)
            if year and (year < 1900 or year > 2100):
                issues.append("PUB_Year poza zakresem 1900-2100")
        except Exception:
            issues.append("PUB_Year nie jest liczbą")

        for field in ("AN_RQ1_Relevance", "AN_RQ2_Relevance"):
            val = int(coding.get(field, 0) or 0)
            if val < 0 or val > 3:
                issues.append(f"{field} poza zakresem 0-3")

        for field in ("AN_QFSR", "AN_SC", "AN_VF", "AN_RI", "AN_HR", "AN_SSS", "AN_CER"):
            val = int(coding.get(field, 0) or 0)
            if val < 0 or val > 2:
                issues.append(f"{field} poza zakresem 0-2")

        res_n = str(coding.get("RES_N", "") or "").strip()
        if res_n:
            nums = re.findall(r"\d+", res_n)
            if nums and int(nums[0]) <= 0:
                issues.append("RES_N wskazuje wartość <= 0")

        for field in ("RES_EffectSize", "RES_KeyStat"):
            txt = str(coding.get(field, "") or "")
            for pct in re.findall(r"(\d+(?:[.,]\d+)?)\s*%", txt):
                val = float(pct.replace(",", "."))
                if val < 0 or val > 100:
                    issues.append(f"{field}: procent poza zakresem 0-100")
        return issues

    def _evidence_first_issues(self, coding: dict, excerpt: str) -> list[str]:
        issues: list[str] = []
        min_quotes = int(self.config.get("min_evidence_quotes_critical", 2))
        quotes = coding.get("evidence_quotes") or []
        if not isinstance(quotes, list):
            quotes = []
        if len(quotes) < min_quotes:
            issues.append(f"evidence_quotes < {min_quotes}")

        critical_fields = self.config.get("critical_fields_for_evidence") or []
        quote_text = " ".join(str(q) for q in quotes)
        for field in critical_fields:
            value = coding.get(field)
            if value in (None, "", []):
                issues.append(f"brak pola krytycznego: {field}")
                continue
            if field.startswith("AN_"):
                continue
            if not self._keyword_supported(str(value), quote_text):
                issues.append(f"słabe powiązanie z cytatami: {field}")
        if quote_text and excerpt and not self._keyword_supported(quote_text[:500], excerpt):
            issues.append("cytaty nie wyglądają na osadzone w source")
        return issues

    def _repair_for_evidence(self, coding: dict, excerpt: str, model_override: Optional[str]) -> dict:
        issues = self._evidence_first_issues(coding, excerpt)
        if not issues:
            return coding
        prompt = prompt_module.build_evidence_repair_prompt(
            previous_json=json.dumps(coding, ensure_ascii=False),
            issues=issues,
            excerpt=excerpt[:14000],
        )
        repaired_text = self._request_text(
            prompt,
            model_override=model_override,
            structured=self._supports_structured_mode(),
        )
        parsed = self._extract_json(repaired_text or "") if repaired_text else None
        if isinstance(parsed, dict):
            return parsed
        return coding

    def _code_with_chunking(
        self,
        doc_content: str,
        doc_filename: str,
        title: str,
        *,
        model_override: Optional[str] = None,
        progress_callback: Optional[Callable[..., None]] = None,
    ) -> Optional[dict]:
        self._log(f"Document large ({len(doc_content):,} chars) — using chunking+synthesis for {doc_filename}")
        target_chars = int(self.config.get("chunk_target_chars", 12000))
        overlap_chars = int(self.config.get("chunk_overlap_chars", 1500))
        chunks = chunks_module.build_semantic_chunks(doc_content, target_chars=target_chars, overlap_chars=overlap_chars)
        if not chunks:
            return None

        chunk_evidences = []
        total = len(chunks)
        for chunk in chunks:
            idx = chunk.get("index")
            if progress_callback:
                progress_callback(
                    phase="chunk",
                    current=int(idx or 0),
                    total=total,
                    label=f"{doc_filename}",
                )
            chunk_prompt = prompt_module.build_chunk_prompt(chunk.get("text", ""), idx, total, doc_filename)
            response = self._request_text(
                chunk_prompt,
                model_override=model_override,
                structured=False,
                expect_json=True,
                prefer_stream=False,
                max_tokens_override=int(self.config.get("chunk_request_max_tokens", 700)),
                timeout_read_override=int(self.config.get("chunk_request_timeout_seconds", 90)),
            )
            parsed = self._extract_json(response or "") if response else None
            if isinstance(parsed, dict):
                parsed.setdefault("chunk_index", idx)
                chunk_evidences.append(parsed)
            else:
                chunk_evidences.append(
                    {
                        "chunk_index": idx,
                        "chunk_focus": chunk.get("section_hint"),
                        "evidence_quotes": [chunk.get("preview", "")[:200]],
                        "confidence": 1,
                    }
                )

        synthesis_prompt = prompt_module.build_synthesis_prompt(chunk_evidences, title)
        if progress_callback:
            progress_callback(phase="synthesis", current=1, total=1, label=f"{doc_filename}")
        synthesis_text = self._request_text(
            synthesis_prompt,
            model_override=model_override,
            structured=self._supports_structured_mode(),
            expect_json=True,
            prefer_stream=False,
            max_tokens_override=int(self.config.get("synthesis_request_max_tokens", 1400)),
            timeout_read_override=int(self.config.get("synthesis_request_timeout_seconds", 180)),
        )
        parsed_synthesis = self._extract_json(synthesis_text or "") if synthesis_text else None
        if isinstance(parsed_synthesis, dict):
            return parsed_synthesis
        return None

    def _code_single_pass(
        self,
        *,
        doc_content: str,
        doc_filename: str,
        title: str,
        excerpt: str,
        model_override: Optional[str],
        coder_role: str,
        progress_callback: Optional[Callable[..., None]] = None,
    ) -> dict:
        parsed: Optional[dict] = None
        if self.config.get("chunking_enabled", True) and len(doc_content) > int(self.config.get("chunking_threshold_chars", 30000)):
            parsed = self._code_with_chunking(
                doc_content,
                doc_filename,
                title,
                model_override=model_override,
                progress_callback=progress_callback,
            )

        if not parsed:
            if progress_callback:
                progress_callback(phase="single", current=1, total=1, label=f"{doc_filename}")
            prompt = prompt_module.build_single_shot_prompt(
                title=title,
                excerpt=excerpt,
                coding_scheme_excerpt=self.coding_scheme_excerpt,
                doc_filename=doc_filename,
                coder_role=coder_role,
            )
            self._log(f"Requesting coding for {doc_filename} [{coder_role}]")
            response_text = self._request_text(
                prompt,
                model_override=model_override,
                structured=self._supports_structured_mode(),
                expect_json=True,
                max_tokens_override=int(self.config.get("max_tokens_per_doc", 8192)),
            )
            parsed = self._extract_json(response_text or "") if response_text else None

        if not isinstance(parsed, dict):
            return self._placeholder(doc_filename, doc_content)
        return parsed

    def _compare_codings(self, a: dict, b: dict) -> tuple[bool, dict, float]:
        fields = self.config.get("dual_compare_fields") or []
        cmp = {}
        agree = 0
        rq_tolerance = int(self.config.get("dual_rq_tolerance", 1))
        for field in fields:
            va = a.get(field)
            vb = b.get(field)
            is_agree = str(va) == str(vb)
            if not is_agree and field in {"AN_RQ1_Relevance", "AN_RQ2_Relevance"}:
                try:
                    ia = int(va)
                    ib = int(vb)
                    is_agree = abs(ia - ib) <= rq_tolerance
                except Exception:
                    pass
            cmp[field] = {"a": va, "b": vb, "agree": is_agree}
            if is_agree:
                agree += 1
        score = agree / max(1, len(fields))
        return score >= 1.0, cmp, round(score, 4)

    def _adjudicate(
        self,
        *,
        title: str,
        excerpt: str,
        coder_a: dict,
        coder_b: dict,
    ) -> Optional[dict]:
        if not self.config.get("enable_adjudication_mode", True):
            return None
        prompt = prompt_module.build_adjudication_prompt(
            title=title,
            excerpt=excerpt[:16000],
            coder_a_json=json.dumps(coder_a, ensure_ascii=False),
            coder_b_json=json.dumps(coder_b, ensure_ascii=False),
        )
        adjudicator_model = self.config.get("adjudication_model") or self.model
        response = self._request_text(
            prompt,
            model_override=adjudicator_model,
            structured=self._supports_structured_mode(),
        )
        parsed = self._extract_json(response or "") if response else None
        if isinstance(parsed, dict):
            return parsed
        return None

    def code_document(
        self,
        doc_content: str,
        doc_filename: str,
        progress_callback: Optional[Callable[..., None]] = None,
    ) -> Optional[dict]:
        title = _safe_stem(doc_filename)
        pub_id = _safe_stem(doc_filename)
        excerpt = self._build_excerpt(doc_content)

        dual_enabled = bool(self.config.get("dual_coding_enabled", True))
        primary_model = self.model
        secondary_model = self.config.get("model_secondary")

        parsed: Optional[dict] = None
        coder_a_raw = self._code_single_pass(
            doc_content=doc_content,
            doc_filename=doc_filename,
            title=title,
            excerpt=excerpt,
            model_override=primary_model,
            coder_role="coder_a",
            progress_callback=progress_callback,
        )
        coder_a = normalize_coding(coder_a_raw, pub_id=pub_id, title=title) if isinstance(coder_a_raw, dict) else {}

        coder_b = None
        comparison = {}
        agreement_score = 1.0
        dual_disagreement = False
        adjudication_succeeded = False
        adjudication_fallback = False
        if dual_enabled and secondary_model:
            coder_b_raw = self._code_single_pass(
                doc_content=doc_content,
                doc_filename=doc_filename,
                title=title,
                excerpt=excerpt,
                model_override=secondary_model,
                coder_role="coder_b",
                progress_callback=progress_callback,
            )
            coder_b = normalize_coding(coder_b_raw, pub_id=pub_id, title=title) if isinstance(coder_b_raw, dict) else {}
            agree, comparison, agreement_score = self._compare_codings(coder_a, coder_b)
            if agree:
                parsed = coder_a
            else:
                dual_disagreement = True
                self._log(f"Dual-coding disagreement for {doc_filename}; entering adjudication.")
                adjudicated = self._adjudicate(title=title, excerpt=excerpt, coder_a=coder_a, coder_b=coder_b)
                if isinstance(adjudicated, dict):
                    parsed = adjudicated
                    adjudication_succeeded = True
                else:
                    parsed = coder_a
                    adjudication_fallback = True
        else:
            parsed = coder_a

        if not isinstance(parsed, dict):
            self._log("LLM response invalid or empty — using placeholder.")
            parsed = self._placeholder(doc_filename, doc_content)

        parsed = self._repair_for_evidence(parsed, excerpt, primary_model)
        normalized = self._normalize_and_validate(parsed, pub_id=pub_id, title=title, excerpt=excerpt)
        if missing_required_fields(normalized):
            self._log("Response still incomplete after repair attempts — keeping conservative fallback fields.")
            fallback = self._placeholder(doc_filename, doc_content)
            fallback.update(normalized)
            normalized = fallback

        evidence_issues = self._evidence_first_issues(normalized, excerpt)
        if evidence_issues:
            normalized["notes"] = (normalized.get("notes", "") + " | evidence_issues: " + "; ".join(evidence_issues)).strip(" |")
        numeric_issues = self._numeric_validation_issues(normalized)
        if numeric_issues:
            normalized["_needs_manual_numeric_review"] = True
            normalized["notes"] = (normalized.get("notes", "") + " | numeric_issues: " + "; ".join(numeric_issues)).strip(" |")

        faithfulness = self._compute_faithfulness(normalized, excerpt)
        normalized["_faithfulness_score"] = faithfulness
        threshold = float(self.config.get("faithfulness_threshold", 0.7))
        if faithfulness < threshold and self.config.get("mark_recode_on_low_faithfulness", True):
            normalized["_needs_recoding"] = True
            normalized["notes"] = (
                normalized.get("notes", "") + f" | low_faithfulness:{faithfulness:.2f}< {threshold:.2f}"
            ).strip(" |")

        normalized["_dual_model_primary"] = primary_model
        normalized["_dual_model_secondary"] = secondary_model or ""
        normalized["_dual_disagreement"] = dual_disagreement
        normalized["_dual_agreement_score"] = agreement_score
        normalized["_dual_comparison"] = comparison
        if coder_a is not None:
            normalized["_coder_a"] = {k: coder_a.get(k) for k in (self.config.get("dual_compare_fields") or [])}
        if coder_b is not None:
            normalized["_coder_b"] = {k: coder_b.get(k) for k in (self.config.get("dual_compare_fields") or [])}

        if dual_disagreement and self.config.get("mark_recode_on_dual_disagreement", False):
            normalized["_needs_recoding"] = True
        if dual_disagreement and (not adjudication_succeeded) and self.config.get(
            "mark_recode_when_adjudication_fallback", True
        ):
            normalized["_needs_recoding"] = True
            normalized["notes"] = (
                normalized.get("notes", "") + " | adjudication_fallback(coder_a)"
            ).strip(" |")
        if adjudication_fallback:
            normalized["_adjudication_fallback"] = True
        if dual_disagreement:
            normalized["_adjudication_mode"] = "llm_adjudication"

        return normalized


enable_function_tracing(globals(), __name__, include_private=True)


__all__ = ["LLMClient"]
