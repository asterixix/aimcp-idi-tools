import os
import uuid
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional
from .progress import get_active_progress


def _load_dotenv_if_present() -> None:
    """Load simple KEY=VALUE pairs from .env files into process env.

    Existing environment variables are not overwritten.
    """
    here = os.path.dirname(__file__)
    candidates = [
        os.path.join(here, ".env"),
        os.path.join(here, "..", ".env"),
    ]

    for dotenv_path in candidates:
        if not os.path.exists(dotenv_path):
            continue
        try:
            with open(dotenv_path, "r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip("'").strip('"')
                    if key and key not in os.environ:
                        os.environ[key] = value
        except Exception:
            # Logging is not initialized yet; fail silently.
            continue


_load_dotenv_if_present()

# Configuration for SLR AutoCoder
CONFIG = {
    # API
    # OpenCode Go (decyzja autora 2026-10-14): koder A i rozjemca Claude Haiku 5.5, koder B DeepSeek V4.1 Flash.
    # Prywatność wg https://opencode.ai/docs/go/#privacy: oba bez trenowania; retencja Haiku 30 dni, DeepSeek 0 dni (ZDR do 31.10.2026, odnawiane co miesiąc).
    "api_base": "https://opencode.ai/inference/go/v1",
    "api_key": os.environ.get("OPENCODEZEN_API_KEY", ""),
    "model": "claude-haiku-5-5",
    "model_secondary": "deepseek-v4.1-flash",
    "adjudication_model": "claude-haiku-5-5",
    # Modele podawane przez OpenCode Go w formacie Anthropic Messages (/v1/messages); pozostałe idą przez /v1/chat/completions.
    "messages_api_models": ["claude-haiku-5-5"],
    # OpenCode Go odrzuca zapytania bez własnego User-Agent i stałego x-opencode-session (HTTP 400 MissingSessionID); sesja = jedno uruchomienie.
    "extra_headers": {"User-Agent": "slr-autocoder/1.0", "x-opencode-session": f"slr-{uuid.uuid4().hex[:12]}"},
    "api_provider": "openai_compatible",
    "structured_output_enabled": True,
    "structured_output_mode": "on",

    # Paths
    "literature_dir": os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "_shared", "literatura", "md"),
    "bibliografia_path": os.path.join(os.path.dirname(__file__), "..", "..", "BIBLIOGRAFIA.md"),
    "coding_scheme_path": os.path.join(os.path.dirname(__file__), "..", "docs", "SLR_CODING_SCHEME.md"),
    "output_dir": os.path.join(os.path.dirname(__file__), "..", "SLR_OUTPUT"),
    # PDF/OCR
    "pdf_dir": os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "_shared", "literatura", "pdf"),
    "literature_ready_dir": os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "_shared", "literatura", "md"),
    "mineru_api_base": "https://mineru.net/api/v1",
    "mineru_api_key": os.environ.get("MINERU_API_KEY", ""),

    # Coding params
    "max_tokens_per_doc": 8192,
    "temperature": 0.15,
    "max_retries": 3,
    "retry_delay": 5,
    "batch_size": 15,
    "inter_request_delay_seconds": 0.0,
    "request_connect_timeout_seconds": 30,
    "request_read_timeout_seconds": 600,
    "log_path": os.path.join(os.path.dirname(__file__), "..", "SLR_OUTPUT", "logs.txt"),
    "llm_log_path": os.path.join(os.path.dirname(__file__), "..", "SLR_OUTPUT", "llm_log.txt"),
    "llm_log_prompt_preview_chars": 1500,
    "llm_log_response_preview_chars": 2000,
    "llm_log_full_prompt": False,
    "llm_log_full_response": False,
    "checkpoint_dir": os.path.join(os.path.dirname(__file__), "..", "SLR_OUTPUT", "checkpoints"),
    "codings_path": os.path.join(os.path.dirname(__file__), "..", "SLR_OUTPUT", "codings.json"),
    "disable_thinking_mode": True,
    "enable_streaming_response": True,
    "force_non_stream_for_json": True,
    "stream_log_preview_chars": 96,
    "chunk_request_max_tokens": 700,
    "synthesis_request_max_tokens": 1400,
    "chunk_request_timeout_seconds": 90,
    "synthesis_request_timeout_seconds": 180,
    "chunking_enabled": True,
    "chunking_threshold_chars": 30000,
    "chunk_target_chars": 12000,
    "chunk_overlap_chars": 1500,
    "chunk_summary_max_tokens": 1024,
    "synthesis_max_tokens": 8192,
    "coding_excerpt_chars": 22000,
    "coding_excerpt_tail_chars": 7000,
    "repair_max_attempts": 2,
    "report_top_items": 10,
    "report_llm_enabled": True,
    "report_llm_model": "",
    "report_llm_max_tokens": 1800,
    "report_llm_timeout_seconds": 120,
    "report_web_research_enabled": True,
    "report_web_email": "",
    "report_web_max_queries": 4,
    "report_web_results_per_query": 5,
    "report_web_timeout_seconds": 20,
    "qda_generate_legacy_qda": False,
    "dual_coding_enabled": True,
    "mark_recode_on_dual_disagreement": False,
    "mark_recode_when_adjudication_fallback": False,
    "dual_compare_fields": [
        "DES_ResearchFocus",
        "AN_RQ1_Relevance",
        "AN_RQ2_Relevance",
        "SLR_Inclusion",
        "SLR_InclusionReason",
    ],
    "dual_rq_tolerance": 1,
    "min_evidence_quotes_critical": 2,
    "critical_fields_for_evidence": [
        "PUB_KeyFinding",
        "SLR_InclusionReason",
        "AN_RQ1_Relevance",
        "AN_RQ2_Relevance",
    ],
    "faithfulness_threshold": 0.7,
    "mark_recode_on_low_faithfulness": True,
    "numeric_validation_enabled": True,
    "enable_adjudication_mode": True,
    "reliability_threshold": 0.67,
    "reliability_gate_enabled": True,
    "reliability_report_json": os.path.join(os.path.dirname(__file__), "..", "SLR_OUTPUT", "reliability_report.json"),
    "reliability_report_md": os.path.join(os.path.dirname(__file__), "..", "SLR_OUTPUT", "reliability_report.md"),
    # quality gate
    "quality_gate_enabled": True,
    "quality_gate_fail_on_threshold": True,
    "quality_gate_uncertainty_threshold": 60,
    "queue_include_needs_recoding_flag": False,
    "quality_gate_queue_json": os.path.join(os.path.dirname(__file__), "..", "SLR_OUTPUT", "recoding_queue.json"),
    "quality_gate_queue_txt": os.path.join(os.path.dirname(__file__), "..", "SLR_OUTPUT", "recoding_queue.txt"),
    # human verification gate
    "human_review_required": True,
    "review_queue_json": os.path.join(os.path.dirname(__file__), "..", "SLR_OUTPUT", "review_queue.json"),
    "final_codings_path": os.path.join(os.path.dirname(__file__), "..", "SLR_OUTPUT", "final_codings.json"),
    "final_rejected_codings_path": os.path.join(os.path.dirname(__file__), "..", "SLR_OUTPUT", "final_rejected_codings.json"),
    "finalize_require_all_documents": True,
    # qc integration
    "qc_integration": False,
    "qc_project_dir": os.path.join(os.path.dirname(__file__), "..", "QC_PROJECT"),

    # Limits
    "max_content_chars": 1000000,
    "min_content_chars": 100,
}


def setup_logging(log_path: Optional[str] = None):
    logger = logging.getLogger("slr_auto_coder")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)

    class ProgressAwareStreamHandler(logging.StreamHandler):
        def emit(self, record):
            progress = get_active_progress()
            if progress is not None:
                try:
                    progress.clear_for_log()
                except Exception:
                    pass
            try:
                super().emit(record)
            finally:
                if progress is not None:
                    try:
                        progress.restore_after_log()
                    except Exception:
                        pass

    ch = ProgressAwareStreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S")
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    try:
        log_file = log_path or CONFIG.get("log_path")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=10, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh_formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s [%(threadName)s]: %(message)s"
        )
        fh.setFormatter(fh_formatter)
        logger.addHandler(fh)
    except Exception:
        logger.exception("Nie udało się skonfigurować handlera plikowego dla logów")

    return logger


# Initialize logger for package
logger = setup_logging()
