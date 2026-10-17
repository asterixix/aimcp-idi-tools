import time
import os
from slr_core import io, config, ocr
from slr_core.config import CONFIG, logger
from slr_core.progress import ProgressBars
from slr_core.schema import missing_required_fields
from slr_core.utils import enable_function_tracing


def run_full_pipeline():
    """Simple orchestration for full pipeline.
    If slr_core.llm exists, uses it; otherwise performs placeholder codings
    so the runner is runnable for testing.
    """
    start_time = time.time()
    codings = io.load_existing_codings(CONFIG.get("output_dir")) or []
    failed = []
    existing_by_filename = {
        c.get("_filename"): c for c in codings if isinstance(c, dict) and c.get("_filename")
    }

    documents = io.load_documents(CONFIG.get("literature_dir"))
    bib = io.load_bibliografia(CONFIG.get("bibliografia_path"))
    progress = ProgressBars(enabled=True)
    processed = 0

    # Try to import an LLM client if user has modularized it
    llm_client = None
    try:
        from slr_core import llm

        llm_client = llm.LLMClient(CONFIG)
    except Exception:
        logger.info("No slr_core.llm found — running in placeholder mode (no LLM).")

    try:
        total_docs = len(documents)
        progress.set_overall(current=0, total=total_docs, label="start")
        for doc in documents:
            current_doc = doc["filename"]
            existing = existing_by_filename.get(current_doc)
            if existing:
                if CONFIG.get("skip_if_checkpoint_exists", True):
                    logger.info("Pomijam (checkpoint istnieje): %s", current_doc)
                    processed += 1
                    progress.set_overall(current=processed, total=total_docs, label=current_doc)
                    continue
                missing = missing_required_fields(existing)
                needs_recode = bool(existing.get("_needs_recoding", False))
                if not missing and not needs_recode:
                    logger.info("Pomijam (już poprawnie zakodowany): %s", current_doc)
                    processed += 1
                    progress.set_overall(current=processed, total=total_docs, label=current_doc)
                    continue
                if needs_recode and missing:
                    logger.info("Rekoduję %s (braki + flaga recoding)", current_doc)
                elif needs_recode:
                    logger.info("Rekoduję %s (flaga recoding)", current_doc)
                else:
                    logger.info("Rekoduję %s (braki: %s)", current_doc, ", ".join(missing))
                codings = [c for c in codings if c.get("_filename") != current_doc]
            logger.info("Kodowanie: %s", current_doc)

            try:
                if llm_client is not None:
                    def _on_progress(**evt):
                        phase = evt.get("phase")
                        if phase == "chunk":
                            progress.set_inner(
                                current=int(evt.get("current", 0)),
                                total=int(evt.get("total", 0)),
                                label=str(evt.get("label", "")),
                            )
                        elif phase in {"synthesis", "single"}:
                            progress.set_inner(current=1, total=1, label=str(evt.get("label", "")))

                    result = llm_client.code_document(doc["content"], doc["filename"], progress_callback=_on_progress)
                else:
                    # placeholder minimal coding
                    result = {
                        "PUB_ID": os.path.splitext(doc["filename"])[0],
                        "PUB_Title": io._safe_stem(doc["filename"]),
                        "PUB_KeyFinding": doc["content"][:300],
                        "SLR_Inclusion": "wątpliwe",
                    }
            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.exception("Błąd podczas kodowania %s: %s", current_doc, e)
                failed.append(current_doc)
                continue

            if result is None:
                logger.warning("Nie uzyskano wyniku dla %s", current_doc)
                failed.append(current_doc)
                processed += 1
                progress.clear_inner()
                progress.set_overall(current=processed, total=total_docs, label=current_doc)
                continue

            result["_filename"] = doc["filename"]
            result["_filepath"] = doc["filepath"]
            result["_source_text"] = doc["content"][:4000]
            cp = io.save_coding_checkpoint(result, CONFIG.get("output_dir"))
            io.save_memo_for_coding(result, CONFIG.get("output_dir"))
            codings.append(result)
            io.save_codings_aggregate(codings, CONFIG.get("output_dir"))
            processed += 1
            progress.clear_inner()
            progress.set_overall(current=processed, total=total_docs, label=current_doc)

    except KeyboardInterrupt:
        progress.finish("Przerwano kodowanie.")
        io.save_run_outputs(codings, llm_client, CONFIG.get("output_dir"), bib, len(documents), failed, start_time, interrupted=True, current_doc=current_doc)
        logger.info("Wyjście po przerwaniu (Ctrl+C).")
        return

    run_outputs = io.save_run_outputs(
        codings,
        llm_client,
        CONFIG.get("output_dir"),
        bib,
        len(documents),
        failed,
        start_time,
        interrupted=False,
    )
    audit = (run_outputs or {}).get("audit") or {}
    reliability = (run_outputs or {}).get("reliability") or {}
    if CONFIG.get("quality_gate_fail_on_threshold", True) and audit.get("gate_failed"):
        threshold = audit.get("threshold", CONFIG.get("quality_gate_uncertainty_threshold", 60))
        queue_count = audit.get("queue_count", 0)
        queue_path = audit.get("queue_json_path", "recoding_queue.json")
        raise RuntimeError(
            f"Quality gate failed: uncertainty_score > {threshold} for {queue_count} documents. "
            f"Queue saved to {queue_path}"
        )
    if CONFIG.get("reliability_gate_enabled", True) and reliability.get("gate_failed"):
        alpha = reliability.get("inclusion_alpha")
        threshold = reliability.get("threshold", CONFIG.get("reliability_threshold", 0.67))
        raise RuntimeError(
            f"Reliability gate failed: Krippendorff alpha={alpha} < {threshold} "
            f"(see {CONFIG.get('reliability_report_json')})"
        )
    if CONFIG.get("human_review_required", True):
        try:
            from slr_core.review import review_queue_summary

            queue = review_queue_summary()
            counts = (queue or {}).get("counts") or {}
            pending = int(counts.get("pending", 0))
            rejected = int(counts.get("rejected", 0))
            if pending > 0 or rejected > 0:
                raise RuntimeError(
                    f"Human verification required: pending={pending}, rejected={rejected}. "
                    "Run `python run.py review` and then `python run.py finalize`."
                )
        except RuntimeError:
            raise
        except Exception:
            logger.exception("Failed to evaluate human review gate")
    progress.finish("Kodowanie zakończone.")


def code_single_document():
    lit_dir = os.path.abspath(CONFIG.get("literature_dir"))
    files = [f for f in os.listdir(lit_dir) if f.endswith(".md")] if os.path.exists(lit_dir) else []
    if not files:
        logger.warning("Brak plików .md w katalogu literatury.")
        return

    print("Dostępne pliki:")
    for i, f in enumerate(files, 1):
        print(f"  {i}. {f}")

    try:
        choice = input("Wybierz numer pliku do zakodowania: ").strip()
        idx = int(choice) - 1
        if idx < 0 or idx >= len(files):
            print("Nieprawidłowy wybór.")
            return
        path = os.path.join(lit_dir, files[idx])
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
    except Exception as e:
        logger.exception("Błąd wyboru pliku: %s", e)
        return

    # Use LLM if available
    try:
        from slr_core import llm
        client = llm.LLMClient(CONFIG)
        progress = ProgressBars(enabled=True)

        def _on_progress(**evt):
            phase = evt.get("phase")
            if phase == "chunk":
                progress.set_overall(
                    current=int(evt.get("current", 0)),
                    total=int(evt.get("total", 0)),
                    label=str(evt.get("label", "")),
                )
            elif phase in {"synthesis", "single"}:
                progress.set_overall(current=1, total=1, label=str(evt.get("label", "")))

        res = client.code_document(content, os.path.basename(path), progress_callback=_on_progress)
        progress.finish("Kodowanie dokumentu zakończone.")
    except Exception:
        logger.info("LLM client not available; running placeholder coding.")
        res = {
            "PUB_ID": os.path.splitext(os.path.basename(path))[0],
            "PUB_Title": os.path.basename(path),
            "PUB_KeyFinding": content[:300],
            "SLR_Inclusion": "wątpliwe",
        }

    if res is None:
        logger.warning("Nie udało się uzyskać kodowania.")
        return

    res["_filename"] = os.path.basename(path)
    res["_filepath"] = path
    io.save_coding_checkpoint(res, CONFIG.get("output_dir"))
    io.save_memo_for_coding(res, CONFIG.get("output_dir"))
    logger.info("Kodowanie zakończone i zapisane jako checkpoint.")


enable_function_tracing(globals(), __name__, include_private=True)
