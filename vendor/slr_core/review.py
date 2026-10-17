"""Human verification queue and finalize flow."""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional

from .config import CONFIG, logger
from .utils import _atomic_write_json, _atomic_write_text, enable_function_tracing


def _load_codings() -> list[dict]:
    path = CONFIG.get("codings_path")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, list) else []


def _load_queue() -> dict:
    path = CONFIG.get("review_queue_json")
    if not os.path.exists(path):
        return {"generated_at": None, "items": []}
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        return {"generated_at": None, "items": []}
    if not isinstance(data.get("items"), list):
        data["items"] = []
    return data


def _save_queue(queue: dict) -> str:
    path = CONFIG.get("review_queue_json")
    _atomic_write_json(path, queue)
    return path


def _sync_codings_review_fields(queue_items: list[dict]) -> None:
    codings = _load_codings()
    if not codings:
        return
    by_filename = {item.get("_filename"): item for item in queue_items}
    changed = False
    for coding in codings:
        item = by_filename.get(coding.get("_filename"))
        if not item:
            continue
        status = item.get("status", "pending")
        reviewer = item.get("reviewer", "")
        comment = item.get("review_comment", "")
        if coding.get("_human_review_status") != status:
            coding["_human_review_status"] = status
            changed = True
        if coding.get("_human_reviewer") != reviewer:
            coding["_human_reviewer"] = reviewer
            changed = True
        if coding.get("_human_review_comment") != comment:
            coding["_human_review_comment"] = comment
            changed = True
    if changed:
        _atomic_write_json(CONFIG.get("codings_path"), codings)


def build_review_queue(force_refresh: bool = False) -> dict:
    codings = _load_codings()
    queue = _load_queue()
    existing = {item.get("_filename"): item for item in queue.get("items", [])}

    items = []
    for coding in codings:
        filename = coding.get("_filename")
        if not filename:
            continue
        prior = existing.get(filename, {})
        status = "pending" if force_refresh else prior.get("status", "pending")
        item = {
            "PUB_ID": coding.get("PUB_ID", ""),
            "PUB_Title": coding.get("PUB_Title", ""),
            "_filename": filename,
            "_filepath": coding.get("_filepath", ""),
            "status": status,
            "reviewer": prior.get("reviewer", ""),
            "review_comment": prior.get("review_comment", ""),
            "updated_at": prior.get("updated_at", ""),
            "flags": {
                "needs_recoding": bool(coding.get("_needs_recoding", False)),
                "dual_disagreement": bool(coding.get("_dual_disagreement", False)),
                "numeric_review": bool(coding.get("_needs_manual_numeric_review", False)),
                "faithfulness_score": coding.get("_faithfulness_score"),
            },
        }
        items.append(item)

    queue_doc = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "items": items,
        "counts": {
            "total": len(items),
            "pending": sum(1 for i in items if i.get("status") == "pending"),
            "approved": sum(1 for i in items if i.get("status") == "approved"),
            "rejected": sum(1 for i in items if i.get("status") == "rejected"),
        },
    }
    _save_queue(queue_doc)
    _sync_codings_review_fields(items)
    logger.info("Review queue saved: %s (total=%d)", CONFIG.get("review_queue_json"), len(items))
    return queue_doc


def _update_item_status(filename: str, status: str, reviewer: str, comment: str) -> dict:
    queue = _load_queue()
    items = queue.get("items", [])
    found = False
    for item in items:
        if item.get("_filename") == filename:
            item["status"] = status
            item["reviewer"] = reviewer
            item["review_comment"] = comment
            item["updated_at"] = datetime.now().isoformat(timespec="seconds")
            found = True
            break
    if not found:
        raise ValueError(f"Nie znaleziono dokumentu w review queue: {filename}")
    queue["counts"] = {
        "total": len(items),
        "pending": sum(1 for i in items if i.get("status") == "pending"),
        "approved": sum(1 for i in items if i.get("status") == "approved"),
        "rejected": sum(1 for i in items if i.get("status") == "rejected"),
    }
    _save_queue(queue)
    _sync_codings_review_fields(items)
    return queue


def approve_document(filename: str, reviewer: str = "human", comment: str = "") -> dict:
    return _update_item_status(filename, "approved", reviewer, comment)


def reject_document(filename: str, reviewer: str = "human", comment: str = "") -> dict:
    return _update_item_status(filename, "rejected", reviewer, comment)


def review_queue_summary() -> dict:
    queue = build_review_queue(force_refresh=False)
    logger.info("Review queue summary: %s", queue.get("counts"))
    return queue


def interactive_review_queue(reviewer: str = "human") -> None:
    queue = build_review_queue(force_refresh=False)
    items = queue.get("items", [])
    pending = [i for i in items if i.get("status") == "pending"]
    if not pending:
        print("Brak dokumentów pending w review queue.")
        return

    print(f"Pending records: {len(pending)}")
    for item in pending:
        print("\n" + "=" * 80)
        print(f"PUB_ID: {item.get('PUB_ID')}")
        print(f"Title : {item.get('PUB_Title')}")
        print(f"File  : {item.get('_filename')}")
        print(f"Flags : {item.get('flags')}")
        choice = input("Decision [a=approve, r=reject, s=skip, q=quit]: ").strip().lower()
        if choice == "q":
            break
        if choice == "s":
            continue
        comment = input("Comment (optional): ").strip()
        if choice == "a":
            approve_document(item.get("_filename"), reviewer=reviewer, comment=comment)
            print("Approved.")
        elif choice == "r":
            reject_document(item.get("_filename"), reviewer=reviewer, comment=comment)
            print("Rejected.")
        else:
            print("Unknown decision, skipping.")


def finalize_approved_codings() -> str:
    queue = build_review_queue(force_refresh=False)
    items = queue.get("items", [])
    pending = [i for i in items if i.get("status") not in {"approved", "rejected"}]
    if pending:
        sample = ", ".join((i.get("_filename") or "unknown") for i in pending[:5])
        raise RuntimeError(
            f"Finalize blocked: {len(pending)} records are still pending human review. "
            f"Resolve review queue first. Sample: {sample}"
        )

    codings = _load_codings()
    approved_files = {i.get("_filename") for i in items if i.get("status") == "approved"}
    rejected_files = {i.get("_filename") for i in items if i.get("status") == "rejected"}
    final = [c for c in codings if c.get("_filename") in approved_files]
    rejected = [c for c in codings if c.get("_filename") in rejected_files]

    if CONFIG.get("finalize_require_all_documents", True):
        # Finalization gate should match reviewed/coded scope, not full literature directory.
        # This supports incremental runs where only a subset is coded and reviewed.
        expected_docs = len(items)
        reviewed_docs = len(final) + len(rejected)
        if reviewed_docs < expected_docs:
            raise RuntimeError(
                "Finalize blocked: reviewed codings "
                f"{reviewed_docs} < expected documents {expected_docs}."
            )

    final_path = CONFIG.get("final_codings_path")
    rejected_path = CONFIG.get("final_rejected_codings_path")
    _atomic_write_json(final_path, final)
    _atomic_write_json(rejected_path, rejected)

    approval_note = os.path.join(CONFIG.get("output_dir"), "finalize_report.md")
    _atomic_write_text(
        approval_note,
        "\n".join(
            [
                "# Finalize report",
                "",
                f"- finalized_at: {datetime.now().isoformat(timespec='seconds')}",
                f"- approved_records: {len(final)}",
                f"- rejected_records: {len(rejected)}",
                f"- pending_records: {len(pending)}",
                f"- source_queue: {CONFIG.get('review_queue_json')}",
                f"- output: {final_path}",
                f"- rejected_output: {rejected_path}",
                "",
                "All records are human-reviewed (approved or rejected).",
            ]
        ),
    )
    logger.info("Finalize completed: %s", final_path)
    return final_path


enable_function_tracing(globals(), __name__, include_private=True)
