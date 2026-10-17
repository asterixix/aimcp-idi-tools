"""Interactive CLI menu for SLR workflows."""
from __future__ import annotations

from .config import logger
from .ocr import ocr_only
from .audit import run_quality_audit_from_existing
from .qda import export_qda_from_existing
from .report import generate_report_from_existing
from .runner import code_single_document, run_full_pipeline
from .review import finalize_approved_codings, interactive_review_queue, review_queue_summary
from .utils import enable_function_tracing


MENU_ITEMS = {
    "1": ("OCR wszystkich PDF", ocr_only),
    "2": ("Kodowanie wszystkich dokumentów", run_full_pipeline),
    "3": ("Kodowanie pojedynczego dokumentu", code_single_document),
    "4": ("Wygeneruj raport z istniejących kodowań", generate_report_from_existing),
    "5": ("Audyt jakości kodowania", run_quality_audit_from_existing),
    "6": ("Eksport do pakietu QDA", export_qda_from_existing),
    "7": ("Review queue (human approval)", interactive_review_queue),
    "8": ("Review queue summary", review_queue_summary),
    "9": ("Finalize approved codings", finalize_approved_codings),
}


def interactive_menu() -> None:
    print("\n=== SLR AutoCoder — menu ===")
    for key, (label, _) in MENU_ITEMS.items():
        print(f"{key}. {label}")
    print("0. Wyjście")

    while True:
        choice = input("\nWybierz opcję: ").strip()
        if choice == "0":
            print("Zamykam.")
            return
        item = MENU_ITEMS.get(choice)
        if item is None:
            print("Nieprawidłowy wybór. Spróbuj ponownie.")
            continue
        label, action = item
        logger.info("Uruchamiam: %s", label)
        try:
            action()
        except Exception as exc:
            logger.exception("Błąd podczas akcji menu: %s", label)
            print(f"\nOperacja nieudana: {exc}")
            print("Sprawdź logi i dokończ review queue (approve/reject), następnie ponów.")
            return
        print("\nOperacja zakończona.")
        return


enable_function_tracing(globals(), __name__, include_private=True)

