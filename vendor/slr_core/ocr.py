import os
import time
import zipfile
from io import BytesIO
from pathlib import Path
import requests

from .config import CONFIG, logger
from .progress import ProgressBars
from .utils import _atomic_write_text, _safe_stem, enable_function_tracing


def ocr_pdf_with_mineru(pdf_path: str, md_output_path: str) -> bool:
    """Wywołanie MinerU OCR: zamienia PDF na Markdown i zapisuje wynik.

    Wymaga ustawienia `CONFIG['mineru_api_base']` i klucza w env `MINERU_API_KEY`
    lub w `CONFIG['mineru_api_key']`.
    Zwraca True jeśli zapisano plik markdown.
    """
    base = CONFIG.get("mineru_api_base")
    api_key = os.environ.get("MINERU_API_KEY") or CONFIG.get("mineru_api_key")

    try:
        file_name = os.path.basename(pdf_path)

        def _try_precision_flow() -> bool:
            logger.debug("MinerU precision flow for %s", file_name)
            url = base.rstrip("/") + "/api/v4/file-urls/batch"
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
            payload = {
                "files": [{"name": file_name, "data_id": _safe_stem(file_name)}],
                "model_version": "vlm",
                "enable_table": True,
                "is_ocr": True,
            }

            r = requests.post(url, headers=headers, json=payload, timeout=(10, 30))
            if r.status_code != 200:
                logger.warning("MinerU precision request failed: %s %s", r.status_code, r.text[:300])
                return False
            j = r.json()
            if j.get("code") != 0:
                logger.warning("MinerU precision error: %s", j.get("msg"))
                return False

            batch_id = j.get("data", {}).get("batch_id")
            file_urls = j.get("data", {}).get("file_urls") or []
            if not batch_id or not file_urls:
                logger.warning("MinerU precision response missing batch_id or file_urls")
                return False

            upload_url = file_urls[0]
            with open(pdf_path, "rb") as fh:
                up = requests.put(upload_url, data=fh, timeout=(10, 600))
            if up.status_code not in (200, 201):
                logger.warning("Upload failed: HTTP %s", up.status_code)
                return False

            poll_url = base.rstrip("/") + f"/api/v4/extract-results/batch/{batch_id}"
            start = time.time()
            timeout = 600
            interval = 3
            while time.time() - start < timeout:
                resp = requests.get(poll_url, headers=headers, timeout=(10, 30))
                if resp.status_code != 200:
                    logger.warning("Polling error: %s", resp.status_code)
                    time.sleep(interval)
                    continue
                respj = resp.json()
                data = respj.get("data", {})
                extract_results = data.get("extract_result") or []
                for item in extract_results:
                    if item.get("state") == "done" and item.get("full_zip_url"):
                        zip_url = item.get("full_zip_url")
                        zd = requests.get(zip_url, timeout=(10, 600))
                        if zd.status_code == 200:
                            zbuf = BytesIO(zd.content)
                            with zipfile.ZipFile(zbuf) as zf:
                                candidates = [n for n in zf.namelist() if n.endswith("full.md") or n.endswith(".md")]
                                if not candidates:
                                    logger.warning("No markdown found inside zip")
                                    return False
                                mdname = candidates[0]
                                md = zf.read(mdname).decode("utf-8")
                                _atomic_write_text(md_output_path, md)
                                logger.info("OCR result written: %s", md_output_path)
                                return True

                state = data.get("state") or (extract_results[0].get("state") if extract_results else None)
                if state == "failed":
                    logger.warning("Extract failed: %s", data.get("err_msg"))
                    return False
                time.sleep(interval)

            logger.warning("Polling timed out for precision API")
            return False

        if api_key:
            try:
                ok = _try_precision_flow()
                if ok:
                    return True
                logger.info("Falling back to MinerU agent flow after precision failure for %s", file_name)
            except Exception:
                logger.exception("Precision flow raised exception, falling back to agent for %s", file_name)

        # Agent lightweight flow (no token)
        logger.debug("MinerU agent lightweight flow for %s", file_name)
        api_base = base.rstrip("/") + "/api/v1/agent"
        create_url = api_base + "/parse/file"
        payload = {"file_name": file_name, "language": "latin", "enable_table": True, "is_ocr": True, "enable_formula": True}
        r = requests.post(create_url, json=payload, timeout=(10, 30))
        if r.status_code != 200:
            logger.warning("MinerU agent create failed: %s %s", r.status_code, r.text[:300])
            return False
        j = r.json()
        if j.get("code") != 0:
            logger.warning("MinerU agent error: %s", j.get("msg"))
            return False

        task_id = j.get("data", {}).get("task_id")
        file_url = j.get("data", {}).get("file_url")
        if file_url:
            with open(pdf_path, "rb") as fh:
                up = requests.put(file_url, data=fh, timeout=(10, 600))
            if up.status_code not in (200, 201):
                logger.warning("Upload failed: HTTP %s", up.status_code)
                return False

        poll_url = api_base + f"/parse/{task_id}"
        start = time.time()
        timeout = 600
        interval = 3
        while time.time() - start < timeout:
            resp = requests.get(poll_url, timeout=(10, 30))
            if resp.status_code != 200:
                logger.warning("Polling error: %s", resp.status_code)
                time.sleep(interval)
                continue
            respj = resp.json()
            data = respj.get("data", {})
            state = data.get("state")
            if state == "done" and data.get("markdown_url"):
                md_resp = requests.get(data.get("markdown_url"), timeout=(10, 60))
                if md_resp.status_code == 200:
                    _atomic_write_text(md_output_path, md_resp.text)
                    logger.info("OCR result written: %s", md_output_path)
                    return True
                else:
                    logger.warning("Failed to download markdown: %s", md_resp.status_code)
                    return False
            if state == "failed":
                logger.warning("Extract failed: %s", data.get("err_msg"))
                return False
            time.sleep(interval)

        logger.warning("Polling timed out for agent lightweight API")
        # If remote services fail or time out, try a local text extraction fallback
        logger.info("Próbuję lokalnego ekstraktora tekstu z PDF (fallback)...")

        def _try_local_extraction() -> bool:
            # Try PyMuPDF (fitz) first
            try:
                import fitz  # PyMuPDF

                logger.debug("Using PyMuPDF for local extraction: %s", pdf_path)
                doc = fitz.open(pdf_path)
                parts = []
                for page in doc:
                    text = page.get_text("text")
                    if text:
                        parts.append(text)
                if parts:
                    md_text = "\n\n".join(parts)
                    header = f"# Extracted text from {file_name}\n\n"
                    _atomic_write_text(md_output_path, header + md_text)
                    logger.info("Local extraction (PyMuPDF) written: %s", md_output_path)
                    return True
            except ImportError:
                logger.debug("PyMuPDF not installed; skipping fitz extraction")
            except Exception:
                logger.exception("PyMuPDF extraction failed for %s", pdf_path)

            # Try pdfminer.six as a fallback
            try:
                from pdfminer.high_level import extract_text

                logger.debug("Using pdfminer.six for local extraction: %s", pdf_path)
                text = extract_text(pdf_path)
                if text and text.strip():
                    header = f"# Extracted text from {file_name}\n\n"
                    _atomic_write_text(md_output_path, header + text)
                    logger.info("Local extraction (pdfminer) written: %s", md_output_path)
                    return True
            except ImportError:
                logger.debug("pdfminer.six not installed; skipping pdfminer extraction")
            except Exception:
                logger.exception("pdfminer extraction failed for %s", pdf_path)

            # No local extractor available
            logger.warning(
                "Brak lokalnych bibliotek do ekstrakcji (PyMuPDF or pdfminer.six). "
                "Zainstaluj je: `pip install pymupdf` lub `pip install pdfminer.six` to enable local fallback."
            )
            return False

        try:
            return _try_local_extraction()
        except Exception:
            logger.exception("Local extraction fallback raised an unexpected error for %s", pdf_path)
            return False

    except Exception:
        logger.exception("Błąd MinerU OCR flow for %s", pdf_path)
        return False



def ocr_only():
    """Przeprowadź tylko OCR wszystkich PDFów w folderze pdf_dir (zapis do literature_ready_dir)."""
    pdf_dir = Path(CONFIG.get("pdf_dir"))
    ready_dir = Path(CONFIG.get("literature_ready_dir"))
    # ensure target dir exists
    ready_dir.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []
    if not pdfs:
        logger.warning("Brak plików PDF do przetworzenia.")
        return

    progress = ProgressBars(enabled=True)
    done = 0
    total = len(pdfs)
    progress.set_overall(current=0, total=total, label="OCR start")
    try:
        for pdf in pdfs:
            target = ready_dir / f"{_safe_stem(pdf.stem)}.md"
            if target.exists():
                logger.info("Już przetworzono: %s", target.name)
                done += 1
                progress.set_overall(current=done, total=total, label=pdf.name)
                continue
            logger.info("OCR: %s", pdf.name)
            try:
                out = ocr_pdf_with_mineru(str(pdf), str(target))
            except Exception as e:
                logger.exception("OCR error for %s: %s", pdf.name, e)
                done += 1
                progress.set_overall(current=done, total=total, label=pdf.name)
                continue
            if out:
                logger.info("Zapisano: %s", target)
            else:
                logger.warning("OCR nie powiódł się dla %s", pdf.name)
            done += 1
            progress.set_overall(current=done, total=total, label=pdf.name)
    except KeyboardInterrupt:
        progress.finish("Przerwano OCR.")
        logger.info("Przerwano OCR (Ctrl+C)")
        return
    progress.finish("OCR zakończone.")


enable_function_tracing(globals(), __name__, include_private=True)
