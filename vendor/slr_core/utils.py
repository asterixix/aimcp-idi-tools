import os
import time
import json
import logging
import inspect
import itertools
from pathlib import Path

logger = logging.getLogger("slr_auto_coder")
_TRACE_COUNTER = itertools.count(1)


def _preview(value, limit: int = 240) -> str:
    try:
        text = repr(value)
    except Exception:
        text = f"<unrepr:{type(value).__name__}>"
    if len(text) > limit:
        return text[:limit] + "...<truncated>"
    return text


def trace(func):
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        call_id = next(_TRACE_COUNTER)
        args_preview = ", ".join(_preview(a, 120) for a in args[:3])
        kwargs_preview = ", ".join(f"{k}={_preview(v, 120)}" for k, v in list(kwargs.items())[:3])
        if args_preview and kwargs_preview:
            combined = f"{args_preview}, {kwargs_preview}"
        else:
            combined = args_preview or kwargs_preview
        logger.debug("TRACE[%s] ENTER %s(%s)", call_id, func.__name__, combined)
        start = time.time()
        try:
            res = func(*args, **kwargs)
            elapsed = time.time() - start
            logger.debug("TRACE[%s] EXIT %s -> %s (elapsed=%.3fs)", call_id, func.__name__, _preview(res, 160), elapsed)
            return res
        except Exception:
            logger.exception("TRACE[%s] EXCEPTION in %s", call_id, func.__name__)
            raise

    wrapper._slr_traced = True  # type: ignore[attr-defined]
    return wrapper


def _atomic_write_text(path: str, text: str):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f"{target.name}.{os.getpid()}.{time.time_ns()}.tmp")
    tmp_path.write_text(text, encoding="utf-8")

    last_error = None
    for _ in range(3):
        try:
            os.replace(tmp_path, target)
            return
        except PermissionError as error:
            last_error = error
            time.sleep(0.1)

    try:
        target.write_text(text, encoding="utf-8")
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        return
    except PermissionError:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        if last_error is not None:
            raise last_error
        raise


def _atomic_write_json(path: str, data):
    _atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2))


def _append_text(path: str, text: str):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf-8") as f:
        f.write(text)


def _safe_stem(name: str) -> str:
    import re

    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return safe or "document"


def enable_function_tracing(namespace: dict, module_name: str, include_private: bool = False):
    """Wrap module-local functions with trace() for full execution tracing."""
    for key, value in list(namespace.items()):
        if not inspect.isfunction(value):
            continue
        if getattr(value, "__module__", None) != module_name:
            continue
        if key.startswith("_") and not include_private:
            continue
        if getattr(value, "_slr_traced", False):
            continue
        namespace[key] = trace(value)
