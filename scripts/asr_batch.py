"""Wsadowa transkrypcja lokalna (faster-whisper) nagrań z Nagrania/Audio.

Wynik: Nagrania/ASR-lokalny/<nagranie>/<model>.json (segmenty + słowa z czasem i pewnością) oraz <model>.txt ([HH:MM:SS] tekst).
Użycie (z katalogu projektu):  .venv-asr/Scripts/python scripts/asr_batch.py [plik.mp3 ...] [--model large-v3-turbo]
Bez plików: wszystkie z Nagrania/Audio. Gotowe wyniki są pomijane (wznawianie). Nagrania/ jest poza git (dane osobowe).
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

import numpy as np

# cuBLAS/cuDNN z pip (nvidia-*-cu12) muszą być w ścieżce DLL
for d in (Path(sys.prefix) / "Lib/site-packages/nvidia").glob("*/bin"):
    os.add_dll_directory(str(d))
    os.environ["PATH"] = str(d) + os.pathsep + os.environ["PATH"]

AUDIO, OUT = Path("Nagrania/Audio"), Path("Nagrania/ASR-lokalny")


def load(path):
    """Audio -> float32 16 kHz mono przez ffmpeg (PyAV w tym środowisku nie pasuje do faster-whisper)."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "16000", "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def hms(s):
    return f"{int(s // 3600):02d}:{int(s % 3600 // 60):02d}:{int(s % 60):02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--model", default="large-v3-turbo")
    a = ap.parse_args()
    files = [Path(f) for f in a.files] or sorted(AUDIO.iterdir())
    from faster_whisper import WhisperModel
    model = WhisperModel(a.model, device="cuda", compute_type="float16")
    for f in files:
        out = OUT / f.stem
        jf = out / f"{a.model}.json"
        if jf.exists():
            print(f"pomijam {f.name} (jest {jf.name})", flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        audio = load(f)
        dur = len(audio) / 16000
        segs, info = model.transcribe(audio, language="pl", beam_size=5, vad_filter=True, condition_on_previous_text=False, word_timestamps=True)
        res = []
        for s in segs:
            res.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip(), "avg_logprob": round(s.avg_logprob, 3),
                        "no_speech_prob": round(s.no_speech_prob, 3), "words": [[w.word.strip(), round(w.start, 2), round(w.end, 2), round(w.probability, 3)] for w in s.words or []]})
            if len(res) % 50 == 0:
                print(f"  {f.name}: {hms(s.end)} / {hms(dur)}", flush=True)
        (out / f"{a.model}.txt").write_text("\n".join(f"[{hms(s['start'])}] {s['text']}" for s in res), encoding="utf-8")
        jf.write_text(json.dumps({"audio": f.name, "model": a.model, "duration": round(dur, 1), "language_probability": round(info.language_probability, 3), "segments": res}, ensure_ascii=False), encoding="utf-8")  # JSON na końcu = znacznik ukończenia
        print(f"gotowe {f.name}: {hms(dur)} audio, {time.time() - t0:.0f} s, {len(res)} segmentów", flush=True)


if __name__ == "__main__":
    main()
