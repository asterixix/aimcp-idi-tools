"""Roboczy zrzut tur do redakcji: słowa lokalnego ASR (large-v3) + mówcy.

Źródło mówców (pierwsze dostępne): Nagrania/ASR-lokalny/<nagranie>/diarization.json (pyannote, scripts/asr_diarize.py)
albo surowy ASR źródłowy z Nagrania/Transkrypt surowy/ (etykiety mówców dopasowane po czasie).
Wynik: Nagrania/ASR-lokalny/<nagranie>/tury.txt  ([HH:MM:SS - HH:MM:SS] S<n>: tekst, słowa o pewności <0,4 jako ⟪słowo|p⟫).
Czasy są bezwzględne (od początku pliku audio); punkt zerowy ustala się przy redakcji.
Użycie (z katalogu projektu):  .venv-asr/Scripts/python scripts/build_draft.py AIBX-2-Nagranie [--model large-v3]
"""
import argparse, bisect, json, re
from collections import Counter
from pathlib import Path

from build_turns import load_recordings

OUT, RAW = Path("Nagrania/ASR-lokalny"), Path("Nagrania/Transkrypt surowy")


def hms(s):
    return f"{int(s // 3600):02d}:{int(s % 3600 // 60):02d}:{int(s % 60):02d}"


def speakers_from_diarization(path):
    segs = json.loads(path.read_text(encoding="utf-8"))["segments"]  # [start, end, speaker]
    starts = [s[0] for s in segs]

    def who(t):
        i = bisect.bisect_right(starts, t) - 1
        if i >= 0 and segs[i][0] <= t <= segs[i][1]:
            return segs[i][2]
        near = [segs[j] for j in (i, i + 1) if 0 <= j < len(segs)]  # poza segmentem: najbliższy
        return min(near, key=lambda s: min(abs(s[0] - t), abs(s[1] - t)))[2] if near else "S?"
    return who, "pyannote"


def speakers_from_raw(name):
    m = re.match(r"^(AIB[A-Z]+-\d+)-nagranie(?:_(\d+))?$", name, re.I)
    f = RAW / f"{m[1]}-Transkrypt-surowy.txt" if m else None
    if not f or not f.exists():
        return None, None
    recs = load_recordings(f)
    entries = recs[int(m[2] or 1) - 1]["diarization"]["entries"]
    times = [float(e["start_time"]) for e in entries]

    def who(t):
        i = bisect.bisect_left(times, t)
        best = min((j for j in (i - 1, i) if 0 <= j < len(times)), key=lambda j: abs(times[j] - t))
        return f"S{entries[best]['speaker']}"
    return who, f"surowy ASR źródłowy ({f.name})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording"), ap.add_argument("--model", default="large-v3")
    a = ap.parse_args()
    d = OUT / a.recording
    asr = json.loads((d / f"{a.model}.json").read_text(encoding="utf-8"))
    who, src = speakers_from_diarization(d / "diarization.json") if (d / "diarization.json").exists() else speakers_from_raw(a.recording)
    if who is None:
        raise SystemExit("Brak źródła mówców: uruchom scripts/asr_diarize.py (pyannote) dla tego nagrania.")
    turns, cur = [], None
    for seg in asr["segments"]:
        for w, s, e, p in seg["words"]:
            spk = who((s + e) / 2)
            tok = w if p >= 0.4 else f"⟪{w}|{p:.2f}⟫"
            if cur and cur["spk"] == spk:
                cur["words"].append(tok)
                cur["end"] = e
            else:
                cur = {"spk": spk, "start": s, "end": e, "words": [tok]}
                turns.append(cur)
    words = Counter()
    for t in turns:
        words[t["spk"]] += len(t["words"])
    head = [f"# {a.recording}: tury robocze ({a.model}; mówcy: {src})", f"# słów wg mówcy: {dict(words)}; tur: {len(turns)}", ""]
    body = [f"[{hms(t['start'])} - {hms(t['end'])}] {t['spk']}: {' '.join(t['words'])}" for t in turns]
    (d / "tury.txt").write_text("\n".join(head + body), encoding="utf-8")
    print(f"{a.recording}: {len(turns)} tur, mówcy {dict(words)} ({src})")


if __name__ == "__main__":
    main()
