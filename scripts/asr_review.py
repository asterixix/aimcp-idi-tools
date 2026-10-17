"""Zestaw do odsłuchu: klipy audio okien, w których modele ASR się rozjeżdżają + arkusz z wersjami tekstu.

Okno = 60 s, w którym large-v3 i turbo różnią się >20% (min. 40 słów). Dla nagrań z transkryptem w public/ arkusz pokazuje też
pasujący fragment transkryptu (dopasowanie przybliżone). Wynik: Nagrania/ASR-lokalny/_do_odsluchu/ (ARKUSZ.md + klipy mp3), poza git.
Użycie (z katalogu projektu):  .venv-asr/Scripts/python scripts/asr_review.py [--thr 0.2] [--win 60] [--pad 3]
"""
import argparse, json, re, subprocess
from pathlib import Path

from rapidfuzz import fuzz

from asr_agree import agree, words, words_text
from asr_compare import compare, hms, turns

OUT, PUBLIC, AUDIO = Path("Nagrania/ASR-lokalny"), Path("public"), Path("Nagrania/Audio")
DEST = OUT / "_do_odsluchu"


def excerpt(rows, query, t0, t1):
    """Fragment transkryptu odpowiadający oknu: tury nakładające się na okno, długie przycięte do dopasowania z tekstem ASR."""
    out = []
    for k, r in enumerate(rows):
        end = rows[k + 1]["audio_time"] if k + 1 < len(rows) else float("inf")
        if r["audio_time"] < t1 and end > t0:
            text = r["text"]
            if len(text) > 1800:
                al = fuzz.partial_ratio_alignment(query.lower(), text.lower())
                text = "…" + text[max(al.dest_start - 150, 0):al.dest_end + 150] + "…"
            out.append(f"[{r['label']}] {text}")
    return "\n  ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=0.20), ap.add_argument("--win", type=int, default=60), ap.add_argument("--pad", type=int, default=3)
    x = ap.parse_args()
    DEST.mkdir(exist_ok=True)
    sheet, n_clips = [], 0
    for d in sorted(p for p in OUT.iterdir() if p.is_dir() and not p.name.startswith("_")):
        fa, fb = d / "large-v3.json", d / "large-v3-turbo.json"
        if not (fa.exists() and fb.exists()):
            continue
        (wa, dur), (wb, _) = words(fa), words(fb)
        err, n = agree(wa, wb, x.win, dur)
        wins = [(err[i] / n[i], i) for i in range(len(n)) if n[i] >= 40 and err[i] / n[i] > x.thr]
        if not wins:
            continue
        ta, tb = words_text(fa), words_text(fb)
        m, rows = re.match(r"^(AIB[A-Z]+-\d+)-nagranie(?:_(\d+))?$", d.name, re.I), []
        if m and (PUBLIC / f"{m[1]}-Transkrypt.md").exists():
            parts, p = turns(PUBLIC / f"{m[1]}-Transkrypt.md"), int(m[2] or 1)
            if p <= len(parts):
                rows = compare(json.loads(fa.read_text(encoding="utf-8")), parts[p - 1], 0)[0]
        audio = next(a for a in AUDIO.iterdir() if a.stem == d.name)
        sheet += [f"## {d.name}" + ("" if rows else " (brak transkryptu w public/)"), ""]
        for r, i in sorted(wins, key=lambda w: w[1]):
            t0, t1 = i * x.win, (i + 1) * x.win
            clip = DEST / f"{d.name}_{hms(t0).replace(':', '')}.mp3"
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(max(t0 - x.pad, 0)), "-t", str(x.win + 2 * x.pad), "-i", str(audio), "-ac", "1", "-b:a", "64k", str(clip)], check=True)
            n_clips += 1
            txt = lambda ws: " ".join(w for w, s in ws if t0 <= s < t1)
            sheet += [f"### [ ] {hms(t0)}-{hms(t1)}, rozbieżność modeli {r:.0%}", f"- klip: `{clip.name}` (z zapasem {x.pad} s)",
                      f"- large-v3: {txt(ta)}", f"- turbo: {txt(tb)}"]
            if rows:
                sheet.append(f"- transkrypt (public/, fragment przybliżony):\n  {excerpt(rows, txt(ta), t0, t1) or '(brak tury w oknie)'}")
            sheet += ["- werdykt (odsłuch): large-v3 / turbo / transkrypt / żadne; poprawny tekst: ______", ""]
    head = ["# Arkusz odsłuchu: okna z rozbieżnością modeli ASR", "",
            f"Okna {x.win} s, w których large-v3 i turbo różnią się >{x.thr:.0%}. Klipy mp3 w tym folderze. Rozbieżność = niepewność ASR, nie dowód błędu żadnej wersji.",
            "Zawiera treść wypowiedzi (dane osobowe): folder poza git, nie wysyłać poza zespół.", ""]
    (DEST / "ARKUSZ.md").write_text("\n".join(head + sheet), encoding="utf-8")
    print(f"{n_clips} klipów -> {DEST}")


if __name__ == "__main__":
    main()
