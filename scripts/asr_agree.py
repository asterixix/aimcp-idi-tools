"""Zgodność dwóch modeli ASR na tym samym nagraniu: okna, w których się rozjeżdżają, to miejsca do odsłuchu.

Nie wymaga transkryptu (działa też dla nagrań bez opracowanego tekstu).
Użycie (z katalogu projektu):  .venv-asr/Scripts/python scripts/asr_agree.py [nagranie ...] [--a large-v3] [--b large-v3-turbo] [--win 60] [--top 25]
Wynik: Nagrania/ASR-lokalny/<nagranie>/zgodnosc-modeli.md (treść wypowiedzi, więc tylko w Nagrania/, poza git).
"""
import argparse, json
from pathlib import Path

from rapidfuzz.distance import Levenshtein

from asr_compare import hms, toks

OUT = Path("Nagrania/ASR-lokalny")


def words(path):
    d = json.loads(path.read_text(encoding="utf-8"))
    return [(t, s) for seg in d["segments"] for w, s, _, _ in seg["words"] for t in toks(w)], d["duration"]


def agree(a, b, win, dur):
    ops = Levenshtein.opcodes([t for t, _ in a], [t for t, _ in b])
    nwin = int(dur // win) + 1
    err, n = [0] * nwin, [0] * nwin
    for _, s in a:
        n[min(int(s // win), nwin - 1)] += 1
    for op in ops:
        if op.tag != "equal":
            i = min(int(a[min(op.src_start, len(a) - 1)][1] // win), nwin - 1)
            err[i] += max(op.src_end - op.src_start, op.dest_end - op.dest_start)
    return err, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recordings", nargs="*")
    ap.add_argument("--a", default="large-v3"), ap.add_argument("--b", default="large-v3-turbo")
    ap.add_argument("--win", type=int, default=60), ap.add_argument("--top", type=int, default=25)
    x = ap.parse_args()
    dirs = [OUT / r for r in x.recordings] or sorted(d for d in OUT.iterdir() if d.is_dir() and d.name != "_bench")
    print(f"{'nagranie':24s} {'słów A':>7s} {'słów B':>7s} {'rozbieżność':>11s} {'okien >25%':>10s}")
    for d in dirs:
        fa, fb = d / f"{x.a}.json", d / f"{x.b}.json"
        if not (fa.exists() and fb.exists()):
            continue
        (a, dur), (b, _) = words(fa), words(fb)
        err, n = agree(a, b, x.win, dur)
        rate = sum(err) / max(len(a), 1)
        wins = [(err[i] / n[i], i) for i in range(len(n)) if n[i] >= 40]
        bad = [w for w in wins if w[0] > 0.25]
        print(f"{d.name:24s} {len(a):7d} {len(b):7d} {rate:10.1%} {len(bad):10d}")
        md = [f"# {d.name}: zgodność `{x.a}` (A) i `{x.b}` (B)", "",
              f"Słów A: {len(a)}, B: {len(b)}. Rozbieżność (Levenshtein/słowa A): {rate:.1%}. Okien {x.win} s o rozbieżności >25% (min. 40 słów): {len(bad)} z {len(wins)}.",
              "Rozbieżność modeli = miejsca niepewne dla ASR; nie mówi, który ma rację. Odsłuchaj.", "", f"## {x.top} okien o największej rozbieżności", ""]
        ta, tb = words_text(fa), words_text(fb)
        for r, i in sorted(wins, reverse=True)[:x.top]:
            t0, t1 = i * x.win, (i + 1) * x.win
            md += [f"### {hms(t0)}-{hms(t1)}: {r:.0%}", f"- A: {' '.join(w for w, s in ta if t0 <= s < t1)[:500]}",
                   f"- B: {' '.join(w for w, s in tb if t0 <= s < t1)[:500]}", ""]
        (d / "zgodnosc-modeli.md").write_text("\n".join(md), encoding="utf-8")


def words_text(path):
    d = json.loads(path.read_text(encoding="utf-8"))
    return [(w, s) for seg in d["segments"] for w, s, _, _ in seg["words"]]


if __name__ == "__main__":
    main()
