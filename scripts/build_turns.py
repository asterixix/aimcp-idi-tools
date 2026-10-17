#!/usr/bin/env python3
"""
build_turns.py — uniwersalny parser surowych transkryptów ASR (seria AIBX/AIBX)

Obsługuje WSZYSTKIE warianty formatu napotkane dotąd w projekcie:
  (a) segment ze spacją wiodącą, tekst = ''.join(segments)          [AIBX-1,2,5,6,7]
  (b) segment bez spacji wiodącej, tekst = ' '.join(segments)        [AIBX-3]
  (c) segment surowy (małe litery, bez interpunkcji); poprawna
      interpunkcja jest tylko w polu "text" — wymaga mapowania
      1:1 po normalizacji (lowercase + usunięcie interpunkcji)       [AIBX-4]
  (d) plik tekstowy z nagłówkami "Nagranie N:" poprzedzającymi
      kolejne, odrębne obiekty JSON (wiele nagrań w jednym pliku)    [AIBX-5]

Użycie:
    python3 build_turns.py <plik_surowy.txt> [--offset SEKUNDY] [--out PLIK_WYJSCIOWY]

Jeśli --offset nie podano, skrypt wypisze pierwsze ~40 tokenów każdego
nagrania, żeby ręcznie ustalić, gdzie kończy się zapowiedź/preambuła
(sprawdzenie mikrofonu, kod identyfikacyjny, wyjaśnienie procedury) —
zgodnie z protokołem transkrypt właściwy zaczyna się od [00:00] w miejscu
PIERWSZEGO PYTANIA BADAWCZEGO, nie od początku pliku źródłowego.

Wyjście: plik tekstowy z turami wypowiedzi, znacznikami czasu
znormalizowanymi względem offsetu, oraz oznaczeniem tokenów o niskiej
pewności ASR (confidence < 0.4) w formie ⟪słowo|pewność⟫ — do dalszej,
ręcznej redakcji (post-editing, anonimizacja) zgodnie z protokołem
(patrz 00-PROTOKOL-opracowania-transkryptow.md, Etapy 0-2).

Ten skrypt NIE wykonuje post-editingu ani anonimizacji — to wymaga
lektury i redakcyjnej oceny (przez Claude Code / człowieka), zgodnie
z resztą protokołu. Skrypt tylko rekonstruuje strukturę i oszczędza
pracę techniczną (parsowanie JSON, wykrywanie formatu, obliczanie
znaczników czasu).
"""

import json
import re
import sys
import argparse
from pathlib import Path


def load_recordings(path: Path):
    """Zwraca listę obiektów danych (jeden na nagranie) z pliku źródłowego.
    Obsługuje zarówno pojedynczy obiekt JSON, jak i plik z nagłówkami
    'Nagranie N:' poprzedzającymi kolejne obiekty JSON."""
    raw = path.read_text(encoding="utf-8")
    if re.match(r"^\s*Nagranie\s+\d+\s*:", raw):
        parts = re.split(r"Nagranie\s+\d+\s*:", raw)
        parts = [p.strip() for p in parts if p.strip()]
        return [json.loads(p) for p in parts]
    else:
        return [json.loads(raw)]


def reconstruct(entries, text):
    """Próbuje trzech metod rekonstrukcji tekstu ze 'entries'; zwraca
    (metoda, lista_slow_do_uzycia) gdzie lista_slow_do_uzycia to lista
    par (poprawny_tekst_slowa, entry_zrodlowy) do budowy tur."""
    concat = "".join(e["segment"] for e in entries)
    if concat.strip() == text.strip():
        return "concat", [(e["segment"], e) for e in entries]

    joined = " ".join(e["segment"] for e in entries)
    if " ".join(joined.split()) == " ".join(text.split()):
        return "joined", [(e["segment"], e) for e in entries]

    # metoda (c): mapowanie 1:1 po normalizacji
    def norm(s):
        s = s.lower()
        s = re.sub(r"[^\w\s]", "", s, flags=re.UNICODE)
        return " ".join(s.split())

    text_words = text.split()
    if len(text_words) == len(entries) and norm(" ".join(text_words)) == norm(
        " ".join(e["segment"] for e in entries)
    ):
        return "mapped", [(w, e) for w, e in zip(text_words, entries)]

    raise ValueError(
        "Żadna z trzech znanych metod rekonstrukcji nie pasuje — "
        "format pliku różni się od wszystkich dotąd napotkanych wariantów. "
        "Wymaga ręcznej analizy (patrz Etap 0 protokołu)."
    )


def build_turns(words_with_entries, offset):
    """words_with_entries: lista (tekst_slowa, entry). Grupuje w tury wg mówcy."""
    filtered = [
        (w, e) for w, e in words_with_entries if float(e["start_time"]) >= offset
    ]
    turns = []
    cur_speaker, cur_words = None, []
    for w, e in filtered:
        if e["speaker"] != cur_speaker:
            if cur_words:
                turns.append({"speaker": cur_speaker, "words": cur_words})
            cur_speaker, cur_words = e["speaker"], []
        cur_words.append((w, e))
    if cur_words:
        turns.append({"speaker": cur_speaker, "words": cur_words})
    return turns


def fmt_time(s, offset):
    s = max(0, s - offset)
    m, sec = int(s // 60), int(round(s % 60))
    if sec == 60:
        sec, m = 0, m + 1
    return f"{m:02d}:{sec:02d}"


def dump_turns(turns, offset, out_path, header="", join_with_space=False):
    sep = " " if join_with_space else ""
    with open(out_path, "w", encoding="utf-8") as out:
        out.write(f"{header}\nOFFSET: {offset}s\nLiczba tur: {len(turns)}\n")
        out.write("=" * 100 + "\n\n")
        for i, t in enumerate(turns):
            start = float(t["words"][0][1]["start_time"])
            end = float(t["words"][-1][1]["end_time"])
            text = sep.join(w.strip() if join_with_space else w for w, e in t["words"]).strip()
            marked = []
            for w, e in t["words"]:
                conf = e.get("confidence", 1.0)
                token = w.strip() if join_with_space else w
                if conf < 0.4:
                    marked.append(f"⟪{token.strip()}|{conf:.2f}⟫")
                else:
                    marked.append(token)
            marked_text = sep.join(marked).strip()
            out.write(
                f"--- TURA {i} | mowca={t['speaker']} | "
                f"[{fmt_time(start, offset)}-{fmt_time(end, offset)}] | "
                f"(surowy {start:.1f}-{end:.1f}s) | n_slow={len(t['words'])}\n"
            )
            out.write(f"TEKST: {text}\n")
            if marked_text != text:
                out.write(f"NISKA-PEWNOSC: {marked_text}\n")
            out.write("\n")
    print(f"Zapisano: {out_path} ({len(turns)} tur)")


def preview_start(entries, n=40):
    """Pomaga recznie ustalic offset — wypisuje pierwsze n tokenow z czasem."""
    print("Podgląd pierwszych tokenów (do ustalenia offsetu końca preambuły):")
    for e in entries[:n]:
        print(f"  {e['start_time']:>10}  spk={e['speaker']}  {e['segment']!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plik", type=Path, help="surowy plik ASR (JSON lub tekst z 'Nagranie N:')")
    ap.add_argument("--offset", type=str, default=None,
                     help="znacznik(i) czasu [s] początku pierwszego pytania badawczego. "
                          "Dla plików z wieloma nagraniami podać po przecinku, jedną wartość "
                          "na nagranie, np. '6.424,0.0' — albo jedną wartość, jeśli ta sama "
                          "pasuje do wszystkich nagrań (często 0.0 dla nagrań uzupełniających bez preambuły).")
    ap.add_argument("--out", type=Path, default=None, help="plik wyjściowy (domyślnie: <nazwa>_turns.txt)")
    ap.add_argument("--preview", action="store_true", help="tylko pokaż pierwsze tokeny, nie buduj tur")
    args = ap.parse_args()

    recordings = load_recordings(args.plik)
    print(f"Znaleziono {len(recordings)} nagranie/nagrania w pliku.")

    offsets = None
    if args.offset is not None:
        offsets = [float(x) for x in args.offset.split(",")]
        if len(offsets) == 1:
            offsets = offsets * len(recordings)
        elif len(offsets) != len(recordings):
            print(f"⚠️  Podano {len(offsets)} offsetów, ale plik zawiera {len(recordings)} nagrania/ń — "
                  "podaj dokładnie tyle offsetów, ile jest nagrań (po przecinku), albo jeden wspólny.")
            sys.exit(1)

    for idx, data in enumerate(recordings, 1):
        entries = data["diarization"]["entries"]
        text = data["text"]
        method, words_with_entries = reconstruct(entries, text)
        print(f"[Nagranie {idx}] metoda rekonstrukcji: {method} | tokenów: {len(entries)} | "
              f"mówcy: {sorted(set(e['speaker'] for e in entries))}")

        offset = offsets[idx - 1] if offsets else None
        if args.preview or offset is None:
            preview_start(entries)
            if offset is None:
                print(f"\n⚠️  [Nagranie {idx}] Podaj --offset i uruchom ponownie, "
                      "żeby zbudować właściwy zrzut tur.")
                continue

        turns = build_turns(words_with_entries, offset)
        if args.out:
            out_path = args.out if len(recordings) == 1 else args.out.with_name(
                f"{args.out.stem}_nagranie{idx}{args.out.suffix}"
            )
        else:
            out_path = args.plik.with_name(f"{args.plik.stem}_nagranie{idx}_turns.txt")
        dump_turns(turns, offset, out_path,
                   header=f"Źródło: {args.plik.name} — Nagranie {idx}/{len(recordings)}",
                   join_with_space=(method in ("joined", "mapped")))


if __name__ == "__main__":
    main()
