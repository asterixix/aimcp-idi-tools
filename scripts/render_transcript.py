"""Składa finalny transkrypt (format z Etapu 5 protokołu) z arkusza redakcji i metadanych.

Arkusz redakcji (Nagrania/ASR-lokalny/<nagranie>/redakcja.txt, poza git): jedna tura w linii,
  [HH:MM:SS - HH:MM:SS] B: treść ¶ kolejny akapit tej samej tury
czasy bezwzględne (od początku pliku audio); linie od '#' to komentarze.
Metadane: metadane/<ID>.json (bez danych osobowych). Kontrola PII: jeśli istnieje internal/<ID>-pii_POUFNE.txt
(jedna fraza w linii), żadna z fraz nie może wystąpić w wyniku; kody wewnętrzne (AIB…) są sprawdzane zawsze.
Publikacja pod kodem zewnętrznym z internal/kody_POUFNE.json (nowy wywiad dostaje losowy wolny kod):
public/<KOD_ZEWNĘTRZNY>-Transkrypt.md.
Użycie (z katalogu projektu):  .venv-asr/Scripts/python scripts/render_transcript.py AIBX-2
"""
import argparse, json, re, secrets, sys
from pathlib import Path

KODY = Path("internal/kody_POUFNE.json")


def external_code(internal_id):
    m = json.loads(KODY.read_text(encoding="utf-8"))
    if internal_id not in m["kody"]:
        used = {int(c.rsplit("-", 1)[1]) for c in m["kody"].values()}
        m["kody"][internal_id] = f"{m['prefiks']}-{secrets.choice([n for n in range(10, 100) if n not in used]):02d}"
        KODY.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    return m["kody"][internal_id]

PROJEKT = "Agenci AI i protokół MCP w repozytoriach bibliotek akademickich — mechanizmy przetwarzania metadanych i implikacje dla organizacji wiedzy"
TYP = "Częściowo ustrukturyzowany wywiad pogłębiony (IDI – Individual In-Depth Interview)"
PROWADZACY = "Prowadzący badanie naukowe - lic. Artur Sendyka (Uniwersytet Jagielloński)"
LICENCJA = "Creative Commons Uznanie autorstwa 4.0 Międzynarodowe (CC BY 4.0)"
METODA = ("Transkrypcja lokalnym modelem ASR (Whisper large-v3, faster-whisper) z diaryzacją mówców, poddana redakcji językowej "
          "wiernej treści nagrania (usunięcie wtrąceń, powtórzeń i fałszywych startów, korekta błędów fonetycznych ASR, standaryzacja "
          "terminologii dziedzinowej) oraz anonimizacji zgodnej z RODO, obejmującej także uogólnienie danych instytucjonalnych "
          "umożliwiających identyfikację rozmówcy.")
TURN = re.compile(r"^\[(\d+):(\d+):(\d+) - (\d+):(\d+):(\d+)\] (\w+): (.+)$")


def sec(h, m, s):
    return int(h) * 3600 + int(m) * 60 + int(s)


def mmss(t):
    t = max(0, int(t))
    return f"{t // 60:02d}:{t % 60:02d}"


def hms(t):
    return f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{int(t % 60):02d}"


def render_part(path, offset, labels):
    out, last = [], 0
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        m = TURN.match(line)
        if not m or m[7] not in labels:
            sys.exit(f"{path}:{n}: zła linia tury: {line[:80]}")
        a, b = sec(*m.groups()[0:3]) - offset, sec(*m.groups()[3:6]) - offset
        text = "  \n".join(p.strip() for p in m[8].split("¶"))
        out.append(f"**[{mmss(a)} - {mmss(b)}] {labels[m[7]]}:**  \n{text}\n")
        last = max(last, b)
    return out, last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("id")
    a = ap.parse_args()
    if a.id in json.loads(KODY.read_text(encoding="utf-8")).get("wykluczone", {}):
        sys.exit(f"STOP: {a.id} jest wykluczony z badania (internal/kody_POUFNE.json); nie publikujemy")
    meta = json.loads(Path(f"metadane/{a.id}.json").read_text(encoding="utf-8"))
    labels = meta.get("etykiety", {"B": "Badacz [B]", "R": "Respondent [R]"})
    body, durs = [], []
    for part in meta["czesci"]:
        turns, last = render_part(part["plik"], part["offset"], labels)
        if part.get("tytul"):
            body.append(f"## {part['tytul']}\n")
        body += turns
        durs.append(last)
    total = sum(durs)
    czas = f"ok. {round(total / 60)} minut ({hms(total)})" if len(durs) == 1 else \
        f"ok. {round(total / 60)} minut łącznie (" + " + ".join(f"część {i}: {hms(d)}" for i, d in enumerate(durs, 1)) + ")"
    ext = external_code(a.id)
    pola = [("Identyfikator transkrypcji", ext), ("Projekt badawczy", PROJEKT), ("Typ wywiadu", TYP),
            ("Data realizacji", meta["data"]), ("Czas trwania", czas + " – liczone od początku pierwszego pytania badawczego"),
            ("Język transkrypcji", "Polski (PL)"), ("Metoda transkrypcji", (meta.get("metoda") or METODA) + (" " + meta["metoda_dodatek"] if meta.get("metoda_dodatek") else ""))]
    pola += [tuple(p) for p in meta.get("dodatkowe_pola", [])]
    lines = ["# METADANE DANYCH BADAWCZYCH (RESEARCH DATA METADATA)", ""] + [f"- **{k}:** {v}" for k, v in pola]
    lines += ["- **Osoby uczestniczące:**", f"  - **Prowadzący wywiad [B / Badacz]:** {PROWADZACY}", f"  - **Informant [R / Respondent]:** {meta['informant']}",
              f"- **Licencja udostępnienia w repozytorium:** {LICENCJA}", "", "---", "", "# WYKAZ ZASTOSOWANYCH STANDARYZACJI I SKRÓTÓW", ""]
    lines += [f"{i}. {s}" for i, s in enumerate(meta["skroty"], 1)]
    lines += ["", "---", "", f"# ZAPIS ROZMOWY ({meta.get('zapis', 'TRANSKRYPCJA BADAWCZA')})", ""]
    text = "\n".join(lines + body)
    if re.search(r"\bAIB[A-Z]*-?\d", text):
        sys.exit("STOP: w wyniku jest kod wewnętrzny (AIB…); publikacja tylko pod kodem zewnętrznym")
    pii = Path(f"internal/{a.id}-pii_POUFNE.txt")
    if pii.exists():
        checked = "\n".join(body + [meta["informant"]] + meta["skroty"])  # afiliacja prowadzącego jest jawna z założenia
        hits = [p.strip() for p in pii.read_text(encoding="utf-8").splitlines()
                if p.strip() and not p.startswith("#") and re.search(r"\b" + re.escape(p.strip()), checked)]  # wielkość liter ma znaczenie, od początku słowa
        if hits:
            sys.exit(f"STOP: w wyniku zostały frazy z listy PII ({len(hits)}): sprawdź internal/{a.id}-pii_POUFNE.txt")
    Path(f"public/{ext}-Transkrypt.md").write_text(text, encoding="utf-8")
    print(f"{a.id} -> public/{ext}-Transkrypt.md: {sum(len(t) for t in body)} znaków, czas {czas}")


if __name__ == "__main__":
    main()
