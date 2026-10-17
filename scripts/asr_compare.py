"""Porównanie lokalnego ASR z transkryptem z public/: które tury rozjeżdżają się z nagraniem.

Dopasowanie globalne słów (Levenshtein na całej rozmowie), więc nie trzeba znać offsetu preambuły.
Użycie (z katalogu projektu):  .venv-asr/Scripts/python scripts/asr_compare.py [AIBX-4 ...] [--model large-v3-turbo] [--top 30]
Bez ID: wszystkie nagrania, które mają i wynik ASR, i transkrypt w public/.
Wynik: Nagrania/ASR-lokalny/<nagranie>/porownanie-<model>.md (zawiera treść wypowiedzi, więc zostaje w Nagrania/, poza git).
"""
import argparse, json, re
from pathlib import Path

from rapidfuzz.distance import Levenshtein

OUT, PUBLIC = Path("Nagrania/ASR-lokalny"), Path("public")
FILLERS = {"yyy", "eee", "yy", "ee", "mhm", "hm", "aha"}  # transkrypty to clean verbatim: ASR-owe wtrącenia nie są błędem
HEAD = re.compile(r"^\*\*\[(\d+):(\d+) - (\d+):(\d+)\]\s*(.*?):\*\*[ \t]*\n+(.+?)(?=\n\*\*\[|\n#|\Z)", re.M | re.S)


def toks(s):
    s = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", s.lower())
    return [t for t in re.sub(r"[^\w\s]", " ", s).split() if t not in FILLERS]


def turns(md):
    """Tury transkryptu pogrupowane w części (znaczniki czasu zaczynają się od nowa = kolejne nagranie w pliku)."""
    parts, prev = [[]], 0
    for m in HEAD.finditer(Path(md).read_text(encoding="utf-8")):
        start, end = int(m[1]) * 60 + int(m[2]), int(m[3]) * 60 + int(m[4])
        if start < prev - 120:
            parts.append([])
        prev = end
        parts[-1].append({"start": start, "label": m[5].strip(), "text": m[6].strip()})
    return parts


def compare(asr, part, top):
    a_tok, a_time = [], []
    for seg in asr["segments"]:
        for w, s, _, _ in seg["words"]:
            for t in toks(w):
                a_tok.append(t), a_time.append(s)
    r_tok, r_turn = [], []
    for i, t in enumerate(part):
        tt = toks(t["text"])
        r_tok += tt
        r_turn += [i] * len(tt)
    ops = Levenshtein.opcodes(a_tok, r_tok)
    eq = sum(op.src_end - op.src_start for op in ops if op.tag == "equal")
    err = [0] * len(part)
    first_asr = [None] * len(part)
    for op in ops:
        n_a, n_r = op.src_end - op.src_start, op.dest_end - op.dest_start
        if op.tag == "equal":
            for k in range(n_r):
                i = r_turn[op.dest_start + k]
                if first_asr[i] is None:
                    first_asr[i] = op.src_start + k
            continue
        if (op.src_start == 0 and op.tag == "delete") or (op.dest_start >= len(r_tok) and op.tag == "delete"):
            continue  # preambuła przed pierwszą turą / dopisek po ostatniej
        i = r_turn[min(op.dest_start, len(r_tok) - 1)]
        err[i] += max(n_a, n_r)
        if first_asr[i] is None:
            first_asr[i] = min(op.src_start, len(a_time) - 1)
    counts = [r_turn.count(i) for i in range(len(part))]
    rows = []
    for i, t in enumerate(part):
        at = a_time[first_asr[i]] if first_asr[i] is not None and a_time else 0
        rows.append({"i": i, "turn_time": t["start"], "audio_time": at, "n": counts[i], "err": err[i], "rate": err[i] / max(counts[i], 1), "label": t["label"], "text": t["text"]})
    # tekst ASR w oknie czasowym tury (do podglądu)
    words = [(s, w) for seg in asr["segments"] for w, s, _, _ in seg["words"]]
    for k, r in enumerate(rows):
        end = rows[k + 1]["audio_time"] if k + 1 < len(rows) else float("inf")
        r["asr"] = " ".join(w for s, w in words if r["audio_time"] <= s < end)
    total_err, total_n = sum(err), max(len(r_tok), 1)
    return rows, total_err / total_n, total_n, (len(a_tok), eq / max(len(a_tok), 1), eq / total_n)


def role_rates(rows):
    """Rozbieżność osobno dla tur badacza i rozmówców (pytania bywają redagowane bardziej niż odpowiedzi)."""
    out = []
    for is_b in (True, False):
        sel = [r for r in rows if r["label"].startswith("Badacz") == is_b]
        out.append(sum(r["err"] for r in sel) / max(sum(r["n"] for r in sel), 1))
    return out


def hms(s):
    return f"{int(s // 3600):02d}:{int(s % 3600 // 60):02d}:{int(s % 60):02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--top", type=int, default=30)
    a = ap.parse_args()
    ids = a.ids or sorted(p.name.split("-Transkrypt")[0] for p in PUBLIC.glob("*-Transkrypt.md"))
    print(f"{'ID':10s} {'część':5s} {'słów tr.':>8s} {'słów ASR':>8s} {'nagranie w tr.':>14s} {'tr. w nagr.':>11s} {'rozbież.':>8s} {'badacz':>7s} {'rozmówca':>8s} {'tur >40%':>8s}")
    for id_ in ids:
        parts = turns(PUBLIC / f"{id_}-Transkrypt.md")
        files = sorted(d for d in OUT.glob(f"{id_}-*") if (d / f"{a.model}.json").exists())
        if not files:
            print(f"{id_:10s} brak wyniku ASR ({a.model})")
            continue
        byp = {int(m[1]) if (m := re.search(r"_(\d+)$", d.name)) else 1: d for d in files}  # sufiks _N = część N
        for p, part in enumerate(parts, 1):
            d = byp.get(p)
            if d is None:
                print(f"{id_:10s} {p:<5d} brak wyniku ASR dla części")
                continue
            rows, rate, n, (n_asr, cover, confirm) = compare(json.loads((d / f"{a.model}.json").read_text(encoding="utf-8")), part, a.top)
            bad = [r for r in rows if r["n"] >= 15 and r["rate"] > 0.4]
            rb, rr = role_rates(rows)
            print(f"{id_:10s} {p:<5d} {n:8d} {n_asr:8d} {cover:14.0%} {confirm:11.0%} {rate:8.0%} {rb:7.0%} {rr:8.0%} {len(bad):8d}")
            md = [f"# {id_}, część {p}: ASR `{a.model}` vs `public/{id_}-Transkrypt.md`", "",
                  f"Słów w transkrypcie: {n}, w ASR: {n_asr}. Słów nagrania znalezionych w transkrypcie: {cover:.0%}; słów transkryptu potwierdzonych przez nagranie: {confirm:.0%}. Rozbieżność (Levenshtein/słowa wzorca): {rate:.1%} (badacz {rb:.1%}, rozmówca {rr:.1%}). Tur >40% (min. 15 słów): {len(bad)} z {len(rows)}.",
                  "Czas audio = czas pierwszego dopasowanego słowa ASR w turze. Rozbieżność nie rozstrzyga, kto ma rację: odsłuchaj.", "",
                  f"## {a.top} najbardziej rozbieżnych tur (min. 15 słów)", ""]
            for r in sorted((r for r in rows if r["n"] >= 15), key=lambda r: -r["rate"])[:a.top]:
                md += [f"### audio {hms(r['audio_time'])} (transkrypt {hms(r['turn_time'])}), {r['label']}: {r['rate']:.0%} ({r['err']}/{r['n']})",
                       f"- transkrypt: {r['text'][:400]}", f"- ASR: {r['asr'][:400]}", ""]
            (d / f"porownanie-{a.model}.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
