"""Ponowny odczyt fragmentów oznaczonych w arkuszach redakcji jako [..., niewyraźne] / [niewyraźne].

Dla każdego znacznika: czas ustalany przez dopasowanie słów poprzedzających znacznik do słów large-v3 z czasami,
klip [t-6 s, t+8 s] odczytywany trzema modelami (large-v3, large-v3-turbo, bardsai large-v2-pl w CT2) z podpowiedzią
(initial_prompt) z kontekstu zdania, bez VAD, beam 10. Wynik (POUFNE, poza git):
  Nagrania/ASR-lokalny/_do_odsluchu/niewyrazne/RAPORT.md + klipy mp3 do odsłuchu.
Użycie (z katalogu projektu):  .venv-asr/Scripts/python scripts/asr_fragmenty.py [nagranie ...]
"""
import json, re, subprocess, sys
from pathlib import Path

from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).parent))
from asr_batch import load  # noqa: E402  (ustawia też ścieżki DLL CUDA)

ASR, AUDIO = Path("Nagrania/ASR-lokalny"), Path("Nagrania/Audio")
OUT = ASR / "_do_odsluchu" / "niewyrazne"
MODELS = {"large-v3": "large-v3", "turbo": "large-v3-turbo", "bardsai-pl": ".asr-models/bardsai-large-v2-pl-v2"}
TURN = re.compile(r"^\[(\d+):(\d+):(\d+) - (\d+):(\d+):(\d+)\] (\w): (.+)$")
MARK = re.compile(r"\[([^\[\]]*?,\s*)?niewyraźne\]")
SR = 16000


def words_of(s):
    return re.findall(r"\w+", re.sub(r"\[[^\]]*\]", " ", s).lower())


def marks(sheet):
    """(nr linii, start, koniec, kontekst przed, znacznik, kontekst po) dla każdego znacznika w arkuszu."""
    out = []
    for n, line in enumerate(sheet.read_text(encoding="utf-8").splitlines(), 1):
        m = TURN.match(line)
        if not m:
            continue
        a = int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3])
        b = int(m[4]) * 3600 + int(m[5]) * 60 + int(m[6])
        text = m[8]
        for k in MARK.finditer(text):
            out.append((n, a, b, text[max(0, k.start() - 220):k.start()], k.group(0), text[k.end():k.end() + 120]))
    return out


def locate(words, a, b, before):
    """Czas znacznika: koniec najlepiej dopasowanego okna 6 słów large-v3 do słów przed znacznikiem."""
    ctx = " ".join(words_of(before)[-6:])
    cand = [w for w in words if a - 5 <= w[1] <= b + 5]
    if not ctx or len(cand) < 6:
        return a
    best = max(range(len(cand) - 5), key=lambda i: fuzz.ratio(ctx, " ".join(w[0].lower() for w in cand[i:i + 6])))
    return cand[best + 5][2]


def main():
    from faster_whisper import WhisperModel
    excluded = json.loads(Path("internal/kody_POUFNE.json").read_text(encoding="utf-8")).get("wykluczone", {})
    dirs = [ASR / d for d in sys.argv[1:]] or sorted(p.parent for p in ASR.glob("*/redakcja.txt"))
    dirs = [d for d in dirs if not any(d.name.startswith(x + "-") for x in excluded)]
    jobs = []
    for d in dirs:
        ms = marks(d / "redakcja.txt")
        if not ms:
            continue
        audio_file = next(AUDIO.glob(d.name + ".*"))
        words = [w for s in json.loads((d / "large-v3.json").read_text(encoding="utf-8"))["segments"] for w in s["words"]]
        for n, a, b, before, mark, after in ms:
            t = locate(words, a, b, before)
            jobs.append((d.name, audio_file, n, max(0.0, t - 6), t + 8, before, mark, after))
    OUT.mkdir(parents=True, exist_ok=True)
    reads = {i: {} for i in range(len(jobs))}
    cache = {}
    for name, path in MODELS.items():
        model = WhisperModel(path, device="cuda", compute_type="float16")
        for i, (rec, af, n, t0, t1, before, mark, after) in enumerate(jobs):
            if af not in cache:
                cache[af] = load(af)
            clip = cache[af][int(t0 * SR):int(t1 * SR)]
            prompt = " ".join(re.sub(r"\[[^\]]*\]", "", before).split()[-25:])
            segs, _ = model.transcribe(clip, language="pl", beam_size=10, vad_filter=False, condition_on_previous_text=False, initial_prompt=prompt)
            reads[i][name] = " ".join(s.text.strip() for s in segs)
        del model
    rep = ["# Ponowny odczyt fragmentów niewyraźnych (POUFNE)", ""]
    for i, (rec, af, n, t0, t1, before, mark, after) in enumerate(jobs):
        clip = OUT / f"{rec}_l{n}_{int(t0)}.mp3"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t0:.1f}", "-t", f"{t1 - t0:.1f}", "-i", str(af), "-ac", "1", "-b:a", "96k", str(clip)], check=True)
        rep += [f"## {rec}, linia {n}, klip {clip.name} ({int(t0) // 60}:{int(t0) % 60:02d})", f"- arkusz: …{before[-120:]} **{mark}** {after[:80]}…"]
        rep += [f"- {k}: {v}" for k, v in reads[i].items()] + [""]
    (OUT / "RAPORT.md").write_text("\n".join(rep), encoding="utf-8")
    print(f"{len(jobs)} znaczników -> {OUT / 'RAPORT.md'}")


if __name__ == "__main__":
    # samokontrola wzorca: łapie oba typy znaczników, pomija zwykłe uogólnienia
    assert MARK.findall("są [dokooptowywane, niewyraźne] i [niewyraźne], ale [uczelnia]") == ["dokooptowywane, ", ""]
    main()
