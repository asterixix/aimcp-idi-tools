"""Diaryzacja mówców (pyannote, GPU) dla nagrań z Nagrania/Audio.

Wymaga jednorazowo: akceptacji warunków modelu pyannote/speaker-diarization-community-1 na Hugging Face
i zalogowania w terminalu: .venv-asr\\Scripts\\hf auth login
Wynik: Nagrania/ASR-lokalny/<nagranie>/diarization.json ({"segments": [[start, end, "S0"], ...]}); gotowe pliki pomijane.
Użycie (z katalogu projektu):  .venv-asr/Scripts/python scripts/asr_diarize.py [plik.mp3 ...]
"""
import argparse, json, time
from pathlib import Path

import torch

from asr_batch import AUDIO, OUT, hms, load

MODEL = "pyannote/speaker-diarization-community-1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--min-speakers", type=int, default=2), ap.add_argument("--max-speakers", type=int, default=3)
    a = ap.parse_args()
    from pyannote.audio import Pipeline
    pipe = Pipeline.from_pretrained(MODEL)
    pipe.to(torch.device("cuda"))
    for f in [Path(p) for p in a.files] or sorted(AUDIO.iterdir()):
        out = OUT / f.stem / "diarization.json"
        if out.exists():
            print(f"pomijam {f.name}", flush=True)
            continue
        t0 = time.time()
        audio = load(f)
        res = pipe({"waveform": torch.from_numpy(audio.copy()).unsqueeze(0), "sample_rate": 16000},
                   min_speakers=a.min_speakers, max_speakers=a.max_speakers)
        ann = getattr(res, "exclusive_speaker_diarization", None) or getattr(res, "speaker_diarization", res)  # pyannote 4.x / 3.x
        segs = [[round(t.start, 2), round(t.end, 2), f"S{spk.split('_')[-1].lstrip('0') or '0'}"] for t, _, spk in ann.itertracks(yield_label=True)]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"model": MODEL, "segments": segs}), encoding="utf-8")
        print(f"gotowe {f.name}: {hms(len(audio) / 16000)} audio, {time.time() - t0:.0f} s, {len({s[2] for s in segs})} mówców", flush=True)


if __name__ == "__main__":
    main()
