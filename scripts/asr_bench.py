"""Test jakości lokalnych modeli ASR (PL) na fragmencie nagrania względem transkryptu.

Nowy klip (wzorzec = tury z public/ w zakresie czasu transkryptu, offset = koniec preambuły w audio):
  .venv-asr/Scripts/python scripts/asr_bench.py Nagrania/Audio/AIBX-4-Nagranie.MP3 public/AIBX-4-Transkrypt.md --offset 6.84 --ref-start 1200 --ref-end 1560
Ponowny test innych modeli na istniejącym klipie:
  .venv-asr/Scripts/python scripts/asr_bench.py --clip Nagrania/ASR-lokalny/_bench/<klip> --models cohere-transcribe
Wyniki: <klip>/results.json (dopisywane), <klip>/<model>.txt. Nagrania/ jest poza git.
"""
import argparse, json, os, re, subprocess, sys, time
from pathlib import Path

import jiwer
import soundfile as sf

# cuBLAS/cuDNN z pip (nvidia-*-cu12) muszą być w ścieżce DLL
for d in (Path(sys.prefix) / "Lib/site-packages/nvidia").glob("*/bin"):
    os.add_dll_directory(str(d))
    os.environ["PATH"] = str(d) + os.pathsep + os.environ["PATH"]

# nazwa: (rodzaj, id modelu, dodatkowe argumenty generate dla HF Whisper)
MODELS = {
    "whisper-large-v3": ("fw", "large-v3", {}),
    "whisper-large-v3-turbo": ("fw", "large-v3-turbo", {}),
    "parakeet-tdt-0.6b-v3": ("onnx", "nemo-parakeet-tdt-0.6b-v3", {}),
    "cohere-transcribe": ("cohere", "CohereLabs/cohere-transcribe-03-2026", {}),
    "whisper-large-v3-hf-control": ("hfw", "openai/whisper-large-v3", {"language": "pl", "task": "transcribe"}),  # kontrola: ten sam potok co modele HF
    "bardsai-ct2": ("fw", ".asr-models/bardsai-large-v2-pl-v2", {}),  # CTranslate2, ten sam potok co baseline
    "buzzasr-ct2": ("fw", ".asr-models/buzzasr-polish", {}),
    "buzzasr-polish-rec": ("hfw", "BuzzASR/polish", {"num_beams": 1, "no_repeat_ngram_size": 3, "repetition_penalty": 1.2}),  # ustawienia zalecane w karcie modelu
    "whisper-large-v3-hf-control-rec": ("hfw", "openai/whisper-large-v3", {"language": "pl", "task": "transcribe", **{"num_beams": 1, "no_repeat_ngram_size": 3, "repetition_penalty": 1.2}}),
    "buzzasr-polish": ("hfw", "BuzzASR/polish", {}),  # język wpisany w generation_config
    "bardsai-whisper-large-v2-pl-v2": ("hfw", "bardsai/whisper-large-v2-pl-v2", {"pl_prompt": True}),  # stary generation_config nie przyjmuje language=, więc ręczny prompt <|pl|><|transcribe|>
}


def norm(s):
    s = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", s.lower())
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s)).strip()


def reference(md, lo, hi):
    """Tekst tur (obu mówców) mieszczących się w [lo, hi] s czasu transkryptu + granice czasu."""
    out, prev, first, last = [], 0, None, 0
    for m in re.finditer(r"^\*\*\[(\d+):(\d+) - (\d+):(\d+)\][^\n]*\n+(.+?)(?=\n\*\*\[|\Z)", Path(md).read_text(encoding="utf-8"), re.M | re.S):
        start, end = int(m[1]) * 60 + int(m[2]), int(m[3]) * 60 + int(m[4])
        if start < prev - 5:  # znaczniki zaczynają się od nowa = kolejne nagranie w pliku
            break
        prev = end
        if start >= lo and end <= hi:
            out.append(m[5])
            first = start if first is None else first
            last = end
    return norm(" ".join(out)), first or 0, last


def run(kind, name, kw, wav):
    t0 = time.time()
    audio = sf.read(str(wav), dtype="float32")[0]  # tablica numpy (PyAV w tym środowisku nie pasuje do faster-whisper)
    if kind == "fw":
        from faster_whisper import WhisperModel
        try:
            m = WhisperModel(name, device="cuda", compute_type="float16")
        except Exception as e:  # ponytail: bez GPU spadamy na CPU int8 (wolno)
            print("  GPU niedostępne:", e)
            m = WhisperModel(name, device="cpu", compute_type="int8")
        t1 = time.time()
        segs, _ = m.transcribe(audio, language="pl", beam_size=5, vad_filter=True, condition_on_previous_text=False)
        text = " ".join(s.text.strip() for s in segs)
    elif kind == "onnx":
        import onnx_asr
        m = onnx_asr.load_model(name)
        t1 = time.time()
        text = m.recognize(str(wav))
    elif kind == "cohere":
        import torch
        from transformers import AutoProcessor, CohereAsrForConditionalGeneration
        proc = AutoProcessor.from_pretrained(name)
        m = CohereAsrForConditionalGeneration.from_pretrained(name, device_map="cuda", dtype=torch.float16)
        t1 = time.time()
        inp = proc(audio, sampling_rate=16000, return_tensors="pt", language="pl")
        idx = inp.get("audio_chunk_index")
        inp.to(m.device, dtype=m.dtype)
        out = m.generate(**inp, max_new_tokens=256)
        text = proc.decode(out, skip_special_tokens=True, audio_chunk_index=idx, language="pl")[0]
    else:  # hfw: Whisper w formacie transformers; stare generation_config bez znaczników czasu, więc tniemy po ciszy na kawałki <=28 s
        import torch
        from faster_whisper.vad import VadOptions, get_speech_timestamps
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        proc = AutoProcessor.from_pretrained(name)
        m = AutoModelForSpeechSeq2Seq.from_pretrained(name, dtype=torch.float16).to("cuda").eval()
        t1 = time.time()
        ts = get_speech_timestamps(audio, VadOptions(max_speech_duration_s=28, min_silence_duration_ms=500))
        chunks = [audio[t["start"]:t["end"]] for t in ts]
        texts = []
        kw = dict(kw)
        if kw.pop("pl_prompt", False):
            prompt = proc.tokenizer.convert_tokens_to_ids(["<|startoftranscript|>", "<|pl|>", "<|transcribe|>", "<|notimestamps|>"])
        for i in range(0, len(chunks), 2):
            inp = proc(chunks[i:i + 2], sampling_rate=16000, return_tensors="pt", return_attention_mask=True).to("cuda", torch.float16)
            if "prompt" in locals():
                kw["decoder_input_ids"] = torch.tensor([prompt] * inp.input_features.shape[0], device="cuda")
            with torch.no_grad():
                out = m.generate(**inp, **{"num_beams": 5, "max_new_tokens": 220, **kw})
            texts += proc.batch_decode(out, skip_special_tokens=True)
        text = " ".join(t.strip() for t in texts)
    return text, t1 - t0, time.time() - t1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="?"), ap.add_argument("transcript", nargs="?")
    ap.add_argument("--clip", help="istniejący katalog klipu (clip.wav + reference.txt) zamiast cięcia nowego")
    ap.add_argument("--offset", type=float, default=0, help="s od początku audio do 00:00 transkryptu (koniec preambuły)")
    ap.add_argument("--ref-start", type=float, default=0), ap.add_argument("--ref-end", type=float)
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    a = ap.parse_args()
    if a.clip:
        out = Path(a.clip)
        wav, ref = out / "clip.wav", (out / "reference.txt").read_text(encoding="utf-8")
    else:
        ref, first, last = reference(a.transcript, a.ref_start, a.ref_end)
        start, dur = a.offset + first, last - first + 1
        out = Path("Nagrania/ASR-lokalny/_bench") / f"{Path(a.audio).stem}_{int(start)}-{int(start + dur)}"
        out.mkdir(parents=True, exist_ok=True)
        wav = out / "clip.wav"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(start), "-t", str(dur), "-i", a.audio, "-ac", "1", "-ar", "16000", str(wav)], check=True)
        (out / "reference.txt").write_text(ref, encoding="utf-8")
    dur = sf.info(str(wav)).duration
    rf = out / "results.json"
    res = json.loads(rf.read_text(encoding="utf-8")) if rf.exists() else {}
    for name in a.models:
        print(name, "...", flush=True)
        try:
            text, tl, tt = run(*MODELS[name][:2], MODELS[name][2], wav)
        except Exception as e:
            import traceback; traceback.print_exc()
            print("  BŁĄD:", repr(e)[:300])
            res[name] = {"error": repr(e)[:300]}
            continue
        hyp = norm(text)
        (out / f"{name}.txt").write_text(text, encoding="utf-8")
        res[name] = {"wer": round(jiwer.wer(ref, hyp), 4), "cer": round(jiwer.cer(ref, hyp), 4), "load_s": round(tl, 1),
                     "transcribe_s": round(tt, 1), "rtf": round(tt / dur, 3), "words_ref": len(ref.split()), "words_hyp": len(hyp.split())}
        print("  ", res[name], flush=True)
        rf.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
