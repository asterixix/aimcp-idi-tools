# AIMCP-IDI tools: transcription and LLM-assisted coding of interviews

Tools used in the AIMCP-IDI study (in-depth interviews with metadata staff of Polish academic libraries on artificial intelligence, 2026) to transcribe and anonymize the recordings and to code the anonymized transcripts with the assistance of large language models. The repository lets other researchers re-code the published transcripts with the same instructions, parameters, and verification interface.

- Transcripts (anonymized, Polish, CC BY 4.0): RODBUK, DOI [to be added after publication].
- Methodology, with references: [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).
- Exact instructions sent to the models: [`docs/prompts.md`](docs/prompts.md).
- Adjudication criteria and procedure: [`docs/adjudication.md`](docs/adjudication.md).
- User manual of the coding tool (Polish): [`docs/kodowanie-README.pl.md`](docs/kodowanie-README.pl.md).

The repository contains no interview data, no coding results, and no credentials.

## Contents

| Path | Purpose |
|---|---|
| `scripts/kodowanie_llm.py` | coding tool: `koduj` (model proposals), `serwer` (verification page), `przeglad` (verification in the terminal), `adjudykacja` (import of adjudication decisions), `eksport` (REFI-QDA project and CSV), `sprawdz` (offline check) |
| `scripts/kodowanie.html`, `kodowanie.cmd` | verification page and its Windows launcher |
| `config/wylaczone-fragmenty.md` | turns excluded before coding (rule and table) |
| `vendor/slr_core/` | LLM client, JSON parsing, and REFI-QDA export used by the coding tool (bundled unchanged) |
| `scripts/asr_batch.py`, `asr_diarize.py`, `build_draft.py`, `build_turns.py`, `asr_fragmenty.py`, `render_transcript.py` | transcription workflow: local Whisper, diarization, draft turns, re-reading of unclear fragments, rendering with anonymization checks |
| `scripts/asr_bench.py`, `asr_compare.py`, `asr_agree.py`, `asr_review.py` | evaluation of ASR models and of disagreements between them, used to choose the model and to find passages for listening |
| `public/` | place the transcripts from RODBUK here |

## Re-coding the published transcripts

1. Install Python 3.14 and the dependencies: `pip install -r requirements-coding.txt`.
2. Download the transcripts from RODBUK and copy the files `AIMCP-IDI-NN-Transkrypt.md` to `public/`.
3. Check the setup without any model call: `python scripts/kodowanie_llm.py sprawdz` (the study setup reports 81 excluded turns in 14 transcripts).
4. Provide an API key for an OpenCode Go account in the environment variable `OPENCODE_GO_API_KEY` or in `kodowanie/.env` (see `.env.example`). To use another provider, change `API_BASE`, `MODEL`, and `MODEL_KONTROLNY` in `scripts/kodowanie_llm.py` and report the change.
5. Run the two coding passes for a transcript, for example IDI-62: `python scripts/kodowanie_llm.py koduj 62` and `python scripts/kodowanie_llm.py koduj 62 --etap kontrolne`.
6. Verify the proposals on the local page: `python scripts/kodowanie_llm.py serwer` (or `kodowanie.cmd` on Windows), then export with `python scripts/kodowanie_llm.py eksport`. Set `KODOWANIE_AUTOR` to the coder's name for the REFI-QDA export.

Outputs go to `kodowanie/`, which is excluded from version control. Model outputs differ between runs, so a re-run reproduces the procedure, not the original proposals (see `docs/METHODOLOGY.md`, section 8).

## Transcription workflow

The transcription scripts need a CUDA GPU and the packages in `requirements-transcription.txt` (Python 3.13). The diarization model requires accepting its terms on Hugging Face and logging in with `hf auth login`. The scripts expect the original project layout: recordings in `Nagrania/Audio/`, intermediate files in `Nagrania/ASR-lokalny/`, metadata in `metadane/`, and confidential files (code mapping, lists of identifying phrases) in `internal/`. None of these folders is published, because they contain personal data. The scripts are shared to document the procedure.

## Differences from the scripts used in the study

The scripts are copies of the versions used in the study, with these changes only: the path to the exclusion table (`config/` instead of the study's `tekst/` folder), the path to `slr_core` (`vendor/` instead of a sibling project), the coder's name taken from `KODOWANIE_AUTOR` instead of a fixed name, the coding tool running without the researcher's table of answers to concepts introduced by the interviewer (not published), and internal recording codes in comments and usage examples replaced with `AIBX`. The instruction sent to the models is unchanged; its SHA-256 is given in `docs/prompts.md`.

## License and citation

Code: MIT License (`LICENSE`). The bundled `vendor/slr_core` comes from the author's earlier project and is released under the same license. Please cite the software with `CITATION.cff` and the dataset with its RODBUK DOI.

## Opis po polsku

Narzędzia z badania AIMCP-IDI: lokalna transkrypcja i anonimizacja wywiadów oraz kodowanie wstępne zanonimizowanych transkrypcji wspomagane przez modele językowe, ze stroną do weryfikacji kodów przez badacza. Transkrypcje są w RODBUK, a opis metody w `docs/METHODOLOGY.md`. Instrukcje dla modeli (po polsku, dosłownie) są w `docs/prompts.md`, a instrukcja obsługi w `docs/kodowanie-README.pl.md`. Repozytorium nie zawiera danych z wywiadów, wyników kodowania ani kluczy dostępu.
