# Methodology of transcription and LLM-assisted coding

This document describes how the interviews of the AIMCP-IDI study were transcribed, anonymized, and coded, and which parts of this procedure the tools in this repository reproduce. It accompanies the dataset of anonymized transcripts deposited in RODBUK and the publications based on the study.

## 1. Study and data

The study consists of semi-structured in-depth interviews with staff of Polish academic libraries who work with metadata (cataloging, institutional and research data repositories, digital libraries, documentation of publication output). Fifteen interviews were conducted in Polish between August and October 2026; one was excluded before analysis because the conditions attached to the participant's consent could not be reconciled with anonymization and deposit. The remaining 14 transcripts form the dataset. All participants consented to the deposit of the anonymized transcripts.

## 2. Transcription and anonymization

The recordings never left the researcher's computer. Each recording was transcribed with a locally run Whisper large-v3 model (faster-whisper, CUDA, `float16`, beam size 5, voice activity detection, word time stamps; `scripts/asr_batch.py --model large-v3`), with large-v3-turbo as a second, independent reading. Speakers were separated with the pyannote `speaker-diarization-community-1` pipeline (`scripts/asr_diarize.py`). `scripts/build_draft.py` combined words and speakers into draft turns and marked words with a confidence below 0.4.

The researcher then edited each transcript by hand against the recording. Editing kept the content of the recording and removed fillers, repetitions, and false starts; it did not shorten, summarize, or add content. Uncertain words were resolved by comparing independent readings; fragments that remained unclear were read again with three models on short clips (`scripts/asr_fragmenty.py`: large-v3, large-v3-turbo, and a Polish fine-tuned Whisper large-v2 model, beam size 10, a prompt from the sentence context) and left marked as `[niewyraźne]` (unclear) when the readings did not agree.

Anonymization followed a written protocol based on the guidance of CESSDA, which recommends pseudonymization over blanking out details. The researcher generalized, in square brackets, institutions and organizational units, institution-specific systems and projects, names of third persons (reduced to their roles), career paths, dates, age, family, places of residence, functions, and features unique to an institution. Talk before the first research question and after the announced end of the interview was not transcribed. `scripts/render_transcript.py` produced the published file under a random external code (`AIMCP-IDI-NN`) unrelated to the institution or the order of interviews, and refused to write the file if it contained an internal recording code or any phrase from a confidential list of identifying phrases kept for each interview. No AI service received any text before anonymization.

## 3. Analytic approach

The analysis follows constructivist grounded theory (Charmaz, 2006). Initial coding stays close to the data and names actions and processes with gerunds; *in vivo* codes keep participants' own words (Charmaz, 2006). Sensitizing concepts give the analysis points of departure without serving as a coding frame (Charmaz, 2006). In this study they were four concepts from the research design (PB1 to PB4, among them cognitive authority and professional jurisdiction) and seven contextual threads (W1 to W7), listed verbatim in `docs/prompts.md`.

## 4. Turns excluded before coding

Interviewer turns are never coded. In addition, the researcher excluded, before coding, participant turns that directly followed an interviewer turn in which the interviewer evaluated the topic, reported the interviewer's own experience, offered to do something, or proposed a ready interpretation (category A), and turns in which a participant proposed collaboration with the interviewer together with the turn preceding them (category B). The rule limits the influence of the interviewer's statements on the material that is coded. The table of excluded turns is `config/wylaczone-fragmenty.md`; the coding script reads it, sends excluded turns to the model as "(treść pominięta)" (content omitted), and stops if a time in the table does not identify exactly one participant turn. In the study the rule excluded 81 participant turns.

## 5. LLM-assisted initial coding

Studies of large language models (LLMs) in qualitative analysis report that models reproduce concrete, descriptive themes more readily than interpretive ones (Morgan, 2023), that they can support deductive coding with a codebook (Tai et al., 2024; Xiao et al., 2023), and that a collaboration between a model and researchers can approach the coding quality of human coders in thematic analysis (Dai et al., 2023). All of these studies treat the model as a tool within an analysis led by researchers. The procedure below follows this position: the model proposes, and the researcher decides.

`scripts/kodowanie_llm.py koduj NN` divides a transcript into fragments of about 12,000 characters at interviewer turns, so that a question stays with its answer, and sends each fragment with one written instruction (verbatim in `docs/prompts.md`). For each participant turn marked for coding, the model returns zero to three proposals, each with a code in gerund form or an *in vivo* code, a verbatim quotation, the identifier of a sensitizing concept or "brak" (none), and a one-sentence analytic memo. The output is constrained by a JSON schema. The script discards proposals for turns that were not to be coded and locates each quotation in the turn as exact, approximate, or not found.

Two models coded every transcript with the same instruction:

| Stage | Model (identifier) | Access | Parameters | Date of the study run |
|---|---|---|---|---|
| `otwarte` (initial) | Claude Haiku 5.5 (`claude-haiku-5-5`), Anthropic | OpenCode Go, Anthropic Messages API (`https://opencode.ai/inference/go/v1/messages`), SDK `anthropic` 1.14.0 | default temperature (the model accepts no other value), `effort: medium`, `max_tokens` 16,000, structured output with the JSON schema, up to 3 retries | 14 October 2026 |
| `kontrolne` (control) | DeepSeek V4.1 Flash (`deepseek-v4.1-flash`), DeepSeek | OpenCode Go, OpenAI-compatible Chat Completions API (`https://opencode.ai/inference/go/v1/chat/completions`), client `vendor/slr_core` | temperature 0.15, reasoning disabled (`reasoning_effort: none`), `max_tokens` 8,192, `response_format` with the JSON schema, up to 3 retries with a 5-second delay | 15 October 2026 |

Using a second model from a different developer gives a second set of proposals for every turn, so that the researcher can see where the two readings converge and where one model adds or misses something. According to the published terms of OpenCode Go (checked on 15 October 2026), data sent to these models are not used for training and are retained for 30 days (Claude Haiku 5.5) and for 0 days (DeepSeek V4.1 Flash). Only anonymized transcripts were sent.

## 6. Adjudication between the two passes

A third model, Claude Opus 5.5 (`claude-opus-5-5`), compared the two passes turn by turn and accepted, renamed, or rejected every proposal according to six written criteria. This step was carried out in an interactive Claude Code session on 15 October 2026 under Anthropic's commercial terms. The criteria, the record format, and the limits of reproducing this step are documented in `docs/adjudication.md`.

## 7. Verification by the researcher

The local coding page (`kodowanie.cmd` or `python scripts/kodowanie_llm.py serwer`; address 127.0.0.1 only, no model calls) shows the whole transcript with the interviewer's turns as context and, for every participant turn, the proposals of both models side by side. The researcher accepts, changes (code name or quotation), or rejects each proposal, writes memos, and adds own codes by selecting text. Every decision is logged with its time in `kodowanie/log-decyzji.csv`, and earlier decisions remain in the log. `eksport` writes a REFI-QDA project (`.qdpx`) that opens in MAXQDA, ATLAS.ti, NVivo, and QualCoder, together with CSV tables and a summary of decisions. A filter restricts the page to a list of evidence turns (`kodowanie/dowody-JAL.json`), so that every code and quotation used as evidence in a publication can be verified.

In the study, the researcher verified every code and quotation on the participant turns used as evidence in publications. A focused-coding stage is implemented (`koduj NN --etap zogniskowane` with a researcher-defined list of focused codes) but was not used for the publications to date; the categories reported there were developed from the verified codes with the assistance of an AI tool and accepted by the researcher, as each publication states.

## 8. Transparency, quality, and reproducibility

Tracy (2010) counts self-reflexivity and transparency about methods and challenges among the criteria of quality in qualitative research. The repository serves this purpose: it gives the exact instructions, parameters, model identifiers, dates, the exclusion table, and the code that produced the proposals, and the dataset gives the transcripts on which the codes were made. Sharing qualitative data supports verification and reuse when participants have agreed to it and when the data are protected (DuBois et al., 2018); the transcripts are deposited according to the FAIR principles (Wilkinson et al., 2016) with a persistent identifier and an open license.

The procedure is reproducible as a method, not as a result. Running the tools again on the same transcripts produces different proposals: the outputs of language models vary between runs even with identical inputs (Ouyang et al., 2025), Claude Haiku 5.5 does not accept a temperature other than its default, and model versions available through an API change and are withdrawn over time. The coding results of the original study (proposals, decisions, memos) are not part of the published materials; the publications report the counts and quote the verified evidence.

## 9. Software versions

| Component | Version |
|---|---|
| Coding environment | Python 3.14.4; `anthropic` 1.14.0, `requests` 2.34.2, `pydantic` 2.14.1 |
| Transcription environment | Python 3.13.14; `faster-whisper` 1.2.1, `ctranslate2` 4.8.2, `pyannote.audio` 4.0.7, `torch` 2.11.0 (CUDA 12.8), `torchaudio` 2.11.0, `numpy` 2.5.3, `rapidfuzz` 3.14.6, `jiwer` 4.0.0, `soundfile` 0.14.0, `transformers` 5.18.0, `huggingface-hub` 1.33.0 |
| Models | Whisper large-v3 and large-v3-turbo; pyannote `speaker-diarization-community-1`; Claude Haiku 5.5, DeepSeek V4.1 Flash, Claude Opus 5.5 (identifiers as of 17 October 2026) |

## References

Charmaz, K. (2006). *Constructing grounded theory: A practical guide through qualitative analysis*. SAGE Publications.

Dai, S.-C., Xiong, A., & Ku, L.-W. (2023). LLM-in-the-loop: Leveraging large language model for thematic analysis. In *Findings of the Association for Computational Linguistics: EMNLP 2023* (pp. 9993-10001). Association for Computational Linguistics. https://doi.org/10.18653/v1/2023.findings-emnlp.669

DuBois, J. M., Strait, M., & Walsh, H. (2018). Is it time to share qualitative research data? *Qualitative Psychology, 5*(3), 380-393. https://doi.org/10.1037/qup0000076

Morgan, D. L. (2023). Exploring the use of artificial intelligence for qualitative data analysis: The case of ChatGPT. *International Journal of Qualitative Methods, 22*, Article 16094069231211248. https://doi.org/10.1177/16094069231211248

Ouyang, S., Zhang, J. M., Harman, M., & Wang, M. (2025). An empirical study of the non-determinism of ChatGPT in code generation. *ACM Transactions on Software Engineering and Methodology, 34*(2), 1-28. https://doi.org/10.1145/3697010

Tai, R. H., Bentley, L. R., Xia, X., Sitt, J. M., Fankhauser, S. C., Chicas-Mosier, A. M., & Monteith, B. G. (2024). An examination of the use of large language models to aid analysis of textual data. *International Journal of Qualitative Methods, 23*, Article 16094069241231168. https://doi.org/10.1177/16094069241231168

Tracy, S. J. (2010). Qualitative quality: Eight "big-tent" criteria for excellent qualitative research. *Qualitative Inquiry, 16*(10), 837-851. https://doi.org/10.1177/1077800410383121

Wilkinson, M. D., Dumontier, M., Aalbersberg, I. J., Appleton, G., Axton, M., Baak, A., Blomberg, N., Boiten, J.-W., da Silva Santos, L. B., Bourne, P. E., Bouwman, J., Brookes, A. J., Clark, T., Crosas, M., Dillo, I., Dumon, O., Edmunds, S., Evelo, C. T., Finkers, R., . . . Mons, B. (2016). The FAIR guiding principles for scientific data management and stewardship. *Scientific Data, 3*, Article 160018. https://doi.org/10.1038/sdata.2016.18

Xiao, Z., Yuan, X., Liao, Q. V., Abdelghani, R., & Oudeyer, P.-Y. (2023). Supporting qualitative analysis with large language models: Combining codebook with GPT-3 for deductive coding. In *Companion Proceedings of the 28th International Conference on Intelligent User Interfaces* (pp. 75-78). ACM. https://doi.org/10.1145/3581754.3584136
