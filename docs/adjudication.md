# Adjudication between the two coding passes

## What was done

On 15 October 2026, Claude Opus 5.5 (model identifier `claude-opus-5-5`, Anthropic) compared the initial coding (stage `otwarte`, Claude Haiku 5.5) with the control coding (stage `kontrolne`, DeepSeek V4.1 Flash) of each transcript. The step was carried out in an interactive session of Claude Code (Anthropic's agentic coding tool), not by a script in this repository. For every participant turn the model read the full text of the turn, the preceding interviewer question, and all code proposals of both passes on that turn, and recorded one decision per proposal: accepted (`przyjęty`), renamed (`zmieniony`, with the new code name), or rejected (`odrzucony`), each with a one-sentence justification. The decisions were stored in `kodowanie/IDI-NN/adjudykacja.json` and written into the coding files with `python scripts/kodowanie_llm.py adjudykacja NN`, which requires a decision for every proposal and leaves decisions already made by the researcher unchanged.

## Criteria (verbatim, as stored in the field `zasady` of every adjudication file)

```text
Claude Opus 5.5 (Claude Code) porównał tura po turze kodowanie otwarte (Claude Haiku 5.5) i kontrolne (DeepSeek V4.1 Flash), czytając pełny tekst tury rozmówcy i poprzedzające pytanie badacza. Kryteria: (1) kod ma oparcie w cytacie i nie przypisuje rozmówcy treści spoza tekstu; (2) dwa kody o tym samym znaczeniu na nakładającym się fragmencie: zostaje jeden, domyślnie z kodowania otwartego, z kontrolnego tylko wtedy, gdy trafniej oddaje fragment albo ma dokładny cytat; (3) kody, które nazywają inny aspekt tego samego fragmentu, zostają oba; (4) odrzucane są kody trywialne (potwierdzenia, formułki) i powtórzenia tego samego kodu w jednej turze; (5) zmiana nazwy tylko przy literówce, formie innej niż rzeczownik odczasownikowy albo wyraźnym rozminięciu z cytatem; (6) kody, które przejmują pojęcie „strażnika” z pytania badacza, są odrzucane (R1.4). Pojęcia uwrażliwiające i cytaty nie są zmieniane.
```

English translation:

1. A code is grounded in its quotation and does not attribute to the participant anything that is not in the text.
2. Of two codes with the same meaning on overlapping text, one is kept: by default the one from initial coding, the one from control coding only if it renders the fragment more accurately or has an exact quotation.
3. Codes that name different aspects of the same fragment are both kept.
4. Trivial codes (acknowledgements, set phrases) and repetitions of the same code within one turn are rejected.
5. A code is renamed only for a typo, a form other than a gerund, or a clear mismatch with the quotation.
6. Codes that adopt the term "guardian" from the interviewer's question are rejected. In practice, some of these codes were renamed without the term instead of being rejected.

Sensitizing concepts and quotations are not changed.

## Format of one decision

```json
{"etap": "otwarte", "id": "IDI-NN-T002-1", "decyzja": "przyjęty", "uzasadnienie": "przyjęty: kod ma oparcie w cytacie, bez duplikatu o tym samym znaczeniu"}
```

`kod` is added when `decyzja` is `zmieniony`. The file also records `model`, `narzedzie` (tool), `data` (date), and `zasady` (criteria).

## Reproducing this step

The step can be repeated with any capable model by giving it, turn by turn, the participant turn, the preceding interviewer turn, the proposals of both passes, and the criteria above, and by saving the decisions in the format shown. It cannot be reproduced exactly: the original run took place in an interactive agent session, and outputs of language models vary between runs and model versions (Ouyang et al., 2025). The coding results of the original study are not part of the published materials.

## Status in the analysis

Adjudication by a language model is not a check by a human coder and not a measure of agreement between coders. In the study, the researcher verified every code and quotation on the participant turns used as evidence in publications; the remaining decisions were accepted without card-by-card verification. Publications based on the study report this as a limitation.

## Reference

Ouyang, S., Zhang, J. M., Harman, M., & Wang, M. (2025). An empirical study of the non-determinism of ChatGPT in code generation. *ACM Transactions on Software Engineering and Methodology, 34*(2), 1-28. https://doi.org/10.1145/3697010
