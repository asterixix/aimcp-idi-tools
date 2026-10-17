"""Wstępne kodowanie transkryptów IDI ze wsparciem LLM (konstruktywistyczna teoria ugruntowana, Charmaz 2006).

LLM proponuje kody, autor weryfikuje każdy (przyjęty, zmieniony, odrzucony); decyzje trafiają do kodowanie/log-decyzji.csv.
Wejście wyłącznie public/AIMCP-IDI-NN-Transkrypt.md. Kodowane są tylko tury rozmówcy [R]; tury badacza [B] idą jako
kontekst, a tury z tabeli w tekst/ANALIZA-wylaczone-fragmenty-2026-10-13.md (kategorie A i B) są pomijane automatycznie.
Model: Claude Haiku 5.5 w OpenCode Go (klucz: kodowanie/.env, OPENCODE_GO_API_KEY). Klient LLM, zapis CSV
i paczka QDPX pochodzą z projekty/preludium-25/slr/slr_core.

Użycie (z katalogu projektu; instrukcja: kodowanie/README.md):
  ../../.venv/Scripts/python scripts/kodowanie_llm.py koduj 62 [--etap zogniskowane] [--nadpisz]
  ../../.venv/Scripts/python scripts/kodowanie_llm.py przeglad 62 [--etap zogniskowane]
  ../../.venv/Scripts/python scripts/kodowanie_llm.py serwer [--port 8765]   (strona: weryfikacja i własne kody)
  ../../.venv/Scripts/python scripts/kodowanie_llm.py adjudykacja 62   (decyzje z kodowanie/IDI-NN/adjudykacja.json)
  ../../.venv/Scripts/python scripts/kodowanie_llm.py eksport
  ../../.venv/Scripts/python scripts/kodowanie_llm.py sprawdz
"""
import argparse, csv, hashlib, json, logging, os, re, shutil, sys, uuid
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import anthropic

P = Path(__file__).resolve().parent.parent
OUT = P / "kodowanie"
WYLACZONE = P / "config" / "wylaczone-fragmenty.md"  # w projekcie badawczym: tekst/ANALIZA-wylaczone-fragmenty-2026-10-13.md
ZOGNISKOWANE = OUT / "kody-zogniskowane.md"
LOG_DECYZJI = OUT / "log-decyzji.csv"
REAKCJE = OUT / "reakcje-na-pojecie-badacza.csv"
KOD_REAKCJI = "reakcja na pojęcie badacza"
AUTOR = os.environ.get("KODOWANIE_AUTOR", "koder")  # nazwa użytkownika w eksporcie REFI-QDA
API_BASE = "https://opencode.ai/inference/go"  # SDK dopisuje /v1/messages
MODEL = "claude-haiku-5-5"  # OpenCode Go: bez trenowania na danych, retencja 30 dni
MODEL_KONTROLNY = "deepseek-v4.1-flash"  # kodowanie kontrolne; OpenCode Go: bez trenowania, retencja 0 dni (ZDR)

# Logi klienta LLM do kodowanie/_logi, nie do slr/SLR_OUTPUT: setup_logging() w slr_core pomija logger, który ma już handler.
(OUT / "_logi").mkdir(parents=True, exist_ok=True)
_fh = logging.FileHandler(OUT / "_logi" / "llm.log", encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logging.getLogger("slr_auto_coder").addHandler(_fh)
logging.getLogger("slr_auto_coder").setLevel(logging.INFO)
sys.path.insert(0, str(P / "vendor"))  # w projekcie badawczym: projekty/preludium-25/slr (slr_core)
from slr_core.config import CONFIG  # noqa: E402
from slr_core.llm import LLMClient  # noqa: E402
from slr_core.qda import _export_qdpx, _write_csv  # noqa: E402
from slr_core.utils import _atomic_write_json, _atomic_write_text  # noqa: E402

TURA = re.compile(r"^\*\*\[(\d+:\d{2}) - (\d+:\d{2})\] (?:Badacz|Respondent) \[([BR])\]:\*\*[ \t]*\n", re.M)
FRAGMENT_ZNAKI = 12000  # tury grupowane w fragmenty tej długości; pytanie badacza zostaje z odpowiedzią

# Pojęcia uwrażliwiające: PB1-PB4 z Rozdzial_3_Metodologia.md (3.2), W1-W7 z ANALIZA-material-a-temat-2026-10-13.md (4).
POJECIA = """PB1: warunki uznania metadanych tworzonych z udziałem AI za wiarygodne; funkcja strażnika jakości (gatekeeper); autorytet poznawczy (Wilson 1983).
PB2: elementy pracy uznawane za niezastępowalne i ich uzasadnienia; jurysdykcja zawodowa (Abbott 1988).
PB3: kompetencje i formy współpracy z AI (narzędzia wdrażane odgórnie i używane z własnej inicjatywy), szkolenia, bariery, szanse rozwoju.
PB4: warunki integracji AI z systemami bibliotecznymi: dostęp do danych, interoperacyjność, uprawnienia i bezpieczeństwo, rola instytucji centralnych i dostawców.
W1: odgórne wdrożenie: asystent AI do katalogowania w Almie udostępniony przez Bibliotekę Narodową.
W2: użycie oddolne, nieformalne (czaty, własne narzędzia).
W3: automatyzacja bez AI jako punkt odniesienia (harvestery, skrypty, współkatalogowanie).
W4: bibliotekarze jako dostawcy danych treningowych i oceniający AI.
W5: ekonomia i etaty.
W6: pokolenia.
W7: dezinformacja i społeczne zaufanie do bibliotek."""

ZASADY = """Jesteś asystentem kodowania w badaniu jakościowym (konstruktywistyczna teoria ugruntowana, Charmaz 2006): wywiady pogłębione z pracownikami bibliotek akademickich o pracy z metadanymi i AI. Proponujesz kody; badacz zweryfikuje każdy.
Zasady:
- Koduj wyłącznie tury oznaczone KODUJ (wypowiedzi rozmówcy). Tury badacza [B] są tylko kontekstem.
- Fragment ma od jednego zdania do krótkiego akapitu. Pole "cytat" to dosłowny, ciągły fragment tury, bez skrótów i zmian.
- Od 0 do 3 kodów na turę; pomijaj tury bez treści analitycznej (potwierdzenia, formułki).
- Pole "pojecie": identyfikator pojęcia uwrażliwiającego (PB1-PB4, W1-W7), gdy fragment wyraźnie go dotyczy, w przeciwnym razie "brak". Pojęcia wskazują kierunek uwagi; nie dopasowuj na siłę.
- Pole "memo": jedno zdanie analityczne: co rozmówca robi lub jakie znaczenie konstruuje w tym fragmencie.
- Nie dopowiadaj faktów spoza tekstu.
Zwróć wyłącznie JSON: {"kody": [{"tura": "T017", "kod": "...", "cytat": "...", "pojecie": "PB1", "memo": "..."}]}"""

ETAPY = {
    "otwarte": "Etap: kodowanie wstępne (otwarte). Czytaj turę po turze, blisko danych. Kod to krótka fraza w formie rzeczownika "
               "odczasownikowego (gerundium), np. \"kwestionowanie wiarygodności automatycznych opisów\", \"negocjowanie granic "
               "własnych kompetencji\". Dozwolone kody in vivo: słowa rozmówcy w cudzysłowie. Nie narzucaj kategorii z góry.",
    "zogniskowane": "Etap: kodowanie skoncentrowane (zogniskowane). Przypisuj fragmentom kody z listy KODY ZOGNISKOWANE, "
                    "dosłownie według nazwy. Jeśli fragment jest analitycznie ważny, a żaden kod nie pasuje, zaproponuj nowy kod "
                    "w formie gerundium z przedrostkiem \"NOWY: \".",
}
ETAPY["kontrolne"] = ETAPY["otwarte"]  # drugi model, ta sama instrukcja: porównanie z kodowaniem otwartym

SCHEMAT = {"type": "object", "additionalProperties": False, "required": ["kody"], "properties": {"kody": {
    "type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["tura", "kod", "cytat", "pojecie", "memo"],
                               "properties": {k: {"type": "string"} for k in ("tura", "kod", "cytat", "pojecie", "memo")}}}}}


class Claude(LLMClient):
    """OpenCode Go podaje modele Claude w formacie Anthropic Messages, nie chat/completions; log i parsowanie JSON z slr_core."""

    def _make_request(self, prompt, *, model_override=None, structured=False, max_tokens_override=None, timeout_read_override=None):
        try:
            r = self.config["sdk"].messages.create(
                model=model_override or self.model, max_tokens=16000, messages=[{"role": "user", "content": prompt}],
                output_config={"effort": "medium", **({"format": {"type": "json_schema", "schema": SCHEMAT}} if structured else {})})
        except anthropic.APIStatusError as exc:  # 429 i 5xx SDK ponawia samo (max_retries)
            self._log(f"LLM HTTP {exc.status_code}: {exc.message}")
            return None
        except anthropic.APIConnectionError as exc:
            self._log(f"LLM błąd połączenia: {exc}")
            return None
        self.total_requests += 1
        self.total_prompt_tokens += r.usage.input_tokens
        self.total_completion_tokens += r.usage.output_tokens
        if r.stop_reason != "end_turn":  # refusal albo max_tokens: JSON niepełny lub brak
            self._log(f"LLM stop_reason={r.stop_reason} {getattr(r, 'stop_details', None)}")
            return None
        return "".join(b.text for b in r.content if b.type == "text")


def klucz():
    env = OUT / ".env"
    if "OPENCODE_GO_API_KEY" not in os.environ and env.exists():
        for linia in env.read_text(encoding="utf-8").splitlines():
            k, _, v = linia.partition("=")
            if k.strip() == "OPENCODE_GO_API_KEY":
                return v.strip()
    return os.environ.get("OPENCODE_GO_API_KEY") or sys.exit(f"Brak OPENCODE_GO_API_KEY w {env} ani w zmiennych środowiska.")


def wczytaj(nr):
    """(IDI-NN, tekst pliku, opis informanta, skrót, tury); tura: id, mowca, start, koniec, tekst, poz (offset w tekście)."""
    nr = re.sub(r"\D", "", str(nr)).zfill(2)
    tekst = (P / "public" / f"AIMCP-IDI-{nr}-Transkrypt.md").read_text(encoding="utf-8")
    hdr = list(TURA.finditer(tekst))
    tury = []
    for i, m in enumerate(hdr):
        seg = tekst[m.end():hdr[i + 1].start() if i + 1 < len(hdr) else len(tekst)]
        seg = re.split(r"\n\s*(?:#|---)", seg)[0]  # nagłówek sesji (IDI-48) nie należy do tury
        poz = m.end() + len(seg) - len(seg.lstrip())
        tury.append({"id": f"T{i + 1:03d}", "mowca": m[3], "start": m[1], "koniec": m[2], "tekst": seg.strip(), "poz": poz})
    inf = re.search(r"\*\*Informant \[R / Respondent\]:\*\*\s*(.+)", tekst)
    return f"IDI-{nr}", tekst, inf[1].strip() if inf else "", hashlib.sha256(tekst.encode()).hexdigest()[:16], tury


def wylaczone(idi, tury):
    """Id tur R wyłączonych wg tabeli w WYLACZONE; czas bez jednoznacznej tury R zatrzymuje skrypt."""
    ids = set()
    for wiersz in WYLACZONE.read_text(encoding="utf-8").splitlines():
        kom = [k.strip() for k in wiersz.strip().strip("|").split("|")]
        if len(kom) < 4 or kom[0] != idi:
            continue
        for c in re.findall(r"\b\d{1,3}:\d{2}\b", kom[2]):
            traf = [t["id"] for t in tury if t["mowca"] == "R" and t["start"] == c]
            if len(traf) != 1:
                sys.exit(f"{WYLACZONE.name}: {idi} {c} pasuje do {len(traf)} tur R; popraw tabelę.")
            ids.add(traf[0])
    return ids


def reakcje(idi, tury):
    """Oznaczenia autora z REAKCJE (R1 w tekst/ANALIZA-ryzyka-metodologiczne-artykuly-2026-10-14.md): odpowiedzi na pytanie badacza
    o strażnika jakości. Tura rozmówcy musi istnieć jednoznacznie i następować bezpośrednio po wskazanej turze badacza."""
    wynik = []
    if not REAKCJE.exists():  # tabela autora nie jest częścią pakietu
        return wynik
    with open(REAKCJE, encoding="utf-8-sig", newline="") as f:
        for w in csv.DictReader(f):
            if w["transkrypt"] != idi:
                continue
            traf = [i for i, t in enumerate(tury) if t["mowca"] == "R" and t["start"] == w["tura_rozmowcy"]]
            if len(traf) != 1 or tury[traf[0] - 1]["mowca"] != "B" or tury[traf[0] - 1]["start"] != w["tura_badacza"]:
                sys.exit(f"{REAKCJE.name}: {idi} {w['tura_badacza']} -> {w['tura_rozmowcy']} nie wskazuje jednoznacznie pary pytanie-odpowiedź; popraw tabelę.")
            t = tury[traf[0]]
            wynik.append({"id": f"{idi}-{t['id']}-reakcja", "tura": t["id"], "czas": t["start"], "tura_badacza": w["tura_badacza"],
                          "kod": KOD_REAKCJI, "wartosc": w["reakcja"].strip(), "start": t["poz"], "koniec": t["poz"] + len(t["tekst"]),
                          "cytat": t["tekst"]})
    return wynik


def wczytaj_kodowanie(plik, skrot, tury):
    """Plik kodowania z oznaczeniami z REAKCJE, dopisywanymi bez zmiany propozycji LLM i decyzji; None, gdy transkrypt się zmienił."""
    dane = json.loads(plik.read_text(encoding="utf-8"))
    if dane["skrot_pliku"] != skrot:
        return None
    dane.setdefault("kody_autora", [])
    r = reakcje(dane["transkrypt"], tury)
    if dane.get("reakcje") != r:
        dane["reakcje"] = r
        _atomic_write_json(str(plik), dane)
        widok(dane, tury)
    return dane


def zlokalizuj(cytat, tekst):
    """(start, koniec, status) cytatu w tekście tury: dokładny, przybliżony (interpunkcja, wielokropek) albo nieznaleziony (cała tura)."""
    i = tekst.find(cytat) if cytat else -1
    if i >= 0:
        return i, i + len(cytat), "dokładny"
    czesci = [c for c in re.split(r"\[?(?:\.\.\.|…)\]?", cytat or "") if re.search(r"\w", c)]
    wzor = r".*?".join(r"\W+".join(map(re.escape, re.findall(r"\w+", c))) for c in czesci)
    m = re.search(wzor, tekst, re.S) if wzor else None
    return (m.start(), m.end(), "przybliżony") if m else (0, len(tekst), "nieznaleziony")


def lista_zogniskowanych():
    if not ZOGNISKOWANE.exists():
        sys.exit(f"Brak {ZOGNISKOWANE}: wpisz kody zogniskowane jako linie '- kod: definicja' (podstawa: kodowanie/eksport/ksiega-kodow.csv).")
    return "\n".join(l for l in ZOGNISKOWANE.read_text(encoding="utf-8").splitlines() if l.startswith("- "))


def koduj(nr, etap, nadpisz):
    idi, tekst, informant, skrot, tury = wczytaj(nr)
    cel = OUT / idi / f"{etap}.json"
    if cel.exists():
        if not nadpisz:
            sys.exit(f"{cel.relative_to(P)} istnieje. Weryfikacja: przeglad {nr}; nowe kodowanie: --nadpisz (stara wersja do kodowanie/_archiwum/).")
        arch = OUT / "_archiwum" / f"{idi}-{etap}-{datetime.now():%Y%m%d-%H%M%S}.json"
        arch.parent.mkdir(exist_ok=True)
        shutil.move(cel, arch)
    wyl = wylaczone(idi, tury)
    reak = reakcje(idi, tury)
    instrukcja = f"{ZASADY}\n\n{ETAPY[etap]}\n\nPOJĘCIA UWRAŻLIWIAJĄCE:\n{POJECIA}"
    if etap == "zogniskowane":
        instrukcja += f"\n\nKODY ZOGNISKOWANE:\n{lista_zogniskowanych()}"

    # Fragmenty według tur; nowy fragment zaczyna się od tury badacza, gdy bieżący przekroczył limit.
    fragmenty, biezacy = [], []
    for t in tury:
        if t["mowca"] == "B" and sum(len(x["tekst"]) for x in biezacy) > FRAGMENT_ZNAKI:
            fragmenty.append(biezacy)
            biezacy = []
        biezacy.append(t)
    fragmenty.append(biezacy)

    sesja = f"kodowanie-{idi}-{etap}-{uuid.uuid4().hex[:12]}"  # OpenCode Go: stały identyfikator sesji na uruchomienie
    naglowki = {"x-opencode-session": sesja, "User-Agent": "kodowanie-idi/1.0"}
    if etap == "kontrolne":  # endpoint zgodny z OpenAI; klient slr_core bez zmian, schemat JSON nasz
        model, endpoint = MODEL_KONTROLNY, f"{API_BASE}/v1/chat/completions"
        klient = LLMClient({**CONFIG, "api_base": f"{API_BASE}/v1", "api_key": klucz(), "model": model, "coding_scheme_path": None,
                            "messages_api_models": [], "extra_headers": naglowki})
        klient._json_schema = SCHEMAT
    else:
        model, endpoint = MODEL, f"{API_BASE}/v1/messages"
        sdk = anthropic.Anthropic(api_key=klucz(), base_url=API_BASE, max_retries=3, default_headers=naglowki)
        klient = Claude({**CONFIG, "model": model, "coding_scheme_path": None, "sdk": sdk})
    po_id = {t["id"]: t for t in tury}
    kody, nieudane, odrzucone_auto = [], [], 0
    for n, frag in enumerate(fragmenty, 1):
        kodowalne = {t["id"] for t in frag if t["mowca"] == "R" and t["id"] not in wyl}
        if not kodowalne:
            continue
        linie = []
        for t in frag:
            if t["mowca"] == "B":
                linie.append(f"[{t['id']} | {t['start']}-{t['koniec']} | B, kontekst] {t['tekst']}")
            elif t["id"] in wyl:
                linie.append(f"[{t['id']} | {t['start']}-{t['koniec']} | R, wyłączona z kodowania] (treść pominięta)")
            else:
                linie.append(f"[{t['id']} | {t['start']}-{t['koniec']} | R, KODUJ] {t['tekst']}")
        prompt = f"{instrukcja}\n\nWYWIAD: {idi}; rozmówca: {informant}\nFRAGMENT {n}/{len(fragmenty)}:\n\n" + "\n\n".join(linie)
        odp = klient._request_text(prompt, structured=True)
        dane = klient._extract_json(odp or "")
        if not isinstance(dane, dict) or not isinstance(dane.get("kody"), list):
            nieudane += sorted(kodowalne)
            print(f"  fragment {n}/{len(fragmenty)}: brak poprawnej odpowiedzi LLM (szczegóły: kodowanie/_logi/llm.log)")
            continue
        przed = len(kody)
        for k in dane["kody"]:
            t = po_id.get(str(k.get("tura", "")).strip())
            if not t or t["id"] not in kodowalne or not str(k.get("kod", "")).strip():
                odrzucone_auto += 1
                continue
            s, e, status = zlokalizuj(str(k.get("cytat", "")).strip(), t["tekst"])
            kody.append({"id": f"{idi}-{t['id']}-{sum(1 for x in kody if x['tura'] == t['id']) + 1}", "tura": t["id"],
                         "czas": t["start"], "kod_llm": str(k["kod"]).strip(), "cytat_llm": str(k.get("cytat", "")).strip(),
                         "pojecie": str(k.get("pojecie", "brak")).strip() or "brak", "memo_llm": str(k.get("memo", "")).strip(),
                         "start": t["poz"] + s, "koniec": t["poz"] + e, "cytat": t["tekst"][s:e], "cytat_status": status,
                         "decyzja": None, "kod": None, "memo_autora": "", "czas_decyzji": None})
        print(f"  fragment {n}/{len(fragmenty)}: {len(kody) - przed} kodów")

    if not kody and nieudane:
        sys.exit("LLM nie zwrócił żadnej poprawnej odpowiedzi; nic nie zapisano (szczegóły: kodowanie/_logi/llm.log).")
    dane = {"transkrypt": idi, "plik": f"public/AIMCP-IDI-{idi[4:]}-Transkrypt.md", "skrot_pliku": skrot, "etap": etap,
            "model": model, "api_base": endpoint, "sesja": sesja,
            "tokeny": {"zapytania": klient.total_requests, "wejscie": klient.total_prompt_tokens, "wyjscie": klient.total_completion_tokens},
            "data": datetime.now().isoformat(timespec="seconds"),
            "instrukcja": instrukcja, "tury_R": sum(t["mowca"] == "R" for t in tury), "tury_wylaczone": sorted(wyl),
            "nieudane_tury": nieudane, "odrzucone_automatycznie": odrzucone_auto, "kody": kody, "reakcje": reak, "kody_autora": []}
    _atomic_write_json(str(cel), dane)
    widok(dane, tury)
    print(f"{idi} ({etap}): {len(kody)} propozycji, {len(wyl)} tur wyłączonych, {odrzucone_auto} odrzuconych automatycznie"
          f" (tura spoza listy KODUJ). Wynik: {cel.relative_to(P)}, widok: {cel.with_suffix('.md').relative_to(P)}")
    if nieudane:
        print(f"UWAGA: bez kodów zostały tury {', '.join(nieudane)}; powtórz z --nadpisz.")
    print(f"Dalej: przeglad {idi[4:]}" + (" --etap zogniskowane" if etap == "zogniskowane" else ""))


def widok(dane, tury):
    """Czytelny widok Markdown: tury R z propozycjami i decyzjami (regenerowany po kodowaniu i przeglądzie)."""
    licz = Counter(k["decyzja"] or "bez decyzji" for k in dane["kody"])
    linie = [f"# {dane['transkrypt']}, kodowanie {dane['etap']}: propozycje LLM i decyzje autora", "",
             f"Model: {dane['model']}; kodowanie: {dane['data']}; propozycje: {len(dane['kody'])} "
             f"({', '.join(f'{k}: {v}' for k, v in sorted(licz.items()))}); oznaczenia autora ({KOD_REAKCJI}): {len(dane['reakcje'])}.", ""]
    reak = {r["tura"]: r for r in dane["reakcje"]}
    po_turze, autora = defaultdict(list), defaultdict(list)
    for k in dane["kody"]:
        po_turze[k["tura"]].append(k)
    for k in dane["kody_autora"]:
        if k["decyzja"] == "dodany przez autora":
            autora[k["tura"]].append(k)
    linie[2] = linie[2][:-1] + f"; kody dodane przez autora: {sum(map(len, autora.values()))}."
    for t in tury:
        if t["mowca"] != "R":
            continue
        stan = " (wyłączona z kodowania)" if t["id"] in dane["tury_wylaczone"] else "" if po_turze[t["id"]] or autora[t["id"]] else " (bez kodów)"
        linie += [f"## {t['id']} [{t['start']} - {t['koniec']}]{stan}", "", f"> {t['tekst']}", ""]
        if t["id"] in reak:
            r = reak[t["id"]]
            linie.append(f"- **{KOD_REAKCJI}: {r['wartosc']}** [oznaczenie autora, odpowiedź na pytanie badacza {r['tura_badacza']}], {r['id']}")
        for k in po_turze[t["id"]]:
            kod = k["kod"] if k["decyzja"] == "zmieniony" else k["kod_llm"]
            zr = f", {k['zrodlo_decyzji']}" if k.get("zrodlo_decyzji") else ""
            linie.append(f"- **{kod}** ({k['pojecie']}) [{k['decyzja'] or 'bez decyzji'}{zr}{', zweryfikowany' if k.get('weryfikacja') else ''}], {k['id']}")
            if k.get("uzasadnienie_adjudykacji"):
                linie.append(f"  - uzasadnienie adjudykacji: {k['uzasadnienie_adjudykacji']}")
            if k["decyzja"] == "zmieniony":
                linie.append(f"  - propozycja LLM: {k['kod_llm']}")
            linie.append(f"  - cytat ({k['cytat_status']}): „{k['cytat']}”")
            linie.append(f"  - memo LLM: {k['memo_llm']}")
            if k["memo_autora"]:
                linie.append(f"  - memo autora: {k['memo_autora']}")
        for k in autora[t["id"]]:
            linie += [f"- **{k['kod']}** [dodany przez autora], {k['id']}", f"  - cytat: „{k['cytat']}”"]
            if k["memo_autora"]:
                linie.append(f"  - memo autora: {k['memo_autora']}")
        linie.append("")
    _atomic_write_text(str(OUT / dane["transkrypt"] / f"{dane['etap']}.md"), "\n".join(linie))


def zapisz_decyzje(dane, k):
    nowy = not LOG_DECYZJI.exists()
    with open(LOG_DECYZJI, "a", newline="", encoding="utf-8-sig" if nowy else "utf-8") as f:
        w = csv.writer(f)
        if nowy:
            w.writerow(["czas_decyzji", "transkrypt", "etap", "id", "tura_czas", "pojecie", "kod_llm", "decyzja", "kod_koncowy",
                        "cytat_zmieniony", "memo_autora", "model"])
        w.writerow([k["czas_decyzji"], dane["transkrypt"], dane["etap"], k["id"], k["czas"], k["pojecie"], k["kod_llm"],
                    k["decyzja"], k["kod"] or "", "tak" if k.get("cytat_zmieniony") else "nie", k["memo_autora"], dane["model"]])


def teraz():
    return datetime.now().isoformat(timespec="seconds")


def decyduj(plik, dane, k, t, odrzuc, kod="", cytat="", memo="", start=None):
    """Decyzja autora o propozycji LLM (terminal i strona): zapis pliku kodowania i wiersza w log-decyzji.csv.
    Zmieniony = inny kod albo cytat niż w propozycji; cytat musi być dosłownym fragmentem tury."""
    if odrzuc:
        k.update(decyzja="odrzucony", kod=None, memo_autora=memo)
    else:
        if cytat and (cytat != k["cytat"] or start is not None and t["poz"] + start != k["start"]):
            s = start if start is not None and t["tekst"][start:start + len(cytat)] == cytat else t["tekst"].index(cytat)
            k.update(start=t["poz"] + s, koniec=t["poz"] + s + len(cytat), cytat=cytat, cytat_status="autor", cytat_zmieniony=True)
        kod = kod or k["kod_llm"]
        k.update(decyzja="zmieniony" if kod != k["kod_llm"] or k.get("cytat_zmieniony") else "przyjęty", kod=kod, memo_autora=memo)
    k["czas_decyzji"] = k["weryfikacja"] = teraz()  # pierwsza runda (zbiorcze przyjęcie) nie ma pola weryfikacja
    k.pop("zrodlo_decyzji", None)  # decyzja autora zastępuje adjudykację; jej uzasadnienie zostaje do wglądu
    _atomic_write_json(str(plik), dane)
    zapisz_decyzje(dane, k)


def dodaj_wlasny(plik, dane, t, kod, cytat, memo, start=None):
    """Kod autora spoza propozycji LLM: osobna lista, żeby nie mieszać go ze statystyką decyzji o propozycjach."""
    s = start if start is not None and t["tekst"][start:start + len(cytat)] == cytat else t["tekst"].index(cytat)
    n = sum(x["tura"] == t["id"] for x in dane["kody_autora"]) + 1
    k = {"id": f"{dane['transkrypt']}-{t['id']}-A{n}", "tura": t["id"], "czas": t["start"], "pojecie": "brak", "kod_llm": "",
         "decyzja": "dodany przez autora", "kod": kod, "cytat": cytat, "start": t["poz"] + s, "koniec": t["poz"] + s + len(cytat),
         "memo_autora": memo, "czas_decyzji": teraz()}
    dane["kody_autora"].append(k)
    _atomic_write_json(str(plik), dane)
    zapisz_decyzje(dane, k)
    return k


def przeglad(nr, etap):
    idi, _, _, skrot, tury = wczytaj(nr)
    plik = OUT / idi / f"{etap}.json"
    if not plik.exists():
        sys.exit(f"Brak {plik.relative_to(P)}; najpierw: koduj {nr}" + (" --etap zogniskowane" if etap == "zogniskowane" else ""))
    dane = wczytaj_kodowanie(plik, skrot, tury)
    if dane is None:
        sys.exit(f"public/AIMCP-{idi}-Transkrypt.md zmienił się po kodowaniu; pozycje cytatów są nieaktualne. Zakoduj ponownie z --nadpisz.")
    reak = {r["tura"]: r for r in dane["reakcje"]}
    po_id = {t["id"]: t for t in tury}
    poprzednia = {t["id"]: tury[i - 1] for i, t in enumerate(tury) if i}
    czeka = [k for k in dane["kody"] if not k["decyzja"]]
    print(f"{idi} ({etap}): {len(czeka)} z {len(dane['kody'])} kodów do weryfikacji. Model: {dane['model']}.")
    print("Klawisze: p przyjmij, m przyjmij z memo, z zmień (kod, cytat, memo), o odrzuć, s pomiń, q zakończ (postęp jest zapisany).")
    try:
        for n, k in enumerate(czeka, 1):
            t, b = po_id[k["tura"]], poprzednia.get(k["tura"])
            s, e = k["start"] - t["poz"], k["koniec"] - t["poz"]
            print("\n" + "=" * 78)
            print(f"[{n}/{len(czeka)}] {k['id']}  {t['start']}-{t['koniec']}  pojęcie: {k['pojecie']}  cytat: {k['cytat_status']}")
            if b and b["mowca"] == "B":
                print(f"\nB: {b['tekst'][:400]}{' (...)' if len(b['tekst']) > 400 else ''}")
            print(f"\nR: {t['tekst'][:s]}»{t['tekst'][s:e]}«{t['tekst'][e:]}")
            if k["tura"] in reak:
                print(f"\nOZNACZENIE: {KOD_REAKCJI}: {reak[k['tura']]['wartosc']} (odpowiedź na pytanie badacza; nie dowód wyłonienia kategorii)")
            print(f"\nKOD: {k['kod_llm']}\nmemo LLM: {k['memo_llm']}")
            while True:
                w = input("[p/m/z/o/s/q] > ").strip().lower()
                if w in ("p", "m"):
                    decyduj(plik, dane, k, t, False, memo=input("memo: ").strip() if w == "m" else "")
                elif w == "z":
                    kod = input(f"kod (Enter = {k['kod_llm']}): ").strip()
                    while True:
                        cyt = input("cytat dosłownie z tury (Enter = bez zmian): ").strip()
                        if not cyt or cyt in t["tekst"]:
                            break
                        print("Tego fragmentu nie ma w turze; skopiuj go dokładnie.")
                    decyduj(plik, dane, k, t, False, kod, cyt, input("memo (Enter = brak): ").strip())
                elif w == "o":
                    decyduj(plik, dane, k, t, True, memo=input("powód (Enter = brak): ").strip())
                elif w == "s":
                    break
                elif w == "q":
                    raise KeyboardInterrupt
                else:
                    continue
                break
    except (KeyboardInterrupt, EOFError):
        print("\nPrzerwano; decyzje do tej pory są zapisane.")
    widok(dane, tury)
    licz = Counter(k["decyzja"] or "bez decyzji" for k in dane["kody"])
    print(f"\n{idi} ({etap}): " + ", ".join(f"{a}: {b}" for a, b in sorted(licz.items())) + f". Widok: {plik.with_suffix('.md').relative_to(P)}")


def adjudykacja(nr):
    """Decyzje z kodowanie/IDI-NN/adjudykacja.json (porównanie kodowania otwartego i kontrolnego przez trzeci model) jako decyzje
    ze źródłem „adjudykacja <model>” i uzasadnieniem; pole weryfikacja (sprawdzenie karty przez autora) zostaje bez zmian,
    a karty z decyzją autora podjętą na stronie są pomijane."""
    idi, _, _, skrot, tury = wczytaj(nr)
    a = json.loads((OUT / idi / "adjudykacja.json").read_text(encoding="utf-8"))
    zrodlo, licz = f"adjudykacja {a['model']}", Counter()
    for etap in ("otwarte", "kontrolne"):
        plik = OUT / idi / f"{etap}.json"
        dane = wczytaj_kodowanie(plik, skrot, tury)
        if dane is None:
            sys.exit(f"{idi}: transkrypt zmienił się po kodowaniu.")
        karty = {k["id"]: k for k in dane["kody"]}
        decyzje = {x["id"]: x for x in a["decyzje"] if x["etap"] == etap}
        if set(decyzje) != set(karty):
            sys.exit(f"{idi} {etap}: adjudykacja musi objąć każdą kartę (brak {len(set(karty) - set(decyzje))}, nadmiar {len(set(decyzje) - set(karty))}).")
        for kid, x in decyzje.items():
            k = karty[kid]
            if k.get("weryfikacja") and not k.get("zrodlo_decyzji"):
                licz[(etap, "pominięta (decyzja autora)")] += 1  # karta rozstrzygnięta przez autora na stronie
                continue
            if x["decyzja"] == "odrzucony":
                nowa = {"decyzja": "odrzucony", "kod": None}
            else:
                kod = x.get("kod") or k["kod_llm"]
                nowa = {"decyzja": "zmieniony" if kod != k["kod_llm"] or k.get("cytat_zmieniony") else "przyjęty", "kod": kod}
            if k.get("zrodlo_decyzji") == zrodlo and all(k.get(p) == v for p, v in nowa.items()):
                continue  # ponowne uruchomienie nie dubluje wpisów w dzienniku
            k.update(nowa, czas_decyzji=teraz(), zrodlo_decyzji=zrodlo, uzasadnienie_adjudykacji=x["uzasadnienie"])
            zapisz_decyzje(dane, {**k, "memo_autora": f"[{zrodlo}] {x['uzasadnienie']}"})
            licz[(etap, k["decyzja"])] += 1
        _atomic_write_json(str(plik), dane)
        widok(dane, tury)
    print(f"{idi}: " + ", ".join(f"{e} {d}: {n}" for (e, d), n in sorted(licz.items())) if licz else f"{idi}: bez zmian")


def guid(*czesci):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "aimcp-idi-kodowanie/" + "/".join(czesci)))


def eksport():
    """Eksport zweryfikowanych kodów: REFI-QDA (.qdpx), CSV wszystkich propozycji, księga kodów, podsumowanie decyzji."""
    cel, zrodla_dir = OUT / "eksport", OUT / "eksport" / "zrodla"
    zrodla_dir.mkdir(parents=True, exist_ok=True)
    teraz = datetime.now().isoformat(timespec="seconds")
    wiersze, podsum, teksty, widziane = [], [], {}, set()
    for plik in sorted(p for p in OUT.glob("IDI-*/*.json") if p.stem in ETAPY):  # bez adjudykacja.json
        idi, tekst, _, skrot, tury = wczytaj(plik.parent.name)
        dane = wczytaj_kodowanie(plik, skrot, tury)
        if dane is None:
            print(f"POMINIĘTO {plik.relative_to(P)}: transkrypt zmienił się po kodowaniu.")
            continue
        teksty[idi] = tekst
        licz = Counter(k["decyzja"] or "bez decyzji" for k in dane["kody"])
        st = Counter(k["cytat_status"] for k in dane["kody"])
        podsum.append([idi, dane["etap"], dane["model"], dane["data"][:10], len(dane["kody"]), licz["przyjęty"], licz["zmieniony"],
                       licz["odrzucony"], licz["bez decyzji"], st["przybliżony"], st["nieznaleziony"],
                       dane["odrzucone_automatycznie"], len(dane["tury_wylaczone"]),
                       sum(k["decyzja"] == "dodany przez autora" for k in dane["kody_autora"]),
                       sum(bool(k.get("weryfikacja")) for k in dane["kody"]),
                       sum(bool(k.get("zrodlo_decyzji")) for k in dane["kody"])])
        for k in dane["kody"]:
            wiersze.append({"transkrypt": idi, "etap": dane["etap"], "id": k["id"], "tura_czas": k["czas"], "pojecie": k["pojecie"],
                            "kod_llm": k["kod_llm"], "decyzja": k["decyzja"] or "bez decyzji", "kod_koncowy": k["kod"] or "",
                            "wartosc": "", "weryfikacja": k.get("weryfikacja") or "",
                            "zrodlo": k.get("zrodlo_decyzji") or ("autor" if k["decyzja"] else ""), "cytat": k["cytat"], "cytat_status": k["cytat_status"], "memo_llm": k["memo_llm"],
                            "memo_autora": k["memo_autora"], "start": k["start"], "koniec": k["koniec"]})
        for r in (r for r in dane["reakcje"] if r["id"] not in widziane):  # ta sama tura w plikach obu etapów
            widziane.add(r["id"])
            wiersze.append({"transkrypt": idi, "etap": "oznaczenie", "id": r["id"], "tura_czas": r["czas"], "pojecie": "nie dotyczy",
                            "kod_llm": "", "decyzja": "oznaczenie autora", "kod_koncowy": f"{KOD_REAKCJI}: {r['wartosc']}",
                            "wartosc": r["wartosc"], "weryfikacja": "", "zrodlo": "autor (tabela)", "cytat": r["cytat"], "cytat_status": "cała tura", "memo_llm": "",
                            "memo_autora": f"odpowiedź na pytanie badacza {r['tura_badacza']} ({REAKCJE.name})",
                            "start": r["start"], "koniec": r["koniec"]})
        for k in (k for k in dane["kody_autora"] if k["decyzja"] == "dodany przez autora"):
            wiersze.append({"transkrypt": idi, "etap": dane["etap"], "id": k["id"], "tura_czas": k["czas"], "pojecie": k["pojecie"],
                            "kod_llm": "", "decyzja": k["decyzja"], "kod_koncowy": k["kod"], "wartosc": "", "weryfikacja": k["czas_decyzji"], "zrodlo": "autor", "cytat": k["cytat"],
                            "cytat_status": "autor", "memo_llm": "", "memo_autora": k["memo_autora"], "start": k["start"], "koniec": k["koniec"]})
    if not wiersze:
        sys.exit("Brak kodowań w kodowanie/IDI-*/; najpierw: koduj NN.")
    _write_csv(str(cel / "kody.csv"), wiersze, list(wiersze[0]))
    ok = [w for w in wiersze if w["decyzja"] in ("przyjęty", "zmieniony", "dodany przez autora", "oznaczenie autora")]

    ksiega = defaultdict(lambda: {"liczba": 0, "transkrypty": set(), "pojecia": Counter()})
    for w in ok:
        x = ksiega[(w["etap"], w["kod_koncowy"])]
        x["liczba"] += 1
        x["transkrypty"].add(w["transkrypt"])
        x["pojecia"][w["pojecie"]] += 1
    _write_csv(str(cel / "ksiega-kodow.csv"),
               [{"etap": e, "kod": kod, "liczba": x["liczba"], "transkrypty": len(x["transkrypty"]),
                 "lista_transkryptow": ", ".join(sorted(x["transkrypty"])), "pojecia": ", ".join(f"{p} ({n})" for p, n in x["pojecia"].most_common())}
                for (e, kod), x in sorted(ksiega.items(), key=lambda i: (i[0][0], -i[1]["liczba"], i[0][1]))],
               ["etap", "kod", "liczba", "transkrypty", "lista_transkryptow", "pojecia"])

    naglowek = ["transkrypt", "etap", "model", "data", "propozycje", "przyjęte", "zmienione", "odrzucone", "bez decyzji",
                "cytat przybliżony", "cytat nieznaleziony", "odrzucone automatycznie", "tury wyłączone", "dodane przez autora", "zweryfikowane", "adjudykacja"]
    suma = ["razem", "", "", ""] + [sum(r[i] for r in podsum) for i in range(4, len(naglowek))]
    rozstrz = suma[5] + suma[6] + suma[7]
    md = ["# Podsumowanie decyzji kodowania wspieranego LLM", "", f"Stan na {teraz}. Źródło: kodowanie/log-decyzji.csv i kodowanie/IDI-*/*.json.", "",
          "| " + " | ".join(naglowek) + " |", "|" + "---|" * len(naglowek)]
    md += ["| " + " | ".join(map(str, r)) + " |" for r in podsum + [suma]]
    if rozstrz:
        md += ["", f"Z {rozstrz} rozstrzygniętych propozycji: przyjęte bez zmian {suma[5] / rozstrz:.0%}, zmienione {suma[6] / rozstrz:.0%}, "
                   f"odrzucone {suma[7] / rozstrz:.0%}."]
    md += ["", f"Oznaczenia autora poza propozycjami LLM ({KOD_REAKCJI}): {len(widziane)} tur rozmówców ({REAKCJE.name})."]
    _atomic_write_text(str(cel / "podsumowanie-decyzji.md"), "\n".join(md))

    # REFI-QDA 1.0 (urn:QDA-XML:project:1.0); pozycje to znaki tekstu z końcami linii LF, tak jak w zapisanych źródłach.
    ns = "urn:QDA-XML:project:1.0"
    root = ET.Element("Project", {"xmlns": ns, "name": "AIMCP-IDI kodowanie", "origin": "kodowanie_llm.py", "creationDateTime": teraz})
    uzytkownik = guid("autor")
    ET.SubElement(ET.SubElement(root, "Users"), "User", guid=uzytkownik, name=AUTOR)
    kody_el = ET.SubElement(ET.SubElement(root, "CodeBook"), "Codes")
    kod_guid = {}
    for etap, nazwa_rodzica in (("otwarte", "Kodowanie otwarte"), ("zogniskowane", "Kodowanie zogniskowane"),
                                ("kontrolne", "Kodowanie kontrolne (DeepSeek V4.1 Flash)"),
                                ("oznaczenie", "Reakcja na pojęcie badacza (oznaczenie autora)")):
        nazwy = sorted({w["kod_koncowy"] for w in ok if w["etap"] == etap})
        if not nazwy:
            continue
        rodzic = ET.SubElement(kody_el, "Code", guid=guid("etap", etap), name=nazwa_rodzica, isCodable="false")
        for nazwa in nazwy:
            kod_guid[(etap, nazwa)] = guid("kod", etap, nazwa)
            ET.SubElement(rodzic, "Code", guid=kod_guid[(etap, nazwa)], name=nazwa, isCodable="true")
    zrodla_el = ET.SubElement(root, "Sources")
    for idi in sorted({w["transkrypt"] for w in ok}):
        g = guid("zrodlo", idi)
        (zrodla_dir / f"{g}.txt").write_bytes(teksty[idi].encode("utf-8"))  # LF bez konwersji, inaczej pozycje się przesuną
        src = ET.SubElement(zrodla_el, "TextSource", guid=g, name=f"AIMCP-{idi}", plainTextPath=f"internal://{g}.txt")
        for w in (w for w in ok if w["transkrypt"] == idi):
            sel = ET.SubElement(src, "PlainTextSelection", guid=guid("sel", w["etap"], w["id"]), name=w["cytat"][:60],
                                startPosition=str(w["start"]), endPosition=str(w["koniec"]), creatingUser=uzytkownik,
                                creationDateTime=teraz)
            opis = "; ".join(x for x in (w["memo_autora"], f"[LLM] {w['memo_llm']}" if w["memo_llm"] else "", f"pojęcie: {w['pojecie']}") if x)
            ET.SubElement(sel, "Description").text = opis
            cod = ET.SubElement(sel, "Coding", guid=guid("cod", w["etap"], w["id"]), creatingUser=uzytkownik, creationDateTime=teraz)
            ET.SubElement(cod, "CodeRef", targetGUID=kod_guid[(w["etap"], w["kod_koncowy"])])
    xml_path = cel / "project.xml"
    ET.ElementTree(root).write(xml_path, encoding="utf-8", xml_declaration=True)
    qdpx = Path(_export_qdpx(str(xml_path), str(cel), str(zrodla_dir)))
    qdpx = qdpx.replace(cel / "kodowanie-IDI.qdpx")
    wlasne = sum(w["decyzja"] == "dodany przez autora" for w in wiersze)
    print(f"Eksport: {len(ok) - len(widziane) - wlasne} kodów przyjętych lub zmienionych z {len(wiersze) - len(widziane) - wlasne} propozycji LLM, "
          f"{wlasne} kodów dodanych przez autora i {len(widziane)} oznaczeń ({KOD_REAKCJI}); {len(podsum)} plików kodowania.")
    print(f"  {qdpx.relative_to(P)} (REFI-QDA: MAXQDA, ATLAS.ti, NVivo, QualCoder)")
    for n in ("kody.csv", "ksiega-kodow.csv", "podsumowanie-decyzji.md"):
        print(f"  {(cel / n).relative_to(P)}")


def serwer(port):
    """Strona do weryfikacji propozycji i własnych kodów; tylko 127.0.0.1, zapis tą samą drogą co przegląd w terminalu."""
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from urllib.parse import parse_qs, urlparse
    import webbrowser
    strona = (Path(__file__).parent / "kodowanie.html").read_bytes()
    adres = f"http://127.0.0.1:{port}"

    def otworz(idi, etap):
        if not re.fullmatch(r"IDI-\d{2}", str(idi)) or etap not in ETAPY:
            raise ValueError("nieznany transkrypt albo etap")
        _, _, _, skrot, tury = wczytaj(idi)
        plik = OUT / idi / f"{etap}.json"
        if not plik.exists():
            raise ValueError(f"brak {plik.relative_to(P)}; najpierw: koduj {idi[4:]}")
        dane = wczytaj_kodowanie(plik, skrot, tury)
        if dane is None:
            raise ValueError("transkrypt zmienił się po kodowaniu; zakoduj ponownie z --nadpisz")
        return plik, dane, tury, {t["id"]: t for t in tury}

    def dowody():
        # tury dowodowe artykułu (lista z planu); brak pliku = tryb dowodów niedostępny
        plik = OUT / "dowody-JAL.json"
        if not plik.exists():
            return {}
        return {idi: [x["tura"] for x in v] for idi, v in json.loads(plik.read_text(encoding="utf-8"))["dowody"].items()}

    def lista(etap):
        wynik, kody, dow = [], set(), dowody()
        for plik in sorted(OUT.glob(f"IDI-*/{etap}.json")):
            d = json.loads(plik.read_text(encoding="utf-8"))
            autora = [k for k in d.get("kody_autora", []) if k["decyzja"] == "dodany przez autora"]
            karty, kp = d["kody"], plik.with_name("kontrolne.json")
            if etap == "otwarte" and kp.exists():
                karty = karty + json.loads(kp.read_text(encoding="utf-8"))["kody"]
            tury_dow = set(dow.get(d["transkrypt"], []))
            karty_dow = [k for k in karty if k["tura"] in tury_dow and k["decyzja"] in ("przyjęty", "zmieniony")]
            wynik.append({"transkrypt": d["transkrypt"], "propozycje": len(karty),
                          "czeka": sum(not k.get("weryfikacja") for k in karty), "autora": len(autora),
                          "dowody": len(karty_dow), "dowody_ok": sum(bool(k.get("weryfikacja")) for k in karty_dow)})
            kody |= {k["kod"] for k in karty + autora if k["decyzja"] in ("przyjęty", "zmieniony", "dodany przez autora")}
        return {"transkrypty": wynik, "kody": sorted(kody)}

    class Obsluga(BaseHTTPRequestHandler):
        def odpowiedz(self, obiekt, status=200, typ="application/json; charset=utf-8"):
            tresc = obiekt if isinstance(obiekt, bytes) else json.dumps(obiekt, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", typ)
            self.send_header("Content-Length", str(len(tresc)))
            self.end_headers()
            self.wfile.write(tresc)

        def do_GET(self):
            u = urlparse(self.path)
            etap = parse_qs(u.query).get("etap", ["otwarte"])[0]
            try:
                if u.path == "/":
                    return self.odpowiedz(strona, typ="text/html; charset=utf-8")
                if u.path == "/api/lista":
                    return self.odpowiedz(lista(etap))
                if u.path.startswith("/api/IDI-"):
                    _, dane, tury, _ = otworz(u.path[5:], etap)
                    wyl = set(dane["tury_wylaczone"])
                    kontrolne = None
                    if etap == "otwarte" and (OUT / dane["transkrypt"] / "kontrolne.json").exists():
                        kd = otworz(dane["transkrypt"], "kontrolne")[1]
                        kontrolne = {"model": kd["model"], "kody": kd["kody"]}
                    return self.odpowiedz({**{k: dane[k] for k in ("transkrypt", "etap", "model", "data", "kody", "kody_autora", "reakcje")},
                                           "kontrolne": kontrolne, "tury": [{**t, "wylaczona": t["id"] in wyl} for t in tury],
                                           "dowody": dowody().get(dane["transkrypt"], [])})
                self.odpowiedz({"blad": "nie ma takiej strony"}, 404)
            except ValueError as exc:
                self.odpowiedz({"blad": str(exc)}, 400)

        def do_POST(self):
            # Tylko z tej strony: inna witryna w przeglądarce nie może zmieniać decyzji (CSRF, DNS rebinding).
            if self.headers.get("Origin", adres) != adres or self.headers.get("Host") != adres[7:] \
                    or not self.headers.get("Content-Type", "").startswith("application/json"):
                return self.odpowiedz({"blad": "zapytanie spoza strony kodowania"}, 403)
            try:
                b = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                plik, dane, _, po_id = otworz(b.get("transkrypt"), b.get("etap", "otwarte"))
                kod, cytat, memo = (str(b.get(x) or "").strip() for x in ("kod", "cytat", "memo"))
                start = b.get("start") if isinstance(b.get("start"), int) else None
                if self.path == "/api/decyzja":
                    k = next((k for k in dane["kody"] if k["id"] == b.get("id")), None)
                    if k is None or b.get("decyzja") not in ("przyjęty", "odrzucony"):
                        raise ValueError("nieznana propozycja albo decyzja")
                    t = po_id[k["tura"]]
                    if cytat and cytat not in t["tekst"]:
                        raise ValueError("cytatu nie ma w turze")
                    decyduj(plik, dane, k, t, b["decyzja"] == "odrzucony", kod, cytat, memo, start)
                elif self.path == "/api/wlasny":
                    t = po_id.get(b.get("tura"))
                    if not t or t["mowca"] != "R" or t["id"] in dane["tury_wylaczone"]:
                        raise ValueError("własne kody tylko w turach rozmówcy, które nie są wyłączone")
                    if not kod or not cytat or cytat not in t["tekst"]:
                        raise ValueError("potrzebny kod i fragment tury")
                    k = dodaj_wlasny(plik, dane, t, kod, cytat, memo, start)
                elif self.path == "/api/memo":
                    # samo memo, decyzja bez zmian; wiersz "memo" w dzienniku zachowuje poprzednie wersje
                    k = next((k for k in dane["kody"] + dane["kody_autora"] if k["id"] == b.get("id")), None)
                    if k is None:
                        raise ValueError("nieznana karta")
                    k["memo_autora"] = memo
                    _atomic_write_json(str(plik), dane)
                    zapisz_decyzje(dane, {**k, "decyzja": "memo", "czas_decyzji": teraz()})
                elif self.path == "/api/usun":
                    k = next((k for k in dane["kody_autora"] if k["id"] == b.get("id") and k["decyzja"] == "dodany przez autora"), None)
                    if k is None:
                        raise ValueError("nieznany kod autora")
                    k.update(decyzja="usunięty przez autora", czas_decyzji=teraz())  # wpis zostaje, eksport go pomija
                    _atomic_write_json(str(plik), dane)
                    zapisz_decyzje(dane, k)
                else:
                    raise ValueError("nieznana akcja")
                widok(dane, list(po_id.values()))
                self.odpowiedz({"ok": True, "karta": k})  # strona podmienia jedną kartę zamiast ładować wywiad od nowa
            except (ValueError, json.JSONDecodeError) as exc:
                self.odpowiedz({"blad": str(exc)}, 400)

        def log_message(self, *args):
            pass

    print(f"Kodowanie: {adres} (zatrzymanie: Ctrl+C). Zapis do kodowanie/IDI-NN/*.json i kodowanie/log-decyzji.csv.")
    webbrowser.open(adres)
    try:
        HTTPServer(("127.0.0.1", port), Obsluga).serve_forever()
    except KeyboardInterrupt:
        print("Zatrzymano.")


def sprawdz():
    """Kontrola bez LLM: parser tur na 14 plikach, tabela wyłączeń, lokalizacja cytatów."""
    razem = razem_reakcji = 0
    for f in sorted((P / "public").glob("AIMCP-IDI-*-Transkrypt.md")):
        idi, tekst, informant, _, tury = wczytaj(f.name.split("-")[2])
        assert tury and informant, idi
        assert all(tekst[t["poz"]:t["poz"] + len(t["tekst"])] == t["tekst"] and "**[" not in t["tekst"] for t in tury), idi
        wyl = wylaczone(idi, tury)
        reak = reakcje(idi, tury)
        assert not {r["tura"] for r in reak} & wyl, f"{idi}: tura z oznaczeniem jest też wyłączona"
        razem += len(wyl)
        razem_reakcji += len(reak)
        print(f"{idi}: tur {len(tury)} (R {sum(t['mowca'] == 'R' for t in tury)}), wyłączonych {len(wyl)}, oznaczonych {len(reak)}")
    assert zlokalizuj("Ala ma kota.", "X. Ala ma kota. Y.") == (3, 15, "dokładny")
    assert zlokalizuj("ala ma, kota", "Ala ma kota.")[2] == "nieznaleziony"
    assert zlokalizuj("Ala ma kota ... psa", "Ala, ma kota oraz psa.") == (0, 21, "przybliżony")
    print(f"OK: wyłączonych tur razem {razem}, oznaczonych reakcji {razem_reakcji}; lokalizacja cytatów działa.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # polskie znaki także w Git Bash i przy przekierowaniu
    sys.stdin.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("koduj", help="propozycje kodów LLM dla jednego transkryptu")
    k.add_argument("nr", help="numer wywiadu, np. 62 albo IDI-62")
    k.add_argument("--etap", choices=list(ETAPY), default="otwarte")
    k.add_argument("--nadpisz", action="store_true", help="nowe kodowanie; poprzednie trafia do kodowanie/_archiwum/")
    r = sub.add_parser("przeglad", help="weryfikacja propozycji: przyjęty, zmieniony, odrzucony")
    r.add_argument("nr")
    r.add_argument("--etap", choices=list(ETAPY), default="otwarte")
    sv = sub.add_parser("serwer", help="strona w przeglądarce: weryfikacja propozycji i własne kody")
    sv.add_argument("--port", type=int, default=8765)
    ad = sub.add_parser("adjudykacja", help="decyzje z kodowanie/IDI-NN/adjudykacja.json (trzeci model), ze źródłem i uzasadnieniem")
    ad.add_argument("nr")
    sub.add_parser("eksport", help="REFI-QDA (.qdpx), CSV, księga kodów, podsumowanie decyzji")
    sub.add_parser("sprawdz", help="kontrola parsera i tabeli wyłączeń (bez LLM)")
    a = ap.parse_args()
    if a.cmd == "koduj":
        koduj(a.nr, a.etap, a.nadpisz)
    elif a.cmd == "przeglad":
        przeglad(a.nr, a.etap)
    elif a.cmd == "adjudykacja":
        adjudykacja(a.nr)
    elif a.cmd == "serwer":
        serwer(a.port)
    else:
        {"eksport": eksport, "sprawdz": sprawdz}[a.cmd]()
