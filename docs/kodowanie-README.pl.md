# Narzędzie do kodowania wstępnego wywiadów ze wsparciem LLM: instrukcja

Model językowy proponuje kody, a badacz weryfikuje każdy (przyjęty, zmieniony, odrzucony). Metoda: `METHODOLOGY.md`; instrukcje dla modeli: `prompts.md`; adjudykacja: `adjudication.md`.

## Uruchomienie

Polecenia z katalogu głównego repozytorium (Python 3.14, `pip install -r requirements-coding.txt`):

```
python scripts/kodowanie_llm.py sprawdz
python scripts/kodowanie_llm.py koduj 62
python scripts/kodowanie_llm.py koduj 62 --etap kontrolne
python scripts/kodowanie_llm.py serwer
python scripts/kodowanie_llm.py przeglad 62
python scripts/kodowanie_llm.py adjudykacja 62
python scripts/kodowanie_llm.py eksport
```

`sprawdz` kontroluje konfigurację bez wywołań modelu: parser tur we wszystkich plikach `public/`, tabelę wyłączeń i lokalizację cytatów. `kodowanie.cmd` (Windows) uruchamia stronę do weryfikacji.

## Model i dostawca

Kodowanie otwarte: Claude Haiku 5.5 (`claude-haiku-5-5`) przez OpenCode Go (`https://opencode.ai/inference/go/v1/messages`, SDK `anthropic`). Kodowanie kontrolne: DeepSeek V4.1 Flash (`deepseek-v4.1-flash`) przez ten sam serwis (`/v1/chat/completions`, klient `vendor/slr_core`). Klucz API: `OPENCODE_GO_API_KEY` w zmiennej środowiskowej albo w `kodowanie/.env` (poza kontrolą wersji). Każde uruchomienie `koduj` wysyła własny identyfikator sesji (`x-opencode-session`), zapisywany w pliku wyniku. Parametry obu modeli: `METHODOLOGY.md`, sekcja 5. Wynik nie jest powtarzalny: ponowne `koduj` daje inne propozycje, więc podstawą analizy jest zweryfikowany plik, nie ponowne uruchomienie.

## Co trafia do modelu

Wyłącznie tekst z `public/AIMCP-IDI-NN-Transkrypt.md`: opis rozmówcy z nagłówka i tury rozmowy. Tury badacza są kontekstem i nie są kodowane. Tury rozmówców z tabeli `config/wylaczone-fragmenty.md` (kategorie A i B) trafiają do modelu jako „(treść pominięta)”; czas z tabeli, który nie wskazuje jednoznacznie tury rozmówcy, zatrzymuje skrypt.

## Przebieg

1. **koduj**: transkrypt jest dzielony na fragmenty według tur (ok. 12 tys. znaków, pytanie badacza zostaje z odpowiedzią). Dla każdej tury do kodowania model zwraca od 0 do 3 propozycji: kod (rzeczownik odczasownikowy albo kod in vivo w cudzysłowie), dosłowny cytat, pojęcie uwrażliwiające (PB1-PB4, W1-W7 albo „brak”) i jednozdaniowe memo. Skrypt odrzuca propozycje dla tur spoza listy i odszukuje cytat w turze: „dokładny”, „przybliżony” (różnice interpunkcji, wielokropek) albo „nieznaleziony” (zaznaczona cała tura, cytat trzeba poprawić przy weryfikacji).
2. **serwer** albo **przeglad**: weryfikacja; oba zapisują to samo. W terminalu (`przeglad`) klawisze: `p` przyjmij, `m` przyjmij z własnym memo, `z` zmień (kod, cytat, memo), `o` odrzuć (z powodem), `s` pomiń, `q` zakończ.
3. **adjudykacja**: przenosi decyzje z `kodowanie/IDI-NN/adjudykacja.json` do kart (format: `adjudication.md`); wymaga decyzji dla każdej karty i pomija karty z decyzją badacza.
4. **eksport**: projekt REFI-QDA `kodowanie/eksport/kodowanie-IDI.qdpx`, tabele `kody.csv`, `ksiega-kodow.csv` i podsumowanie decyzji.

## Pliki wyników (w `kodowanie/`, poza kontrolą wersji)

| Plik | Zawartość |
|---|---|
| `IDI-NN/otwarte.json`, `kontrolne.json` | propozycje modelu, decyzje, pozycje cytatów, model, endpoint, data, zużycie tokenów, pełna instrukcja (`instrukcja`), skrót pliku źródłowego |
| `IDI-NN/*.md` | widok do czytania: tury rozmówcy z kodami i stanem decyzji |
| `log-decyzji.csv` | dziennik decyzji (czas, kod modelu, decyzja, kod końcowy, zmiana cytatu, memo, model) |
| `eksport/` | projekt REFI-QDA i tabele |
| `_logi/llm.log` | zapytania i odpowiedzi modeli |
| `_archiwum/` | poprzednie wersje po `koduj --nadpisz` |

## Strona do weryfikacji

Lokalny serwer (tylko 127.0.0.1:8765, bez dostępu z sieci i bez wywołań modelu) pokazuje cały transkrypt: tury badacza na szaro, przy turach rozmówcy karty propozycji. W etapie otwartym są dwie kolumny: po lewej Claude Haiku 5.5 i kody badacza, po prawej DeepSeek V4.1 Flash; po wybraniu karty kody drugiego modelu na nakładającym się fragmencie mają przerywaną ramkę. Panel na dole dotyczy aktywnej karty: model, kod, stan, pole memo.

- Klawisze poza polem tekstowym: `p` przyjmij, `o` odrzuć, `z` zmień, `m` pole memo, `s` zapisz samo memo, `j`/`k` następna/poprzednia karta, `n` następna do weryfikacji, Esc zamyka edycję.
- W polu memo: Enter przyjmuje z memo, Shift+Enter nowa linia, Ctrl+S zapisuje samo memo.
- `z` zmień: nowa nazwa kodu (podpowiedzi z przyjętych kodów) i nowy cytat przez zaznaczenie fragmentu w tej samej turze.
- Własny kod: zaznacz fragment w turze rozmówcy, wpisz kod i memo, Enter. „Usuń” oznacza kod badacza jako usunięty; wpis zostaje w pliku i w dzienniku.
- Filtr „tylko do weryfikacji” ukrywa zweryfikowane karty. Filtr „tylko dowody” zostawia tury z listy `kodowanie/dowody-JAL.json` (lista identyfikatorów tur używanych jako dowód w publikacji) wraz z poprzedzającym pytaniem badacza.

Decyzję można zmienić później (nowy wiersz w dzienniku). Nie używaj strony i terminala naraz dla tego samego transkryptu: ostatni zapis wygrywa.

## Oznaczenia odpowiedzi na pojęcia badacza

Narzędzie obsługuje opcjonalną tabelę `kodowanie/reakcje-na-pojecie-badacza.csv` (kolumny `transkrypt`, `tura_badacza`, `tura_rozmowcy`, `reakcja`), która oznacza odpowiedzi na pytania sugerujące jako „reakcja na pojęcie badacza”. Tabela z badania nie jest publikowana; bez niej narzędzie działa bez oznaczeń.

## Kodowanie zogniskowane

Lista badacza w `kodowanie/kody-zogniskowane.md` (linie `- kod: definicja`), potem `koduj NN --etap zogniskowane` i weryfikacja. Model przypisuje kody z listy; ważny fragment bez pasującego kodu dostaje propozycję z przedrostkiem „NOWY: ”.

## Zmiana transkryptu

Pozycje cytatów odnoszą się do konkretnej wersji pliku z `public/` (skrót SHA-256 w pliku wyniku). Jeśli plik zmieni się po kodowaniu, `przeglad` i `eksport` go pominą; potrzebne jest `koduj NN --nadpisz`.
