# Walidacja kąta stawu kolanowego z MediaPipe

Projekt wspiera pracę inżynierską dotyczącą dokładności i wiarygodności
wyznaczania kąta stawu kolanowego podczas biegu na podstawie nagrania z widoku
bocznego. Analizowana jest jedna osoba i jawnie wybrana noga bliższa kamerze.

## Cel badawczy

Główne pytanie brzmi: z jaką dokładnością MediaPipe pozwala wyznaczyć kąt
stawu kolanowego podczas biegu na podstawie nagrania z widoku bocznego?
Projekt porównuje kąt MediaPipe z ręczną referencją i bada błędy punktów,
visibility, stabilność czasową oraz wpływ wygładzania.

System nie diagnozuje techniki biegu, urazów ani chorób i nie jest narzędziem
medycznym.

## Architektura

```text
pose_run/
  app.py                       interfejs Streamlit
  pose_video.py                ekstrakcja z CLI
  analyze_run_csv.py            walidacja z CLI
  plot_angles.py                wykres kątów z CLI
  generate_report.py            tabela statystyk jako HTML
  requirements.txt              zależności aktualnego projektu
  .streamlit/config.toml        wygląd aplikacji
  models/
    pose_landmarker_full.task   model MediaPipe
  src/knee_angle_validation/
    __init__.py                publiczna funkcja kąta
    pose_extraction.py         MediaPipe, wideo i odczyt klatek
    overlay.py                 podgląd punktów i eksport filmu z nakładką
    knee_angle.py              geometria kąta 2D
    smoothing.py               filtry sygnału
    frame_quality.py           heurystyki jakości klatek
    reference_annotation.py    wybór klatek i rekord anotacji
    reference_quality.py       jakość ręcznej referencji i obsługa nieoznaczalnych punktów
    annotation_storage.py      bezpieczny zapis sesji bez zastępowania pomiarów
    provenance.py              identyfikacja filmu, analizy i geometrii obrazu
    video_timing.py            odczyt PTS i kontrola osi czasu CFR/VFR
    experiment_metadata.py     mapowanie tabeli osób, prędkości i stron nagrań
    annotation_component.py    klikalna anotacja Streamlit
    validation.py              łączenie danych i obliczanie błędów
    statistics.py              statystyki opisowe
    io_utils.py                odczyt, zapis i schemat CSV
    config.py                  progi badawcze
    models.py                  wynik oceny jakości klatki
  tests/                       testy aktualnego pakietu
  data/
    raw/                       oryginalne nagrania
    extracted/                 wcześniejsze ekstrakcje (bez automatycznej migracji)
    reference/                 wcześniejsze anotacje (bez automatycznej migracji)
    results/analyses/           nowe analizy, osobny katalog dla każdego analysis_id
  archive/pilot/               lokalne wyniki wcześniejszych prób
```

Projekt zawiera jeden pakiet analityczny: `src/knee_angle_validation`.
Starszy pakiet `running_biomechanics` i jego dwa testy zostały usunięte.
Kadencja, ocena techniki biegu, symetria, analiza bioder, tułowia i ramion
nie należą do aktualnego zakresu.

Nagrania z katalogu głównego przeniesiono bez zmiany nazw do `data/raw/`.
Istniejące wyniki, raporty i obrazy diagnostyczne zachowano w `archive/pilot/`.
Opis archiwum znajduje się w [archive/pilot/README.md](archive/pilot/README.md).
Archiwalne CSV używają starszego schematu i nie są wejściem aktualnego CLI.

## Zasady pracy z plikami

- Polecenia uruchamiaj z katalogu głównego projektu.
- Oryginały nagrań przechowuj w `data/raw/`; wyników nie zapisuj obok kodu.
- Bez `--csv` CLI tworzy osobny katalog analizy; jawna ścieżka nie może już istnieć.
- Bez `--results` i `--summary` każde porównanie CLI dostaje osobny podkatalog.
- Dane badawcze i wyniki są pomijane przez Git; wymagają osobnej kopii zapasowej.
- Środowisko `.venv` oraz ustawienia PyCharma pozostają lokalne.

## Uruchomienie

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\streamlit.exe run app.py
```

Ekstrakcja poza interfejsem:

```powershell
.\.venv\Scripts\python.exe pose_video.py `
  --source data\raw\run13.MOV --video-id video_001 `
  --person-id person_001 --side left
```

Walidacja poza interfejsem:

```powershell
.\.venv\Scripts\python.exe analyze_run_csv.py `
  --extracted data\results\analyses\ANALYSIS_ID\extracted_landmarks.csv `
  --reference data\results\analyses\ANALYSIS_ID\reference_annotations.csv `
  --annotator-id annotator_01 --session IDENTYFIKATOR_SESJI
```

Testy:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Podgląd filmu z punktami

1. Wgraj nagranie w zakładce „Nagranie i ekstrakcja”. Wykonaj ekstrakcję
   albo wczytaj zapisany CSV dotyczący dokładnie tego nagrania.
2. Otwórz zakładkę „Podgląd szkieletu” i zaznacz „Pokaż punkty na filmie”.
3. Wybierz numer klatki suwakiem; kadrowanie pochodzi automatycznie z CSV.
4. Kliknij „Wygeneruj film z punktami”, aby odtworzyć lub pobrać MP4.

Nakładka pokazuje trzy zapisane punkty wybranej nogi: biodro, kolano i kostkę,
połączone odcinkami. Nie odtwarza pełnego szkieletu 33 punktów, ponieważ CSV
zawiera tylko punkty badanej kończyny. Braki nie są interpolowane. Oryginał
i CSV nie są zmieniane, a detekcja MediaPipe nie jest wykonywana ponownie.
Ręczna anotacja pozostaje na obrazie bez nakładki.

Filmy są zapisywane pod unikalnymi nazwami w `data/results/analyses/ANALYSIS_ID/overlays/`,
bez dźwięku, w H.264 i z wysokością do 720 px. FPS pochodzi z CSV;
eksport nie odtwarza zmiennych odstępów czasowych VFR. Sprawdzane są wymiary,
FPS oraz liczba klatek. SHA-256 musi odpowiadać wgranemu filmowi, a kadrowanie
i skalowanie są odczytywane z metadanych CSV. Nakładka służy do kontroli punktów,
nie potwierdza ich dokładności.

## Identyfikacja nagrań i odtwarzanie analizy

Każda nowa ekstrakcja otrzymuje unikalny `analysis_id`, także przy powtórnym
przetworzeniu tego samego filmu. Film wgrany w aplikacji jest kopiowany do
`data/raw/SHA256.mov` (z zachowaniem rozszerzenia). Nazwa pochodzi z zawartości,
więc dwa różne filmy o tej samej nazwie nie zastępują się nawzajem.

```text
data/results/analyses/ANALYSIS_ID/
  extracted_landmarks.csv
  extracted_landmarks.metadata.json
  selected_reference_frames.csv
  reference_annotations.csv
  validation_results.csv
  summary_statistics.csv
  overlays/
  validations/                   osobne porównania uruchomione z CLI
```

Punkty, wybór klatek, nowe anotacje i wyniki porównania zawierają `analysis_id`,
`source_sha256`, `crop_mode` i `resize_height`. Identyfikatory `video_id` oraz
`person_id` podaje użytkownik; SHA-256 rozróżnia fizyczne pliki niezależnie od nazw.
Manifest JSON nowej ekstrakcji zawiera też hash modelu, nazwę pliku źródłowego,
datę, FPS, liczbę klatek, wymiary i ustawienia wygładzania.

Zmiana filmu, osoby, Video ID, strony lub kadrowania czyści bieżące punkty,
wybór klatek, walidację, statystyki, szkice anotacji i podgląd nakładki.
Pliki wcześniejszych analiz pozostają na dysku. Aby wrócić do analizy,
wgraj właściwy film i jego oryginalny `extracted_landmarks.csv`; aplikacja
odnajdzie anotacje w katalogu `analysis_id`. Jeśli CSV był już wybrany przed
zmianą filmu lub ustawień, usuń go z pola importu i wczytaj ponownie.
Zmieniony CSV z tym samym identyfikatorem nie zastąpi zapisanej ekstrakcji.

Anotacja i nakładka wymagają zgodności SHA-256 i używają geometrii zapisanej
w CSV. Walidacja odrzuca pomieszanie nowych danych z inną analizą, filmem,
kadrowaniem lub wysokością skalowania. Statystyki do pobrania pochodzą
z bieżącej walidacji w pamięci, a nie ze wspólnego pliku poprzedniego filmu.
Nazwy pobieranych CSV mają prefiks `analysis_id`.

Starsze CSV bez metadanych można analizować tabelarycznie; nie wolno na ich
podstawie anotować ani nakładać punktów na dowolnie wgrany film. Do takiej pracy
wykonaj nową ekstrakcję. Starszych danych i plików nie migrujemy automatycznie.
Porównanie dwóch starszych CSV nie potwierdza tożsamości nagrania; użytkownik
musi sprawdzić ich pochodzenie. Nowej ekstrakcji nie łączymy ze starą referencją.
Hash identyfikuje plik. Nowa ekstrakcja odczytuje PTS klatek; orientacja obrazu
i poprawność wskazań anatomicznych nadal wymagają osobnej weryfikacji.

## Format danych CSV

### Wybór fragmentu 20 s

Przed ekstrakcją ustaw „Początek analizy w filmie” i „Koniec analizy w filmie”.
Domyślny zakres to **0–20 s**; dla właściwej próby wybierz wcześniej ustalony
fragment stabilnego biegu, np. **5–25 s**. Dla krótszego pilotażu można podać
inny zakres. Zmiana granic unieważnia poprzedni stan; nowa ekstrakcja tworzy
nowy `analysis_id` i zachowuje wcześniejsze pliki.

Zakres jest półotwarty `[początek, koniec)`: klatka o czasie równym końcowi
nie jest analizowana. Przy dokładnym CFR 60 FPS zakres 5–25 s zawiera 1200 klatek o numerach
300–1499. `frame_number` zaczyna się od 0 w pełnym filmie i nigdy nie jest
przenumerowywany po wybraniu fragmentu; `time_s` i znaczniki MediaPipe nadal
odnoszą się do pełnego nagrania. Model otrzymuje wyłącznie klatki z zakresu;
klatki przed początkiem są odczytywane sekwencyjnie bez detekcji, aby zachować
źródłowe indeksy. Wygładzanie i ocena jakości dotyczą wybranego odcinka.

`analysis_start_s` i `analysis_end_s` są zapisywane w punktach, wyborze klatek,
anotacjach, wynikach, statystykach i manifeście JSON. Walidacja wymaga zgodności
zakresu między nowymi punktami i referencją. Wybór 30 klatek działa na tym
fragmencie, z zachowaniem źródłowych numerów i czasów. Anotacja czyta wskazaną
klatkę pełnego filmu; eksport MP4 obejmuje tylko analizowany fragment i nadal
wyświetla źródłowy numer klatki na nakładce.

Zakres pusty, odwrócony, ujemny lub niekompletny (np. 20 s z filmu krótszego
niż 20 s) jest odrzucany bez zapisu częściowego CSV. W takim przypadku ustaw
krótszy, prawidłowy zakres. Niepoprawne FPS również zatrzymuje ekstrakcję;
nie podstawiamy domyślnego 30 FPS. Nowe analizy korzystają z rzeczywistych PTS
zgodnie z opisem poniżej; liczba klatek w 20 s może zależeć od VFR.

W CLI `--start` wskazuje początek, a `--end` wyłączony koniec. Bez `--end`
koniec wynosi `--start + 20 s`:

```powershell
.\.venv\Scripts\python.exe pose_video.py --source data\raw\IMG_1302.MOV --start 5 --end 25
```

Niskopoziomowe `process_video()` zachowuje zgodność ze starszym wywołaniem:
domyślne `end_time_s=None` oznacza odczyt do końca filmu. Interfejs i CLI
przekazują granice jawnie. Starsze CSV bez pól zakresu nie są automatycznie
przycinane; nową próbę przygotuj przez ponowną ekstrakcję.

### Czas filmu i VFR

Przed ekstrakcją FFmpeg z istniejącego `imageio-ffmpeg` odczytuje PTS wszystkich
klatek pierwszego strumienia wideo. Używany jest filtr
[`showinfo`](https://ffmpeg.org/ffmpeg-filters.html#showinfo); plik źródłowy nie
jest konwertowany. Ten dodatkowy przebieg dekodowania wydłuża przygotowanie analizy.
Nie instalujemy dodatkowego FFmpeg ani biblioteki.

- `source_pts_s`: odczytany PTS klatki w sekundach, z zachowaniem początku strumienia.
- `time_s` i `time_pts_s`: PTS minus PTS pierwszej klatki pełnego filmu; początek filmu to 0 s.
- `time_nominal_s`: pomocniczy `frame_number / fps`, do kontroli różnicy względem PTS.
- `mediapipe_timestamp_ms`: rzeczywisty czas względny zaokrąglony do ms, przekazany modelowi.
- `time_source=ffmpeg_pts_v1`: oznaczenie nowego sposobu wyznaczania czasu.

Zakres `[start, end)` jest wybierany według `time_s`, również przy VFR.
`analysis_first_frame` i `analysis_stop_frame` zapisują źródłowe granice numerów
klatek (koniec wyłączony); służą też do eksportu nakładki bez przeliczania
zakresu przez nominalny FPS. Wybór 30 klatek nadal korzysta wyłącznie z czasu.
Anotacja czyta klatki kolejno od początku filmu, aby zachować ich indeksy
także przy VFR; odczyt odległej klatki może potrwać dłużej niż przewijanie.
Referencja zachowuje PTS, czas i sposób jego wyznaczania. Walidacja sprawdza
zgodność czasów dla tych samych kluczy klatek i zachowuje czas w wynikach.

Manifest zawiera `timing_audit`: liczbę klatek, podstawę czasu, pierwszy PTS,
różnice względem nominalnego FPS i klasyfikację CFR lub VFR/niezgodności FPS.
Klasyfikacja jest diagnostyczna: przy VFR używamy PTS bez usuwania klatek,
interpolacji ani dopasowywania czasu do referencji. Tolerancja klasyfikacji
uwzględnia kwantyzację PTS i nie przekracza 1 ms ani 5% okresu klatki.

Brakujące, nierosnące lub nieprawidłowe PTS, rozbieżność odczytu OpenCV z osią
PTS i kolizje znaczników po zaokrągleniu do ms zatrzymują analizę bez zapisu
częściowych wyników. Końcowy czas całego filmu korzysta z czasu trwania
ostatniej klatki podanego przez dekoder; jeśli go brak, ostatni okres jest
szacowany z FPS, co manifest jawnie oznacza. Nie zmienia to PTS samych klatek.

Eksport nakładki pozostaje podglądem CFR, więc nie odtwarza nierównych odstępów
VFR. Właściwą osią pomiaru są CSV z PTS. Starszy CSV bez `time_source` nie
zostaje automatycznie poprawiony: aplikacja ostrzega, że potrzebna jest nowa
ekstrakcja. Nie łączymy nowych wyników PTS ze starszą referencją o innej osi czasu.

### Metadane eksperymentu

Tabela `data/metadata/videos_metadata.csv` jest odczytywana z separatorem `;`.
Obecny zbiór obejmuje 10 nagrań, 5 osób i prędkości 7.5 oraz 8.5 km/h.
Dopasowanie korzysta z oryginalnej nazwy filmu, bez rozróżniania wielkości liter:

| Pole tabeli | Pole analizy | Mapowanie |
|---|---|---|
| `original_filename` | `original_filename` | Oryginalna nazwa, zachowana mimo kopii filmu pod SHA-256 |
| `participant_id` | `person_id` | Identyfikator tekstowy, np. `P01` |
| `speed_kmh` | `speed_kmh` | Dodatnia liczba w km/h; akceptowany także przecinek dziesiętny |
| `analyzed_side` | `side` | `RIGHT/LEFT` → `right/left` |
| `inclusion_status` | `inclusion_status` | Ekstrakcja badawcza tylko dla `INCLUDED` |
| `notes` | `metadata_notes` | Notatka dotycząca nagrania, oddzielna od notatki anotatora |

Po wgraniu filmu z tabeli aplikacja uzupełnia osobę, stronę i prędkość;
te pola są zablokowane do edycji, aby uniknąć sprzeczności z tabelą. Korekty
wprowadź w tabeli przed nową ekstrakcją. Dla filmu spoza tabeli wpisz osobę,
wybierz stronę i opcjonalnie podaj prędkość. Nieznana prędkość pozostaje pusta;
aplikacja nie przypisuje jej wartości 0 ani prędkości poprzedniego nagrania.
Zmiana metadanych wejściowych unieważnia poprzedni stan analizy.

Nowe punkty, wybrane klatki, anotacje i wyniki zachowują te metadane.
Manifest JSON i statystyki zapisują również `metadata_table_sha256`, czyli
sumę SHA-256 wersji tabeli użytej podczas ekstrakcji (pustą dla danych ręcznych).
Zapisane rekordy stanowią migawkę metadanych; późniejsza edycja tabeli nie
zmienia historycznych wyników. Walidacja blokuje referencję z inną migawką
metadanych, np. z przypisaną inną prędkością. Starszych CSV nie uzupełniamy
automatycznie; do nowego schematu wykonaj nową ekstrakcję i anotację.

Powielone nazwy, brak osoby/prędkości, niepoprawna strona lub status w tabeli
powodują błąd. `EXCLUDED` i `PENDING` blokują ekstrakcję. Tabela opisuje dane
eksperymentu; poprawność przypisania nazwy do osoby trzeba sprawdzić podczas
przygotowywania nagrań. Podana prędkość dotyczy ustawienia bieżni.

CLI również korzysta z tabeli. Dla znanego nagrania wystarczy:

```powershell
.\.venv\Scripts\python.exe pose_video.py --source data\raw\IMG_1302.MOV
```

Opcja `--metadata` wskazuje inną tabelę w tym samym formacie.
Przy kopii filmu nazwanej sumą SHA-256 podaj dodatkowo
`--original-filename IMG_1302.MOV`. Dla nagrań spoza tabeli wymagane są
`--person-id` i `--side`; opcjonalnie podaj `--speed-kmh`.
Sprzeczne wartości argumentów dla filmu z tabeli są odrzucane.

### Geometria i konwencja kąta

Obie metody obliczają kąty z pikseli tego samego obrazu analitycznego
(po tym samym kadrowaniu i skalowaniu). Nie obliczamy kąta bezpośrednio
z `x_norm, y_norm`, ponieważ osie te są normalizowane różnymi wymiarami.

- `knee_internal_angle_deg`: surowy kąt wewnętrzny biodro–kolano–kostka,
  około 180° przy wyproście.
- `knee_flexion_deg`: surowe zgięcie 2D, `180° - knee_internal_angle_deg`,
  około 0° przy wyproście. To główna miara na wykresach i w walidacji.
- `knee_flexion_smoothed_deg`: zgięcie po wybranym filtrze, oddzielone od surowego.
- `angle_geometry = pixel_2d_v1`: oznaczenie poprawionej geometrii.
- Dotychczasowe `knee_angle_raw`, `knee_angle_smoothed` i `knee_angle_reference`
  pozostają aliasami **kąta wewnętrznego**, aby nie zmieniać ich znaczenia.

Anotacja referencyjna zapisuje oba surowe kąty tymi samymi nazwami. Walidacja
ponownie oblicza je z zapisanych pikseli obu metod; wspólne nazwy uzyskują
sufiksy `_mediapipe` i `_reference`. `angle_mediapipe`, `angle_reference`
oraz `error_signed` odnoszą się do **zgięcia**; dodatni błąd oznacza większe
zgięcie MediaPipe. `error_internal_signed` zachowuje różnicę kątów wewnętrznych
(przeciwny znak). MAE i RMSE dla obu konwencji są takie same.

Brak punktu, NaN/inf lub długość odcinka do `1e-6` piksela daje brak kąta,
nie 0°. Pomiar jest bez znaku i nie rozróżnia przeprostu.

Starszy CSV z punktami pikselowymi można wczytać do aplikacji: kąty oraz
cechy jakości zostaną przeliczone w pamięci, a wcześniejszy wybór klatek
i walidacja w sesji zostaną wyzerowane. Poprawione dane pobierz z zakładki
„Eksport”. Plik źródłowy nie jest nadpisywany. Nie porównuj nowych wyników
z dawnymi tabelami i wykresami wyliczonymi przed poprawką.

`extracted_landmarks.csv` zawiera identyfikatory filmu i osoby, numer i czas
klatki, FPS, wymiary obrazu, stronę, współrzędne znormalizowane i pikselowe oraz
visibility biodra, kolana i kostki. Zawiera również surowy i wygładzony kąt,
minimalne i średnie visibility, cechy stabilności, status jakości oraz powody
oznaczenia klatki.

`reference_annotations.csv` przechowuje trzy ręcznie kliknięte punkty,
identyfikator anotatora i sesji, notatki oraz kąt obliczony dokładnie tą samą
funkcją `calculate_knee_angle()`.

Łączenie odbywa się po `video_id`, `person_id`, `frame_number` i `side`.

## Metryki

### Wygładzanie i brakujące pomiary

Podstawą walidacji pozostaje **surowy kąt wyliczony z punktów pikselowych**.
Wynik filtrowania jest zapisywany oddzielnie i służy do pomocniczego porównania
przebiegu sygnału. Doboru filtra ani okna nie optymalizujemy pod zgodność
z ręczną referencją. Wygładzanie może zmieniać minima, maksima i położenie
ekstremów — nie traktujemy przefiltrowanych ekstremów jako wartości surowych.

Wszystkie trzy filtry zachowują pozycje braków: NaN/inf na wejściu daje NaN
na wyjściu. Każdy ciągły fragment poprawnych danych jest filtrowany osobno.
Nie ma interpolacji przez lukę ani wpływu danych z drugiej strony luki,
także w filtrze wielomianowym. Pusty sygnał, same braki i sygnał krótszy od
okna zachowują dokładnie długość wejścia. Wykres przerywa linię na brakach.

- Mediana i średnia używają wyśrodkowanego okna, skróconego na granicach fragmentu.
- Filtr `savgol` dopasowuje lokalny wielomian tylko do danych w tym samym
  fragmencie; krótkie fragmenty bez wystarczającej liczby próbek pozostawia
  bez zmian. Na brzegach zmniejsza okno i w razie potrzeby stopień wielomianu.
- Okno musi być dodatnią, nieparzystą liczbą całkowitą. Jest liczone w klatkach,
  więc przy VFR nie oznacza stałej szerokości w sekundach. To pomocnicze
  filtrowanie próbek, a nie filtr z ustaloną częstotliwością odcięcia w Hz.

Nowe CSV i manifest zapisują `smoothing_method`, `smoothing_window` oraz
`smoothing_gap_policy=finite_segments_v1`. Interfejs aktualizuje te pola
przy zmianie ustawień; nie nadpisuje źródłowego CSV. Surowe współrzędne i kąty
pozostają zachowane. Starsze kolumny wygładzone mogły wypełniać braki — do
nowego porównania należy je przeliczyć z surowych danych.

CLI `plot_angles.py` domyślnie pokazuje surowy sygnał. Opcjonalny filtr jest
zawsze liczony ponownie, bez korzystania ze starej kolumny wygładzonej:

```powershell
.\.venv\Scripts\python.exe plot_angles.py --csv data\results\analyses\ANALYSIS_ID\extracted_landmarks.csv --smoothing median --window 5
```

Raportowane są MAE, RMSE, bias, mediana błędu bezwzględnego, odchylenie
standardowe, zakres błędu oraz odsetek obserwacji z błędem do 3, 5 i 10 stopni.
Wyniki można grupować według filmu, osoby i przedziału visibility. Dla biodra,
kolana i kostki obliczany jest błąd pikselowy oraz błąd znormalizowany długością
uda referencyjnego.

Visibility jest analizowane jako cecha modelu, ale nie jest traktowane jako
bezpośrednia miara dokładności punktu.

## Anotacja referencyjna

Próba główna jest wybierana wyłącznie według czasu, niezależnie od kąta,
visibility i powodzenia detekcji MediaPipe. Domyślnie wyznaczanych jest
30 środków równych przedziałów w zakresie wskazanym w zakładce „Wybór klatek”.
Dla każdego znacznika wybierana jest najbliższa klatka z tego zakresu;
przy remisie wybierana jest wcześniejsza. Numery klatek i ich czasy pozostają
źródłowe, bez przenumerowania po wybraniu fragmentu.

Jeżeli kilka znaczników wskazuje tę samą klatkę, zachowywana jest jedna
obserwacja z najbliższym znacznikiem (wcześniejszym przy remisie). Nie dobieramy
zastępczych klatek. Dla bardzo krótkiego lub rzadko próbkowanego fragmentu
wynik może zawierać mniej niż 30 klatek — aplikacja pokazuje ostrzeżenie.
Także klatki bez pozy, z NaN kąta lub niskim visibility pozostają w próbie.

`selected_reference_frames.csv` zachowuje dane źródłowe oraz pola
`selection_method`, `selection_index`, `selection_target_time_s`,
`selection_time_error_s` (czas klatki minus znacznik),
`selection_requested_count`, `selection_start_s` i `selection_end_s`.
Wybór jest deterministyczny i nie wymaga ziarna losowania. Algorytm korzysta
z `time_s` w CSV, także przy nierównych odstępach; w nowych ekstrakcjach
kolumna ta pochodzi z rzeczywistych PTS klatek.

W widoku anotacji wyświetlany jest obraz bez punktów MediaPipe. Anotator wskazuje kolejno
biodro, kolano i kostkę; może cofnąć lub wyczyścić zaznaczenia. Odcinki są
rysowane dopiero po wybraniu wszystkich trzech punktów.

## Sesje anotacji i powtórne pomiary

### Niepewność ręcznej referencji

Przed zapisem oceń osobno biodro, kolano i kostkę:

| Flaga | Znaczenie | Wymagane kliknięcie |
|---|---|---|
| `CLEAR` | Punkt dobrze widoczny | Tak |
| `UNCERTAIN` | Położenie trudne do pewnego ustalenia | Tak, jeśli można je oszacować |
| `UNMARKABLE` | Nie można wiarygodnie wskazać punktu | Nie |

Komponent prosi o kliknięcia wyłącznie dla punktów innych niż `UNMARKABLE`,
w kolejności biodro–kolano–kostka z pominięciem punktów nieoznaczalnych.
Zmiana zestawu wymaganych punktów otwiera nowy szkic kliknięć, aby np. punkt
kostki nie został zapisany jako kolano. Zmiana samej flagi CLEAR/UNCERTAIN
zachowuje współrzędne. Flagi i notatki szkicu są przypisane do klatki oraz sesji.

Jeżeli wszystkie punkty są `UNMARKABLE`, można zapisać klatkę bez żadnego
kliknięcia. Pozostaje ona w CSV; nie zastępujemy jej inną klatką. Współrzędne
nieoznaczalnego punktu i kąt referencyjny są puste (`NaN` w obliczeniach),
nigdy równe 0. Pozostałe wskazane punkty są zachowywane. Notatka pozwala
opisać przyczynę braku pomiaru. Pełna anotacja wymaga poprawnych, różnych
punktów pozwalających obliczyć kąt; flagi nie służą do akceptowania błędnej geometrii.

CSV zapisuje `hip_quality`, `knee_quality`, `ankle_quality` oraz `frame_quality`.
Ocena klatki jest wyliczana: UNMARKABLE ma pierwszeństwo, następnie UNCERTAIN;
CLEAR wymaga trzech punktów CLEAR. Ręczna ocena nie jest kopiowana z visibility
ani z jakości detekcji MediaPipe.

Walidacja zachowuje także wiersze UNMARKABLE. Główne statystyki
(`analysis_subset=primary_clear`) obejmują kompletne pary z referencją CLEAR.
Analiza wrażliwości (`sensitivity_clear_uncertain`) obejmuje CLEAR + UNCERTAIN.
Wykresy błędów i zależności od visibility w widoku statystyk dotyczą CLEAR.
Nie należy interpretować dobrego wyniku samego CLEAR jako dokładności dla
wszystkich klatek — trzeba podać również wyniki analizy wrażliwości i liczebności.

Tabela jakości raportuje liczby i odsetki CLEAR/UNCERTAIN/UNMARKABLE oraz liczbę
par z poprawnym pomiarem obu metod. Jest zapisywana jako
`reference_quality_summary.csv` i dostępna do pobrania. CLI zapisuje analogiczny
plik obok podsumowania, z końcówką `_reference_quality.csv`.
Liczebności dotyczą wierszy referencji dopasowanych po kluczu klatki; nie są
jeszcze raportem kompletności całej próby ani brakujących kluczy po łączeniu.

Starsze anotacje bez flag otrzymują `UNASSESSED`, a nie automatyczne CLEAR.
Ich wyniki są raportowane osobno jako `legacy_unassessed`; nie trafiają do
głównej analizy ani analizy wrażliwości. Nie nadpisujemy wcześniejszych pomiarów;
ponowna anotacja z oceną jakości wymaga nowej sesji. Przy dopisywaniu do starego
CSV jego wcześniejsze wiersze pozostają jawnie nieocenione.

W zakładce „Ręczna anotacja” wybierz anotatora i identyfikator sesji.
Przycisk „Nowa sesja pomiarowa” tworzy unikalny identyfikator, także przy
kilku sesjach tego samego dnia. Nowa sesja zaczyna się bez wcześniejszych
punktów i notatek. Powrót do rozpoczętej sesji i klatki przywraca jej szkic
w bieżącym połączeniu Streamlit; po zamknięciu połączenia niezapisany szkic
może zostać utracony. Do trwałego zachowania pomiaru użyj „Zapisz anotację”.

Zapis dopisuje rekord z pełnym kluczem:
`video_id, person_id, frame_number, side, annotator_id, annotation_session`.
Nie zastępuje istniejącego pomiaru o takim samym kluczu — kolejny pomiar lub
korekta wymaga nowej sesji. Zapis odbywa się atomowo, a blokada pliku zapobiega
równoczesnym zapisom. Błąd zapisu nie powinien zastąpić poprzedniego CSV.
Jeżeli proces zostanie przerwany i pozostawi plik `.lock`, przed jego ręcznym
usunięciem sprawdź, że żaden zapis nie jest w toku.

Anotacje zawierają również datę zapisu, wymiary obrazu i SHA-256 wyświetlanego
PNG. Klucz szkicu uwzględnia cały klucz pomiaru i obraz, więc punkty nie są
przenoszone między stronami, osobami, sesjami ani wersjami obrazu.
Obraz anotacyjny jest PNG bez punktów MediaPipe. Punkty są przeliczane
z rozmiaru wyświetlanego obrazu na jego pełną rozdzielczość.

W zakładce „Walidacja” wybierz **jednego anotatora i jedną sesję** referencyjną.
Zmiana referencji unieważnia poprzedni wynik w sesji aplikacji. W CLI przy
wielu sesjach podaj `--annotator-id annotator_01 --session IDENTYFIKATOR`;
bez jednoznacznego wyboru program zatrzyma się przed zapisaniem wyników.
Nie traktujemy powtórnych anotacji jako dodatkowych niezależnych klatek.

Łączenie w `validation.merge_with_reference` wymaga relacji **one-to-one**
po `video_id, person_id, frame_number, side`. Powielony klucz w ekstrakcji
(także poza próbą referencyjną) lub w wybranej referencji zatrzymuje analizę
z komunikatem wskazującym źródło i przykładowe klucze. Rekordy nie są
automatycznie usuwane ani uśredniane. Puste identyfikatory, nieprawidłowa strona
i niecałkowite/ujemne numery klatek również blokują porównanie.
Kontrola działa także przy bezpośrednim wywołaniu funkcji: przy wielu parach
anotator/sesja wymagany jest wybór argumentami `annotator_id` i `session`.
Referencja zawierająca tylko jedną parę jest jednoznaczna bez tych argumentów.
Starszy CSV bez pól anotatora lub sesji wymaga ich świadomego uzupełnienia.
Po błędzie aplikacja unieważnia poprzedni wynik w pamięci i nie zapisuje
nowego CSV; istniejące pliki wynikowe pozostają bez zmian.
Łączenie nadal jest typu `inner`: zawiera wyłącznie pasujące klucze.
Sam brak duplikatów nie potwierdza kompletności pokrycia klatek referencyjnych.

Testy Python obejmują zachowanie sesji, ochronę przed zastąpieniem wpisu,
awarię zapisu i przepływ stanu Streamlit. Testy JavaScript można uruchomić
poleceniem `node --test tests/test_annotation_component.mjs` — sprawdzają
kliknięcia, odtwarzanie stanu, cofanie, czyszczenie i skalowanie na imitacji DOM.
Nie zastępują testu układu i interakcji w rzeczywistej przeglądarce.

## Ograniczenia badania

- Analiza jest dwuwymiarowa i dotyczy wyłącznie widoku bocznego.
- Perspektywa, obrót kończyny i zasłonięcia mogą zmieniać wynik.
- MediaPipe visibility nie zastępuje błędu względem ręcznej referencji.
- Ręczna anotacja także ma niepewność; warto stosować wielu anotatorów lub
  powtórzone sesje i analizować zgodność między nimi.
- Progi wiarygodności w konfiguracji są początkowe i powinny zostać
  skalibrowane na podstawie zebranych wyników, a nie interpretowane jako normy.

## Stan po uporządkowaniu repozytorium

Porządkowanie struktury nie obejmowało napraw algorytmów. Przed właściwym
badaniem pozostają m.in. następujące problemy wykryte podczas przeglądu:

- Poprawiono geometrię: ekstrakcja i referencja używają pikseli; zapisywane są
  osobno kąt wewnętrzny i zgięcie kolana. Wykresy i walidacja używają zgięcia.
- Wybór klatek jest czasowy i niezależny od wyników modelu; usunięto wcześniejszy
  algorytm warstwowy powodujący błąd indeksów.
- Poprawiono zapis sesji: rekordy są dopisywane, a powtórzenie tego samego
  klucza jest blokowane bez zastępowania wcześniejszego pomiaru.
- Nowe analizy mają osobne katalogi i powiązanie filmu, punktów oraz anotacji
  przez SHA-256, identyfikator analizy i geometrię obrazu. Zmiana wejścia czyści stan.
- Należy zweryfikować synchronizację punktów komponentu anotacyjnego,
  orientację klatek oraz obsługę brakujących danych. Nowe ekstrakcje zapisują PTS.

Podstawowe testy jednostkowe nie potwierdzają jeszcze poprawności całego
procesu badawczego. Kolejne etapy powinny objąć raportowanie kompletności
dopasowania, orientację klatek i obsługę brakujących danych.
