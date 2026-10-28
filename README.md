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
    annotation_component.py    klikalna anotacja Streamlit
    validation.py              łączenie danych i obliczanie błędów
    statistics.py              statystyki opisowe
    io_utils.py                odczyt, zapis i schemat CSV
    config.py                  progi badawcze
    models.py                  wynik oceny jakości klatki
  tests/                       testy aktualnego pakietu
  data/
    raw/                       oryginalne nagrania
    extracted/                 punkty i kąty MediaPipe
    reference/                 wybór klatek i ręczne anotacje
    results/                   wyniki walidacji i statystyki
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
- Przy ekstrakcji kolejnego filmu podawaj osobną ścieżkę `--csv`.
- Dla osobnych porównań podawaj osobne `--results` i `--summary`.
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
  --csv data\extracted\video_001.csv `
  --person-id person_001 --side left
```

Walidacja poza interfejsem:

```powershell
.\.venv\Scripts\python.exe analyze_run_csv.py `
  --extracted data\extracted\video_001.csv `
  --reference data\reference\reference_annotations.csv `
  --results data\results\video_001_validation.csv `
  --summary data\results\video_001_summary.csv
```

Testy:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Podgląd filmu z punktami

1. Wgraj nagranie w zakładce „Nagranie i ekstrakcja”. Wykonaj ekstrakcję
   albo wczytaj zapisany CSV dotyczący dokładnie tego nagrania.
2. Otwórz zakładkę „Podgląd szkieletu” i zaznacz „Pokaż punkty na filmie”.
3. Sprawdź kadrowanie i wybierz numer klatki suwakiem.
4. Kliknij „Wygeneruj film z punktami”, aby odtworzyć lub pobrać MP4.

Nakładka pokazuje trzy zapisane punkty wybranej nogi: biodro, kolano i kostkę,
połączone odcinkami. Nie odtwarza pełnego szkieletu 33 punktów, ponieważ CSV
zawiera tylko punkty badanej kończyny. Braki nie są interpolowane. Oryginał
i CSV nie są zmieniane, a detekcja MediaPipe nie jest wykonywana ponownie.
Ręczna anotacja pozostaje na obrazie bez nakładki.

Filmy są zapisywane pod unikalnymi nazwami w `data/results/overlays/`,
bez dźwięku, w H.264 i z wysokością do 720 px. FPS pochodzi z CSV;
eksport nie odtwarza zmiennych odstępów czasowych VFR. Sprawdzane są wymiary,
FPS oraz liczba klatek, ale przy imporcie CSV trzeba samodzielnie dopasować
film i kadrowanie. Nakładka służy do kontroli punktów, nie potwierdza ich dokładności.

## Format danych CSV

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

Raportowane są MAE, RMSE, bias, mediana błędu bezwzględnego, odchylenie
standardowe, zakres błędu oraz odsetek obserwacji z błędem do 3, 5 i 10 stopni.
Wyniki można grupować według filmu, osoby i przedziału visibility. Dla biodra,
kolana i kostki obliczany jest błąd pikselowy oraz błąd znormalizowany długością
uda referencyjnego.

Visibility jest analizowane jako cecha modelu, ale nie jest traktowane jako
bezpośrednia miara dokładności punktu.

## Anotacja referencyjna

Klatki są wybierane warstwowo względem visibility i zakresu kąta. W widoku
anotacji wyświetlany jest obraz bez punktów MediaPipe. Anotator wskazuje kolejno
biodro, kolano i kostkę; może cofnąć lub wyczyścić zaznaczenia. Odcinki są
rysowane dopiero po wybraniu wszystkich trzech punktów.

## Ograniczenia

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
- Wybór klatek z wielu warstw może zakończyć się błędem usuwania indeksów.
- Zapis anotacji nie uwzględnia sesji w kluczu zastępowania poprzedniego wpisu.
- Aplikacja używa wspólnych nazw filmów i wyników; trzeba powiązać stan i zapis
  z identyfikatorem nagrania oraz przebiegu analizy.
- Należy zweryfikować synchronizację punktów komponentu anotacyjnego,
  rzeczywiste czasy i orientację klatek oraz obsługę brakujących danych.

Podstawowe testy jednostkowe nie potwierdzają jeszcze poprawności całego
procesu badawczego. Kolejny etap powinien zacząć się od geometrii kąta
i bezpiecznego powiązania danych z nagraniem.
