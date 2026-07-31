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
src/knee_angle_validation/
  pose_extraction.py       MediaPipe, film i kanoniczny format klatek
  knee_angle.py            wspólna geometria MediaPipe/referencja
  smoothing.py             mediana, średnia krocząca, Savitzky-Golay
  frame_quality.py         visibility, skoki i jakość pojedynczej klatki
  reference_annotation.py wybór klatek i rekord anotacji
  annotation_component.py klikalna anotacja Streamlit CCv2
  validation.py            łączenie danych i błędy punktów/kąta
  statistics.py            MAE, RMSE, bias, grupy i korelacje
  io_utils.py              zapis, odczyt i kontrola schematu CSV
  config.py                konfigurowalne progi badawcze
app.py                     osiem widoków procesu badawczego
data/                      raw, extracted, reference, results
legacy/                    opis zachowanej warstwy zgodności
```

Dotychczasowy pakiet `src/running_biomechanics` pozostaje wyłącznie dla
odtwarzalności wcześniejszych eksperymentów. Nowa aplikacja go nie importuje.

## Uruchomienie

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\streamlit.exe run app.py
```

Ekstrakcja poza interfejsem:

```powershell
.\.venv\Scripts\python.exe pose_video.py `
  --source data\raw\run.mp4 --video-id video_001 `
  --person-id person_001 --side left
```

Walidacja poza interfejsem:

```powershell
.\.venv\Scripts\python.exe analyze_run_csv.py `
  --extracted data\extracted\extracted_landmarks.csv `
  --reference data\reference\reference_annotations.csv
```

Testy:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Format danych

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
