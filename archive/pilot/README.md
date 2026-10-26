# Archiwum prób pilotażowych

Ten katalog przechowuje wcześniejsze dane i wyniki. Nie jest częścią
aktywnego procesu walidacji kąta kolana i nie zawiera kodu do uruchamiania.

Podczas porządkowania pliki przeniesiono bez zmiany zawartości. Zgodność
przeniesionych plików sprawdzono sumami SHA-256. Nie ustalano ponownie
pochodzenia poszczególnych wyników ani nie przeliczano ich wartości.

| Zawartość | Poprzednie położenie | Znaczenie |
|---|---|---|
| `app_runs/` | katalog główny | Dawne pliki wejściowe i wyniki aplikacji, film ze szkieletem, logi. |
| `debug_frames/` | katalog główny | Obrazy diagnostyczne wcześniejszej ekstrakcji. |
| `metrics_checks/` | katalog główny | Stare CSV punktów i podsumowania szerszej analizy biegu. |
| `pose_quality_checks/` | katalog główny | Obrazy i podsumowanie kontroli jakości pozy. |
| `angles.csv` | katalog główny | Stary zapis dwóch kątów i czasu. |
| `run_landmarks.csv`, `run_landmarks_smoke.csv` | katalog główny | Stare CSV punktów. |
| `report.md`, `report_smoke.md`, `report_smoke.json` | katalog główny | Raporty wcześniejszej analizy techniki biegu. |
| `pose_quality_frame.jpg` | katalog główny | Pojedynczy obraz diagnostyczny. |

Oryginalne nagrania `run*.mp4` i `run*.MOV` z katalogu głównego znajdują się
teraz w `data/raw/`, pod dotychczasowymi nazwami.

Stare CSV nie mają schematu pakietu `knee_angle_validation`. Nie należy ich
bezpośrednio wczytywać do bieżącej walidacji ani traktować dawnych raportów
techniki biegu jako wyników właściwego badania. Odtworzenie dawnych obliczeń
wymaga wcześniejszej wersji kodu i ustalenia konfiguracji eksperymentu.

Zawartość danych w tym katalogu jest lokalna i ignorowana przez Git.
W repozytorium wersjonowany jest wyłącznie ten opis. Dane wymagają osobnej
kopii zapasowej.
