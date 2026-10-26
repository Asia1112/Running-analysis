# Protokół nagrań i organizacji danych

**Projekt:** Analiza kinematyki kończyny dolnej podczas biegu z wykorzystaniem MediaPipe Pose  
**Wersja:** 1.0 — ustalenia etapu 0  
**Status:** Protokół roboczy; kwestie wymagające potwierdzenia z promotorem oznaczono jako **DO USTALENIA**.

## 1. Cel

Zgromadzenie i uporządkowanie bocznych nagrań biegu na bieżni, aby obliczać dwuwymiarowy kąt zgięcia kolana i porównywać wyniki MediaPipe z ręczną anotacją tych samych klatek.

## 2. Warunki badania

- Rejestracja biegu na bieżni z perspektywy możliwie prostopadłej do płaszczyzny ruchu (widok boczny).
- Prędkości docelowe: **7,5 km/h** i **8,5 km/h**; dla każdej osoby zakładane są dwa nagrania, po jednym dla każdej prędkości.
- Kamera: iPhone 15; rzeczywistą rozdzielczość, FPS, czas trwania i ewentualny zmienny odstęp czasu między klatkami (VFR) należy odczytać z plików, nie zakładać stałego FPS.
- Analizowana jest **jedna kończyna — bliższa kamerze**. Anatomiczną stronę `LEFT` albo `RIGHT` ustala się ręcznie osobno dla każdego filmu.
- Kamera powinna pozostawać nieruchoma; w kadrze powinny być widoczne biodro, kolano i kostka nogi analizowanej w całym możliwym zakresie ruchu.
- Nie porównujemy bezpośrednio kątów 2D z pełnymi kątami anatomicznymi 3D.

## 3. Identyfikacja i metadane

Każdemu uczestnikowi przypisujemy anonimowy identyfikator (`P01`, `P02`, ...), a każdemu nagraniu jednoznaczny identyfikator (`P01_7p5`, `P01_8p5`). Oryginalnej nazwy pliku nie zmieniamy w archiwum; powiązanie zapisujemy w tabeli metadanych poza publicznym repozytorium.

Minimalne kolumny `recordings.csv`:

```csv
recording_id,participant_id,original_filename,speed_kmh,near_leg,recording_date,resolution,fps_nominal,duration_s,analysis_start_s,analysis_end_s,notes
P01_7p5,P01,DO_UZUPELNIENIA,7.5,LEFT,DO_UZUPELNIENIA,DO_UZUPELNIENIA,DO_UZUPELNIENIA,DO_UZUPELNIENIA,DO_UZUPELNIENIA,DO_UZUPELNIENIA,
```

**Uwaga:** `LEFT` jest wyłącznie przykładem formatu, a nie ustaloną stroną dla P01. Nie należy przypisywać uczestników, stron ani prędkości do plików `IMG_*.MOV` na podstawie numerów nazw.

## 4. Przygotowanie odcinków analizy

1. Zweryfikuj dla każdego filmu osobę, prędkość i nogę bliższą kamerze.
2. Sprawdź, czy nagranie obejmuje ustabilizowany bieg (bez wejścia na bieżnię, zmiany prędkości i zatrzymania).
3. Wybierz i zapisz początek oraz koniec analizowanego fragmentu (`analysis_start_s`, `analysis_end_s`).
4. Zapisz czas każdej analizowanej klatki na podstawie rzeczywistych znaczników czasu filmu, jeśli są dostępne.
5. Zachowaj informację o odrzuconych fragmentach i przyczynach odrzucenia.

## 5. Zasady jakości

Oceniamy: boczne ustawienie kamery, widoczność biodra/kolana/kostki, ostrość, oświetlenie, zasłonięcia, ruch kamery oraz stabilność tempa. Nagranie może być oznaczone jako `ACCEPTED`, `ACCEPTED_WITH_LIMITATIONS` albo `REJECTED` wraz z krótkim uzasadnieniem. Decyzja o przydatności do analizy nie może wynikać wyłącznie z pojedynczej klatki.

## 6. Organizacja plików i bezpieczeństwo

Zalecana struktura:

```text
mediapipe-running-gait-analysis/
  README.md
  docs/
    recording_protocol.md
    annotation_protocol.md
  src/
  tests/

private_gait_data/                 # poza repozytorium Git
  originals/                       # oryginalne pliki MOV — tylko do odczytu
  metadata/                        # przyporządkowanie uczestników i nagrań
  annotations/                     # ręczne punkty i oceny jakości
  processed/                       # wyniki analizy i wizualizacje
  backups/                         # opcjonalna kopia lokalna; druga kopia na osobnym nośniku
```

- Nie dodawaj do Git filmów uczestników, danych identyfikujących ani wyników umożliwiających identyfikację.
- Zabezpiecz oryginały przed nadpisaniem; analizuj kopie robocze.
- **Kopia zapasowa: PENDING.** Wykonaj niezależną kopię oryginałów (np. na innym dysku lub w kontrolowanym magazynie chmurowym) i sprawdź możliwość odczytu plików.
- Ustal zasady zgody uczestników, dostępu do danych, przechowywania i wykorzystania w pracy zgodnie z wymaganiami uczelni i promotora.
- Zapisuj wersję kodu i konfigurację analizy, aby wyniki można było odtworzyć.

## 7. Jednostka analizy i interpretacja

Podstawową jednostką organizacyjną jest nagranie przypisane do uczestnika i prędkości. Klatki z jednego nagrania są zależne czasowo i **nie stanowią niezależnych uczestników**. Wyniki przedstawiaj co najmniej na poziomie nagrania, a następnie uczestnika/prędkości.

## 8. DO USTALENIA

- [ ] Liczba uczestników i kompletność par nagrań 7,5/8,5 km/h.
- [ ] Przypisanie `IMG_*.MOV` do uczestników, prędkości i stron `LEFT`/`RIGHT`.
- [ ] Zgody, zasady przechowywania i ewentualnej publikacji materiałów.
- [ ] Długość i kryteria wyboru stabilnego odcinka każdego nagrania.
- [ ] Potwierdzenie z promotorem, że kąt zgięcia 2D jest główną zmienną wynikową.
- [ ] Wykonanie i weryfikacja kopii zapasowej.

## 9. Warunek zakończenia etapu 0

Istnieje bezpieczna kopia oryginałów, kompletna tabela nagrań i zapisane zasady doboru odcinków, kończyny i definicji kąta. Nie rozpoczynamy masowego przetwarzania z niezweryfikowanymi metadanymi.
