# Protokół ręcznej anotacji punktów i kąta kolana

**Projekt:** MediaPipe — analiza biegu z widoku bocznego  
**Wersja:** 1.0 — ustalenia etapu 0  
**Status:** Protokół roboczy do przetestowania na pilotażu.

## 1. Cel

Uzyskanie referencyjnych współrzędnych 2D biodra, kolana i kostki **nogi bliższej kamerze**, na dokładnie tych samych klatkach, które przetwarza MediaPipe. Na ich podstawie wyznaczamy kąt zgięcia kolana i porównujemy go z wynikiem automatycznym.

**Ograniczenie:** anotacja z pojedynczej kamery jest subiektywnym oszacowaniem położenia punktów w obrazie, a nie pomiarem laboratoryjnym 3D ani bezbłędną prawdą odniesienia.

## 2. Konwencja kąta

Punkty w układzie współrzędnych obrazu: `H = (hip_x, hip_y)`, `K = (knee_x, knee_y)`, `A = (ankle_x, ankle_y)`.

Wektory wychodzące z kolana: `u = H - K`, `v = A - K`.

Kąt wewnętrzny:

```text
theta_deg = degrees(acos(clip(dot(u, v) / (norm(u) * norm(v)), -1, 1)))
```

**Główna zmienna wynikowa:**

```text
knee_flexion_deg = 180.0 - theta_deg
```

- Przy wyproście: `knee_flexion_deg` około 0°.
- Większe zgięcie: większa wartość kąta.
- Zapisuj oba pola: `knee_internal_angle_deg` i `knee_flexion_deg`.
- Jeśli dwa punkty pokrywają się albo długość któregoś wektora jest zerowa lub bliska zeru, kąt oznacz jako brak danych, nie 0°.
- Wszystkie wartości dotyczą **projekcji 2D** i zależą od ustawienia kamery.

## 3. Wybór nogi

- Dla każdego nagrania ręcznie ustal `near_leg = LEFT` lub `RIGHT` według anatomicznej strony uczestnika.
- W obrębie jednego nagrania **nie zmieniaj** analizowanej strony w celu uzupełnienia brakujących punktów.
- Używaj punktów MediaPipe o tej samej stronie: `LEFT_HIP`, `LEFT_KNEE`, `LEFT_ANKLE` lub odpowiedników `RIGHT_*`.
- Nie utożsamiaj lewej strony obrazu z lewą nogą uczestnika.

## 4. Wybór 30 klatek na nagranie

1. Wybierz wcześniej zdefiniowany odcinek stabilnego biegu.
   Dla próby głównej ustaw 20 s przed ekstrakcją (np. 5–25 s). Zakres jest
   półotwarty `[początek, koniec)` i zapisany jako `analysis_start_s` oraz
   `analysis_end_s`. Numery klatek i czas pozostają względem pełnego filmu.
   W nowych ekstrakcjach czas pochodzi z PTS klatek (pierwsza klatka filmu
   ma czas względny 0), także przy VFR; nie wyznaczamy go z samego FPS.
2. Zastosuj **30 z góry określonych znaczników czasu** równomiernie rozłożonych w tym odcinku (np. środki 30 równych przedziałów).
3. Dla każdego znacznika wybierz najbliższą rzeczywistą klatkę i zapisz `frame_index` oraz `timestamp_s`; pilnuj, by nie wybierać dwa razy tej samej klatki.
4. Nie wybieraj klatek na podstawie tego, czy MediaPipe poprawnie wykrył punkty — to mogłoby sztucznie poprawić ocenę algorytmu.
5. Jeśli wylosowana/wyznaczona klatka jest trudna, **zachowaj ją w zestawie** i oznacz odpowiednią flagą. Nie zastępuj jej łatwiejszą klatką bez udokumentowanej reguły.
6. Po pilotażu sprawdź, czy próba obejmuje różne fazy cyklu biegu. Ewentualną zmianę metody próbkowania wprowadź przed anotacją całego zbioru i udokumentuj.

**Implementacja:** znaczniki są środkami równych przedziałów między wskazanym
początkiem i końcem. Najbliższa klatka musi należeć do tego zakresu; przy
remisie wybierana jest wcześniejsza. Jeżeli kilka znaczników wskazuje tę samą
klatkę, zachowujemy ją raz z najbliższym znacznikiem (wcześniejszym przy remisie),
bez dobierania zamienników. Aplikacja raportuje mniejszą liczbę unikalnych
klatek i zapisuje zakres, znaczniki oraz różnice czasów w CSV wyboru.
Klatki bez detekcji MediaPipe lub bez kąta nie są odrzucane.

**30 klatek na nagranie to przyjęta liczba startowa, a nie gwarancja reprezentatywności statystycznej.**

## 5. Definicje punktów

- **HIP:** przybliżone położenie środka stawu biodrowego w projekcji bocznej, konsekwentnie względem sylwetki i okolicy miednicy.
- **KNEE:** przybliżony środek stawu kolanowego w projekcji bocznej, nie krawędź spodni ani najbardziej wysunięty punkt skóry.
- **ANKLE:** przybliżony środek stawu skokowego w projekcji bocznej, nie czubek buta ani pięta.

Anotator powinien przed główną serią przejrzeć przykładowe klatki i ustalić spójny sposób wskazywania tych punktów. Jeśli staw jest zasłonięty, nie należy przypisywać fikcyjnej dokładności współrzędnym.

## 6. Flagi jakości i niepewności

Każdemu punktowi przypisz jedną z wartości:

| Flaga | Znaczenie | Dalsze użycie |
|---|---|---|
| `CLEAR` | Punkt dobrze widoczny | Analiza podstawowa |
| `UNCERTAIN` | Punkt częściowo zasłonięty / trudny do jednoznacznego oszacowania | Zachowaj współrzędne, analizuj osobno lub w analizie wrażliwości |
| `UNMARKABLE` | Punkt niewidoczny lub nie da się go wiarygodnie wskazać | Współrzędne puste, brak kąta referencyjnego |

Klatka otrzymuje `frame_quality = CLEAR`, gdy wszystkie trzy punkty mają `CLEAR`; `UNCERTAIN`, jeśli co najmniej jeden ma `UNCERTAIN` i żaden nie jest `UNMARKABLE`; `UNMARKABLE`, jeśli co najmniej jeden punkt jest `UNMARKABLE`.

**Zasada raportowania:** nie usuwaj po cichu trudnych klatek. Podaj liczbę i odsetek klatek z każdą flagą. Główne MAE/RMSE oblicz dla kompletnych klatek `CLEAR`; wyniki z uwzględnieniem `UNCERTAIN` przedstaw osobno jako analizę wrażliwości. Klatki `UNMARKABLE` raportuj jako brak pomiaru referencyjnego.

## 7. Format danych

Proponowane kolumny `annotations.csv` (przechowywany poza publicznym repozytorium):

```csv
recording_id,participant_id,near_leg,frame_index,timestamp_s,hip_x,hip_y,hip_quality,knee_x,knee_y,knee_quality,ankle_x,ankle_y,ankle_quality,frame_quality,knee_internal_angle_deg,knee_flexion_deg,annotator_id,annotation_session,notes
```

- Współrzędne podawaj w pikselach **obrazu analitycznego** użytego przez MediaPipe
  (to samo kadrowanie i skalowanie), nie okna narzędzia anotacyjnego. Nowe CSV
  zapisują `analysis_id`, `source_sha256`, `crop_mode` i `resize_height`;
  aplikacja przenosi je do anotacji i sprawdza zgodność filmu przed wyświetleniem obrazu.
- Ustal i zapisz, czy obraz był obracany lub odbijany lustrzanie. Anotacja i MediaPipe muszą korzystać z tej samej orientacji.
- Zachowaj dokładne `frame_index` i `timestamp_s` oraz identyfikator nagrania.
- Braki zapisuj jako puste wartości / `NaN`, nie jako `0`.

## 8. Procedura anotacji

1. Otwórz dokładnie wskazaną klatkę z oryginalnego filmu.
2. Sprawdź przypisaną nogę i orientację obrazu.
3. Zaznacz biodro, kolano i kostkę tej samej nogi.
4. Nadaj flagę jakości każdemu punktowi.
5. Jeśli punkt jest `UNMARKABLE`, pozostaw współrzędne puste i opisz przyczynę.
6. Oblicz oba kąty automatycznie z anotowanych współrzędnych (nie wpisuj ich ręcznie).
7. Zapisz rekord i ewentualny komentarz.
8. Po zakończeniu przejrzyj wizualizację punktów i wykres kąta, aby znaleźć błędy anotacji.

## 9. Kontrola jakości i porównanie z MediaPipe

- Pilotaż: oznacz 30 klatek jednego nagrania i sprawdź spójność oraz czas pracy.
- Zalecane: powtórz anotację części klatek po przerwie lub z udziałem drugiego anotatora, aby oszacować powtarzalność referencji.
- Każde powtórzenie zapisuj w nowej `annotation_session`; nie zastępuj pierwszego
  pomiaru. W aplikacji użyj „Nowa sesja pomiarowa”. W nowej sesji nie podglądaj
  wcześniejszych punktów ani notatek. Do podstawowej walidacji wybierz jedną
  sesję referencyjną i jednego anotatora; pozostałe zachowaj do oceny powtarzalności.
- Porównuj **identyczne klatki, tę samą nogę i tę samą konwencję kąta**.
- Dla par z dostępnymi wartościami licz błąd `MediaPipe - reference`, MAE, RMSE i bias; podawaj liczbę porównanych i wykluczonych klatek.
- Rozdziel wyniki według nagrania i prędkości; pamiętaj, że kolejne klatki jednego uczestnika nie są niezależnymi obserwacjami osobowymi.
- Nie stosuj wygładzania tylko do jednej serii bez opisania konsekwencji; zachowaj wyniki surowe.

## 10. DO USTALENIA / zatwierdzenia

- [ ] Potwierdzenie przez promotora definicji kąta zgięcia 2D jako głównej miary.
- [ ] Wybór narzędzia do ręcznej anotacji i sposobu eksportu współrzędnych.
- [ ] Sposób kontroli zgodności anotacji (powtórne oznaczenie / drugi anotator).
- [ ] Czy 30 klatek na film zapewnia wystarczającą reprezentację faz biegu — ocena po pilotażu.
- [ ] Dokładna reguła doboru odcinków analizy i synchronizacji klatek przy zmiennym FPS.

## 11. Warunek zakończenia etapu 0

Protokół jest zapisany i przetestowany na krótkim pilotażu, a decyzje o kącie, stronie, liczbie klatek i flagach jakości są jednoznaczne. Pełna anotacja rozpoczyna się dopiero po zatwierdzeniu procedury.
