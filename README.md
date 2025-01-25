# Wyzwanie Segmentacji Sygnału ECG

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Projekt ten został stworzony na potrzeby wyzwania segmentacji sygnału ECG i wykrywania arytmii. Naszym celem było opracowanie narzędzia do analizy sygnałów ECG, które będzie w stanie wykrywać migotanie przedsionków z jak najwyższą dokładnością. Zbudowaliśmy model, który efektywnie klasyfikuje dane ECG, identyfikując nieregularne rytmy serca związane z migotaniem przedsionków. Rozwiązanie wykorzystuje zaawansowane algorytmy, techniki przetwarzania sygnałów oraz metody głębokiego uczenia, aby poprawić precyzję i zmniejszyć liczbę fałszywych alarmów.

Wykorzystaliśmy sieć neuronową z wieloskalowymi blokami konwolucyjnymi i LSTM oraz połączeniami rezydualnymi do lepszej propagacji gradientu. Sieć przyjmowała interwały RR wyliczane na podstawie analizy QRS.

## Organizacja Projektu

```
├── LICENSE            <- Licencja open-source, jeśli wybrana
├── Makefile           <- Makefile z wygodnymi komendami jak `make data` lub `make train`
├── README.md          <- Główny plik README dla deweloperów używających tego projektu.
├── data
│   ├── external       <- Dane z zewnętrznych źródeł.
│   ├── interim        <- Przetworzone dane pośrednie.
│   ├── processed      <- Ostateczne, kanoniczne zestawy danych do modelowania.
│   └── raw            <- Oryginalne, niezmienione dane.
│
├── docs               <- Domyślny projekt mkdocs; zobacz www.mkdocs.org dla szczegółów
│
├── models             <- Wytrenowane i zserializowane modele, predykcje modeli lub podsumowania modeli
│
├── notebooks          <- Notatniki Jupyter. Konwencja nazewnictwa to numer (dla porządku),
│                         inicjały twórcy i krótki opis, np. `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Plik konfiguracyjny projektu z metadanymi pakietu dla src i konfiguracją narzędzi jak black
│
├── references         <- Słowniki danych, podręczniki i inne materiały wyjaśniające.
│
├── reports            <- Wygenerowane analizy w formacie HTML, PDF, LaTeX itp.
│   └── figures        <- Wygenerowane grafiki i wykresy do raportowania
│
├── environment.yml    <- Plik środowiska do odtworzenia środowiska analizy, np. wygenerowany za pomocą `conda env export --no-builds > environment.yml`
│
├── setup.cfg          <- Plik konfiguracyjny dla flake8
│
└── src   <- Kod źródłowy używany w tym projekcie.
    │
    ├── __init__.py             <- Czyni src modułem Pythona
    │
    ├── config.py               <- Przechowuje przydatne zmienne i konfiguracje
    │
    ├── dataset.py              <- Skrypty do pobierania lub generowania danych
    │
    ├── features.py             <- Kod do tworzenia cech do modelowania
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Kod do uruchamiania inferencji modelu z wytrenowanymi modelami          
    │   └── train.py            <- Kod do trenowania modeli
    │
    └── plots.py                <- Kod do tworzenia wizualizacji
```

## Aplikacja CLI

Projekt wykorzystuje bibliotekę Typer do stworzenia aplikacji CLI, która jest zawarta w pliku `app.py`. Poniżej znajdują się przykłady użycia tej aplikacji.

### Przykłady użycia

1. **Pobranie zestawu danych:**
    ```bash
    python app.py download-dataset --dataset-name mitdb
    ```

2. **Przetworzenie zestawu danych:**
    ```bash
    python app.py preprocess-dataset --dataset-name mitdb --target-fs 250 --chunk-size 1000 --step 5 --verbosity INFO --version 2
    ```

3. **Podział danych na zestawy treningowe, walidacyjne i testowe:**
    ```bash
    python app.py process-dataset --test-size 0.2 --val-size 0.1 --verbosity INFO --version 2
    ```

4. **Trenowanie modelu:**
    ```bash
    python app.py train-model --epochs 10 --batch-size 16 --learning-rate 0.001 --model-type LSTMModel --verbosity INFO
    ```

5. **Ocena modelu:**
    ```bash
    python app.py evaluate-model --model-type LSTMModel --state-dict-name model.pth --num-samples 5
    ```

6. **Podsumowanie modelu:**
    ```bash
    python app.py model-summary --model-type LSTMModel
    ```

## Autorzy

Projekt został stworzony przez studentów w ramach wyzwania akademickiego.
