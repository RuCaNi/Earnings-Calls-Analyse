<div align="center">

### Bachelorarbeit – Universität St.Gallen (HSG)
# Weniger Bias, mehr Rendite? Hybrides Modell aus Large Language Models und Machine Learning zur Analyse von Earnings Calls
## Ruben Nicolas Cardell
### Prof. Dr. Ivo Blohm & Tobias Kotzian – Institut für Wirtschaftsinformatik (IWI-HSG)
#### 14.10.2026

</div>

In diesem Repository findet sich der Code zu der Bachelorarbeit von Ruben Cardell.
Die Arbeit untersucht, **inwiefern ein hybrides Modell aus LLM-basierter
Feature-Extraktion und klassischem Machine Learning den Distraction Bias
verringern und die Rendite eines Long-Short Portfolios steigern kann.**

Ein lokales Open-Weights-Modell (Gemma 4 E4B) bewertet die anonymisierten
Frage-Antwort-Teile von 32'621 Earnings Calls des S&P 500 (2005–2025) auf neun
sprachlichen Dimensionen. Diese Merkmale dienen anschliessend als Input
klassischer ML-Modelle, die die relative Kursentwicklung über die folgenden 90
Tage prognostizieren. Bewertet werden die Modelle über die Rendite eines Long-Short
Portfolios.

## Versuchsaufbau

| Versuch | Eingabe an das LLM                                                      | Merkmale je Transkript |
|---|-------------------------------------------------------------------------|---|
| **1** | Jedes Frage-Antwort-Paar einzeln (341'312 Paare)  | 27 – Minimum, Maximum und Mittelwert je Dimension |
| **2** | Gesamter Frage-Antwort-Teil in einem Aufruf                             | 9 – eine Bewertung je Dimension |
| **3** | Kein neuer LLM-Lauf, Zusammenführung der Datensätze 1 und 2  | 36 |
| **Kontrolle** | Wie Versuch 2, aber ohne Anonymisierung (Stichprobe: 1'358 Transkripte) | 9 + Direktprognose |


### Extrahierte Dimensionen

Skala jeweils 0 bis 1 in Schritten von 0.1.

- **Valence, Arousal, Dominance** – dimensionales Emotionsmodell nach Warriner et al. (2013)
- **Relevant, Clear, Optimistic, Specific, Cautious, Assertive** – kommunikative Qualität einer Antwort nach Pardawala et al. (2024)

Zusätzlich gibt das Modell im selben Aufruf eine direkte Kursprognose aus, die
als Vergleichsgrösse dient.

## Repository-Struktur

Die Dateien sind in der Reihenfolge der Pipeline aufgeführt.

| Datei                          | Zweck |
|--------------------------------|---|
| `PREprocessing.ipynb`          | Unternehmensnamen ergänzen, Frage-Antwort-Paare extrahieren, Namen per Fuzzy-Matching anonymisieren, Datensätze für Versuch 1 und 2 erzeugen, Token-Anzahl berechnen |
| `PREprocessing CONTROL.ipynb`  | Stichprobengrösse per Power-Analyse bestimmen, Datensatz des Kontrollversuchs ohne Anonymisierung erzeugen |
| `LLM V1.py`                    | Masseninferenz je Frage-Antwort-Paar (Versuch 1), Gemma 4 E4B über vLLM |
| `LLM V2.py`                    | Masseninferenz je Transkript (Versuch 2 und Kontrollversuch), identisch bis auf die fehlende Spalte `sequence` |
| `POSTprocessing.ipynb`         | JSON-Ausgaben in Spalten auftrennen, je Transkript aggregieren, Zielvariable ergänzen, Datensatz für Versuch 3 zusammenführen |
| `POSTprocessing CONTROL.ipynb` | Auswertung des Kontrollversuchs: gepaarter t-Test, Cohen's *d<sub>z</sub>*, Einzeltests je Dimension mit Benjamini-Hochberg-Korrektur |
| `ML Training V1.ipynb`         | Modelltraining und Evaluation für Versuch 1 |
| `ML Training V2.ipynb`         | Modelltraining und Evaluation für Versuch 2 |
| `ML Training V3.ipynb`         | Modelltraining und Evaluation für Versuch 3 |
| `RESULTS Processing.ipynb`     | Kennzahlen aller Modelle und Versuche in einer Ergebnistabelle |
| `GRAPH Processing.ipynb`       | Erzeugt sämtliche Abbildungen der Arbeit |

### Namenskonvention

Die Dateinamen im Code folgen der Entwicklungsreihenfolge, nicht der
Versuchsnummerierung der Arbeit:

| Code | Arbeit |
|---|---|
| `v3` | Versuch 1 – Frage-Antwort-Paare |
| `v4` | Versuch 2 – gesamte Transkripte |
| `v5` | Versuch 3 – Kombination |

## Reproduktion

Die Notebooks sind für **Google Colab** geschrieben und lesen bzw. schreiben
unter `/content/drive/MyDrive/Bachelorarbeit/`. Die beiden `LLM V*.py` laufen auf
einer gemieteten GPU-Instanz.

| Schritt | Datei                        | Eingabe | Ausgabe |
|---|------------------------------|---|---|
| 1 | `PREprocessing.ipynb`        | `raw_hf.parquet` | `prep_data_v3.csv`, `prep_data_v4.csv`, `prep_stats_v3.csv` |
| 2a | `LLM V1.py`                  | `prep_data_v3.csv` | `output_v3.csv` |
| 2b | `LLM V2.py`                  | `prep_data_v4.csv` | `output_v4.csv` |
| 3 | `POSTprocessing.ipynb`       | `output_v3.csv`, `output_v4.csv`, `features.parquet` | `ml_data_v3.csv`, `ml_data_v4.csv`, `ml_data_v5.csv` |
| 4 | `ML Training V1/V2/V3.ipynb` | `ml_data_v3/v4/v5.csv` | `pred_v3/v4/v5.csv`, `y_test.csv`, `y_actual.csv`, `models/*.joblib` |
| 5 | `RESULTS Processing.ipynb`   | `pred_v3/v4/v5.csv` | Ergebnistabelle |
| 6 | `GRAPH Processing.ipynb`     | Datensätze und gespeicherte Modelle | Abbildungen 1–8 |

Kontrollversuch, unabhängig von Schritt 4 bis 6:

| Schritt | Datei                          | Eingabe | Ausgabe |
|---|--------------------------------|---|---|
| K1 | `PREprocessing CONTROL.ipynb`  | `raw_hf.parquet` | `prep_data_control.csv` |
| K2 | `LLM V2.py`                    | `prep_data_control.csv` | `output_control.csv` |
| K3 | `POSTprocessing CONTROL.ipynb` | `output_control.csv`, `ml_data_v4.csv` | Testergebnisse zu H1 |

### Daten

Grundlage ist ein öffentlich verfügbarer Datensatz von Hugging Face:

- Nanavaty, R. (2026). *S&P 500 earnings episodes (2005–2025)* [Data set]. Hugging Face.
<https://huggingface.co/datasets/RudrakshNanavaty/earnings-call-data>

`raw_hf.parquet` enthält die Transkripte als Liste von Sprecher-Turns
(`structured_content`) je `episode_id` im Format `TICKER_JAHR_QUARTAL`.

`features.parquet` enthält die zugehörigen Kursentwicklungen (`move_1d`, `move_30d`,
`move_next_qtr`).

## Requirements

**Notebooks (Colab):**

```
pandas  numpy  pyarrow  matplotlib  seaborn
scikit-learn  xgboost  lightgbm  statsmodels  scipy  joblib
rapidfuzz  transformers  yfinance  tqdm
```

**GPU-Instanz (Lightning AI):**

```
vllm  pandas
```

## Lizenz und Nutzung
<div align="center">

© 2026 Ruben Cardell — lizenziert unter [PolyForm Noncommercial 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0)

Die Nutzung, Bearbeitung und Weitergabe ist für private Zwecke, Forschung und Lehre sowie durch gemeinnützige Organisationen gestattet, sofern Urheber- und Lizenzhinweis erhalten bleiben.
Eine kommerzielle Nutzung ist ohne vorherige schriftliche Genehmigung ausgeschlossen.\
Die Software wird ohne Gewähr bereitgestellt. Eine Haftung für Schäden ist ausgeschlossen.
</div>