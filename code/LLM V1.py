"""
Masseninferenz der Frage-Antwort-Paare mit Gemma 4 E4B über vLLM (Versuch 1).

Analyse von Earnings Calls: Hybrides Modell aus Machine Learning und Large Language Models zur Verringerung von Distraction Bias
Bachelorarbeit an der Universität St.Gallen (HSG) / Institut für Wirtschaftsinformatik (IWI-HSG)

Autor: Ruben Nicolas Cardell
Kontakt: ruben.cardell@student.unisg.ch
Referent: Prof. Dr. Ivo Blohm
Betreuer: Tobias Kotzian

---

Ablauf:
1. INPUT_CSV einlesen; benötigte Spalten: episode_id, sequence, qa_pair
2. Je Zeile SYSTEM_PROMPT + qa_pair im Batch an Gemma 4 E4B übergeben
3. Die Roh-Ausgaben des Modells als einzelne CSV in OUTPUT_DIR schreiben

Das Skript läuft auf einer gemieteten GPU-Instanz (Nvidia L4). Das Parsen der
JSON-Ausgaben erfolgt anschliessend im Notebook "POSTprocessing.ipynb".
"""

import os
import csv
import time
import traceback

import pandas as pd

from vllm import LLM, SamplingParams

# ========================= KONFIGURATION =========================

INPUT_CSV = "preproc_data_v3.csv"       # Ausgabe des Preprocessing: eine Zeile je Frage-Antwort-Paar
OUTPUT_DIR = "outputs"                  # innerhalb des Lightning-Projektordners, wird automatisch synchronisiert

# Maximale Kontextlänge des Modells. Wird im Preprocessing-Notebook aus der
# längsten Eingabe (inkl. System-Prompt) bestimmt und hier mit Reserve gesetzt.
MAX_MODEL_LEN = 18500

MODEL_NAME = "google/gemma-4-E4B-it"

MAX_OUTPUT_TOKENS = 200                 # Reserve über den beobachteten ~100 Tokens des JSON-Objekts
TEMPERATURE = 0.0                       # deterministisch, zur Reproduzierbarkeit der Ergebnisse
GPU_MEMORY_UTILIZATION = 0.90

# Kleine Zahl (z.B. 1000) für einen kurzen Testlauf zur Zeitschätzung,
# None für den vollständigen Datensatz.
LIMIT = None

# System-Prompt (Appendix B): Aufgabe, Definition der neun Sentiment-Dimensionen,
# direkte Kursprognose und JSON-Ausgabeformat.
SYSTEM_PROMPT = """You are a helpful assistant. Your job is to analyze the sentiment, tone, and emotion of an earnings call question-answer pair. Measure each dimension on a scale of 0 to 1 in increments of 0.1.
- Relevant
0 = The answer does not address the question — the speaker deflects, changes subject, or responds with generic/unrelated information.
1 = The answer directly addresses the specific question asked, staying on-topic and matching its scope.

- Clear
0 = The answer is vague, ambiguous, or contradictory — a listener could not confidently restate the speaker's main point.
1 = The answer is direct and unambiguous — a listener could confidently restate the speaker's main point in one sentence.

- Optimistic
0 = The speaker frames future outcomes negatively or neutrally — emphasizing headwinds, declines, challenges, or "wait and see" language.
1 = The speaker frames future outcomes positively — emphasizing growth, improvement, opportunity, or favorable expectations.

- Specific
0 = The answer relies on generic, qualitative language with no concrete figures, dates, or technical detail (e.g., "strong performance," "on track," "good progress").
1 = The answer includes at least one concrete number, percentage, date, metric, or technical/operational detail supporting the claim.

- Cautious
0 = The speaker takes a bold or committed stance — no acknowledgment of risk, downside scenarios, or uncertainty; commitments are unqualified.
1 = The speaker explicitly acknowledges risk, downside scenarios, or uncertainty, or hedges a commitment/guidance (e.g., "depending on," "if conditions permit," "we can't guarantee").

- Assertive
0 = The speaker uses uncertain or hedged language about what will happen (e.g., "might," "could," "we hope," "possibly").
1 = The speaker uses definitive, confident language about what will happen (e.g., "will," "is," "we are confident that").

- Valence
0 = The content is negative — words signaling dissatisfaction, disappointment, frustration, or distress.
1 = The content is positive — words signaling satisfaction, happiness, pride, or contentment.

- Arousal
0 = The tone is calm and low-energy — measured, flat, unhurried delivery, regardless of whether the content is positive or negative.
1 = The tone is high-energy and intense — excited, urgent, or agitated delivery, regardless of whether the content is positive or negative.

- Dominance
0 = The speaker frames the company as controlled by external forces — outcomes depend on the market, regulators, customers, or circumstances beyond their control.
1 = The speaker frames the company as in control — actively driving outcomes through their own decisions and actions.

Additionally, predict the performance of the companies stock price over the following 90 days. Give an estimate in percent like this: 0.05 (= 5% up); -0.05 (= 5% down)

Output the values in json format like this. Do not output any explanation or other content, just the json as text.
{"relevant": value, "clear": value, "optimistic": value, "specific": value, "cautious": value, "assertive": value, "valence": value, "arousal": value, "dominance": value, "price": value}
"""

# ================================================================


def safe_truncate(text, tokenizer, max_model_len, max_output_tokens, buffer_tokens=300):
    """Kürzt übermässig lange Eingaben, damit eine einzelne Zeile das Kontextfenster
    nicht sprengt und den gesamten Lauf abbricht.

    Rückgabe: (Text, ursprüngliche Token-Anzahl, ob gekürzt wurde), damit die
    Kürzung protokolliert und nicht stillschweigend hingenommen wird.
    """
    budget = max(64, max_model_len - max_output_tokens - buffer_tokens)
    ids = tokenizer.encode(text)
    if len(ids) <= budget:
        return text, len(ids), False
    return tokenizer.decode(ids[:budget]), len(ids), True


def log_truncation(episode_id, sequence, original_token_count):
    """Hängt gekürzte Zeilen an ein Protokoll an, damit sie später gefunden und
    (z.B. mit höherem MAX_MODEL_LEN) erneut verarbeitet werden können."""
    log_path = os.path.join(OUTPUT_DIR, "truncated_rows.csv")
    write_header = not os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["episode_id", "sequence", "original_token_count"])
        writer.writerow([episode_id, sequence, original_token_count])


def main():
    """Führt den gesamten Inferenzlauf aus: Daten laden, Modell starten,
    Prompts aufbauen, im Batch generieren und Ergebnisse schreiben."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Daten laden -------------------------------------------------------
    print(f"Loading {INPUT_CSV} ...")
    df = pd.read_csv(INPUT_CSV, dtype={"episode_id": str, "sequence": str})
    if LIMIT:
        df = df.head(LIMIT)
        print(f"LIMIT is set: only processing the first {LIMIT} rows (pilot run).")

    remaining = len(df)
    print(f"{remaining} rows to process.")
    if remaining == 0:
        print("Nothing to do.")
        return

    # --- Modell laden ------------------------------------------------------
    print("Loading vLLM + Gemma 4 E4B (this can take a minute or two)...")
    llm_kwargs = dict(
        model=MODEL_NAME,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        trust_remote_code=True,
    )
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_OUTPUT_TOKENS)

    # --- Prompts aufbauen --------------------------------------------------
    # Je Zeile ein Chat aus System-Prompt (Instruktionen) und User-Prompt (Text).
    conversations = []
    n_truncated = 0
    for _, row in df.iterrows():
        text = row.get("qa_pair", "")
        if pd.isna(text):
            text = ""
        text = str(text)
        if text.strip() == "":
            conversations.append(None)  # Markierung: kein Modellaufruf für diese Zeile
            continue
        text, orig_len, was_truncated = safe_truncate(text, tokenizer, MAX_MODEL_LEN, MAX_OUTPUT_TOKENS)
        if was_truncated:
            n_truncated += 1
            log_truncation(row["episode_id"], row["sequence"], orig_len)
        conversations.append([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ])
    if n_truncated:
        print(f"[warn] {n_truncated} row(s) exceeded MAX_MODEL_LEN and were truncated "
              f"-> see {OUTPUT_DIR}/truncated_rows.csv")

    # Leere Zeilen vom Modellaufruf ausnehmen und separat als Fehler markieren
    real_indices = [i for i, c in enumerate(conversations) if c is not None]
    real_conversations = [conversations[i] for i in real_indices]

    # --- Inferenz ----------------------------------------------------------
    t0 = time.time()
    if real_conversations:
        outputs = llm.chat(real_conversations, sampling_params=sampling_params, use_tqdm=True, chat_template_kwargs={"enable_thinking": False})
    else:
        outputs = []
    elapsed = time.time() - t0

    # Ausgaben wieder an die ursprünglichen Zeilenpositionen einsetzen
    texts_out = ["ERROR: empty qa_pair"] * len(conversations)
    for idx, out in zip(real_indices, outputs):
        try:
            texts_out[idx] = out.outputs[0].text
        except Exception:
            texts_out[idx] = "ERROR: no output generated"

    rate = len(real_conversations) / elapsed if elapsed > 0 else 0
    print(f"\nProcessed {len(df)} rows in {elapsed/60:.1f} min ({rate:.2f} rows/sec).\n")

    # --- Ergebnisse schreiben ----------------------------------------------
    # Nur Schlüssel und Roh-Ausgabe; das Parsen erfolgt im Postprocessing.
    result_df = pd.DataFrame({
        "episode_id": df["episode_id"],
        "sequence": df["sequence"],
        "output": texts_out,
    })

    output_path = os.path.join(OUTPUT_DIR, "final_output.csv")
    try:
        result_df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL, encoding="utf-8")
    except Exception as e:
        # Ausweichpfad, falls das Schreiben der Hauptdatei fehlschlägt
        fallback_path = os.path.join(OUTPUT_DIR, "final_output_fallback.csv")
        print(f"[error] failed writing {output_path}: {e}\nTrying fallback {fallback_path}")
        result_df.to_csv(fallback_path, index=False, quoting=csv.QUOTE_ALL, encoding="utf-8")
    print(f"Done. Final CSV -> {output_path} ({len(result_df)} rows)")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Vollständigen Traceback ausgeben, damit ein Abbruch auf der gemieteten
        # GPU-Instanz im Log nachvollziehbar bleibt.
        print("\n[FATAL] Unhandled error, dumping traceback below.")
        traceback.print_exc()
        raise
