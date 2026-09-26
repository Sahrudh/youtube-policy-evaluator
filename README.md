# youtube-policy-evaluator
Async LLM moderation &amp; adversarial evaluation pipeline using Gemini and Pydantic structured outputs.

# YouTube Policy Violation Classifier & Adversarial Evaluator

An automated Trust & Safety moderation and evaluation pipeline built on **Gemini 3.5 Flash**. This project classifies user comments and transcripts against YouTube Community Guidelines (**Harassment**, **Spam**, **Hate Speech**) while minimizing false positives on aggressive gaming slang and harsh creator feedback.

---

## Key Engineering Highlights

- **Enforced JSON Schemas:** Leverages Pydantic validation to guarantee deterministic JSON output, eliminating parse errors in production ingestion pipelines.
- **Latent Reasoning (Chain-of-Thought):** Prompts intermediate policy checks prior to final category assignment, preventing misclassifications on obfuscated slang.
- **Adversarial Robustness:** Tested against adversarial leetspeak, phonetic masking, spaced URLs, and doxxing patterns.
- **Controlled High-Throughput Ingestion:** Implements `asyncio` with explicit semaphore concurrency limits (`Semaphore(3)`) and exponential backoff to handle API rate quotas reliably.
- **Evaluation-Driven Architecture:** Uses Scikit-learn to benchmark Model Precision, Recall, Macro F1, and cross-class confusion matrix drift.

---

## Policy Taxonomy & Ground Truth Rules

| Policy Class | Description | Edge Case Nuance (Hard Negatives) |
| :--- | :--- | :--- |
| **`NONE`** | Compliant content, hyperbole, benign profanity. | Video critique or idioms are protected critique, not violations. |
| **`HARASSMENT`** | Targeted degradation, doxxing, threats, self-harm. | Sarcastic or disguised encouragement of harm is classified as harassment. |
| **`SPAM`** | Deceptive promos, bot schemes, handle scraping. | Obfuscated links and spaced contact handles are flagged. |
| **`HATE_SPEECH`** | Attacks targeting protected collective characteristics. | Dehumanizing metaphors applied to groups trigger removal. |

---
## System Architecture

```text
User Comment / Transcript
          │
          ▼
┌────────────────────────────────────────────────────────┐
│  Async Ingestion & Semaphore Concurrency Controller    │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│  Gemini 3.5 Flash (Structured Outputs Engine)          │
│  - Step 1: Execute Latent Chain-of-Thought Check       │
│  - Step 2: Extract Flagged Tokens                      │
│  - Step 3: Emit Strict Pydantic JSON Payload           │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│  Evaluation & Metrics Engine                           │
│  - Automated Error Logging & Retries                   │
│  - Precision / Recall / F1-Score Breakdown             │
│  - Confusion Matrix Error-Drift Analysis               │
└────────────────────────────────────────────────────────┘
```
## Benchmark Results

Evaluated against an adversarial benchmark ($N=30$) containing obfuscated spam, masked profanity, and gaming slang:

```text
================================================================================
CLASSIFICATION REPORT
================================================================================
              precision    recall  f1-score   support

        NONE       1.00      1.00      1.00         8
  HARASSMENT       1.00      1.00      1.00         7
        SPAM       1.00      1.00      1.00         7
 HATE_SPEECH       1.00      1.00      1.00         8

    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30

CONFUSION MATRIX (Rows: Actual, Columns: Predicted)
             NONE  HARASSMENT  SPAM  HATE_SPEECH
NONE            8           0     0            0
HARASSMENT      0           7     0            0
SPAM            0           0     7            0
HATE_SPEECH     0           0     0            8
```
## Output Schema Example

The model is constrained via Pydantic to enforce the following deterministic JSON response structure:

```json
{
  "chain_of_thought": "(1) Targets an individual creator maliciously. (2) Uses masked acronym for self-harm. (3) No protected group or commercial solicitation involved.",
  "category": "HARASSMENT",
  "confidence_score": 0.98,
  "action": "REMOVE",
  "flagged_tokens": [
    "subhuman garbage",
    "kys"
  ]
}
```
## Local Setup & Execution

### 1. Clone the Repository
```bash
git clone [https://github.com/](https://github.com/)<YOUR-USERNAME>/youtube-policy-evaluator.git
cd youtube-policy-evaluator
```
### 2. Set Up a Virtual Environment

#### macOS/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```
#### Windows (Command Prompt)
```bash
python -m venv venv
venv\Scripts\activate.bat
```
#### Windows (PowerShell)
```bash
python -m venv venv
venv\Scripts\Activate.ps1
```
### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
### 4. Configure Your API Key

Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey?utm_source=gemini).

#### macOS/Linux
```bash
export GEMINI_API_KEY="your-gemini-api-key-here"
```
#### Windows (Command Prompt)
```bash
set GEMINI_API_KEY=your-gemini-api-key-here
```
#### Windows (PowerShell)
```bash
$env:GEMINI_API_KEY="your-gemini-api-key-here"
```

Note for Local Execution: Update the script initialization from userdata.get('GEMINI_API_KEY') (Colab-specific) to read from system environment variables:
```python
import os
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)
```

### 5. Run the Evaluator
```bash
python main.py
```
