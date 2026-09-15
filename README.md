# project-NLP

# E-commerce Customer Support Chatbot

A RAG-based customer support chatbot for e-commerce, combining classical NLP,
fine-tuned/pretrained transformers, and retrieval-augmented generation to understand,
route, and answer customer messages.

**Live demo:**[ _add your Streamlit Cloud URL here_](https://project-nlp-u5z2macad4fecaoavkogky.streamlit.app/)

---

## Overview

Every incoming customer message goes through four stages before a reply is generated:

```
Customer message
      │
      ▼
┌─────────────────────┐
│ 1. Language          │  TF-IDF (char n-grams) + classical ML
│    Detection          │  → detects the message language
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│ 2. Sentiment          │  Pretrained multilingual model (XLM-RoBERTa)
│    Classifier         │  → negative / neutral / positive
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│ 3. Intent             │  TF-IDF (word n-grams) + LinearSVC
│    Classifier         │  → routes to 1 of 7 buckets
└─────────────────────┘
      │
      ▼
┌─────────────────────┐        human_handoff bucket
│  Router               │ ─────────────────────────────► Escalate to a human agent
└─────────────────────┘
      │ everything else
      ▼
┌─────────────────────┐
│ 4. Q&A RAG Pipeline    │  FAISS + sentence-transformers + Groq LLM
│                        │  → grounded, context-aware answer
└─────────────────────┘
      │
      ▼
   Reply to customer
   (+ apologetic framing if sentiment is negative)
```

---

## Modules

### 1. Language Detection
- **Approach:** TF-IDF with character n-grams (`analyzer='char_wb'`, 1–3 grams) +
  best-of Multinomial Naive Bayes / LinearSVC / Logistic Regression.
- **Dataset:** [`papluca/language-identification`](https://huggingface.co/datasets/papluca/language-identification)
  (20 languages).
- **Why character n-grams:** language ID is an orthographic-pattern problem, not a
  vocabulary problem — char n-grams generalize better to short, informal text.
- Notebook: `Module1_Language_Detection.ipynb`

### 2. Sentiment / Emotion Classifier
- **Approach:** pretrained multilingual sentiment model
  ([`cardiffnlp/twitter-xlm-roberta-base-sentiment`](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment))
  — outputs negative / neutral / positive directly.
- **Note:** an earlier version fine-tuned an English-only DistilBERT on
  [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) (6 emotions
  condensed to 3 buckets). It was replaced after testing showed it had no real
  understanding of non-Latin-script text (e.g. Arabic), producing near-random
  predictions on it. The notebook documenting that original approach and rationale is
  kept for reference: `Module2_Sentiment_Emotion_Classifier.ipynb`.

### 3. Intent Classifier
- **Approach:** TF-IDF (word n-grams) + LinearSVC, trained at two granularities:
  27 fine-grained intents (analytics/logging) and 7 routing buckets (used by the
  router): `order_delivery`, `payment_invoice`, `refund`, `account`, `feedback`,
  `human_handoff`, `subscription`.
- **Dataset:** [`bitext/Bitext-customer-support-llm-chatbot-training-dataset`](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset).
- Notebook: `Module3_Intent_Classifier.ipynb`

### 4. Q&A RAG Pipeline
- **Knowledge base:** built from the same Bitext dataset — one representative
  `(question, answer)` pair per intent (27 documents).
- **Embeddings:** `all-MiniLM-L6-v2` (sentence-transformers).
- **Vector DB:** FAISS (`IndexFlatIP` over normalized vectors = cosine similarity).
- **Generation:** Groq LLM (`openai/gpt-oss-20b`), constrained to answer only from
  retrieved context, with an honest fallback ("I don't have that information") for
  out-of-scope questions, and instructed to reply in the customer's own language.
- Notebook: `Module4_QA_RAG_Pipeline.ipynb`

---

## Routing logic

| Condition | Behavior |
|---|---|
| `intent_bucket == human_handoff` | Escalate directly to a human agent — skips the LLM entirely |
| `sentiment == negative` | Reply prefixed with an apology; flagged `priority: high` |
| Everything else | Answered by the RAG pipeline |

---

## Repository structure

```
.
├── Module1_Language_Detection.ipynb
├── Module2_Sentiment_Emotion_Classifier.ipynb   # kept for reference (see note above)
├── Module3_Intent_Classifier.ipynb
├── Module4_QA_RAG_Pipeline.ipynb
├── streamlit_app.py          # Streamlit chat UI (primary deployment)
├── app.py                    # Flask REST API (alternative deployment)
├── requirements.txt
├── language_detection_vectorizer.joblib
├── language_detection_model.joblib
├── language_detection_label_encoder.joblib
├── intent_vectorizer.joblib
├── intent_bucket_model.joblib
├── intent_bucket_label_encoder.joblib
├── intent_finegrained_model.joblib
├── intent_finegrained_label_encoder.joblib
├── rag_knowledge_base.index
├── rag_knowledge_base.csv
└── README.md
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your Groq API key
Create a `.env` file in the project root (never commit this file):
```
GROQ_API_KEY=your-key-here
```

### 3. Run locally

**Streamlit (recommended):**
```bash
streamlit run streamlit_app.py
```

**Flask API:**
```bash
python app.py
# POST http://localhost:5000/chat  {"message": "..."}
```

---

## Deployment (Streamlit Community Cloud)

1. Push the repo to GitHub (`.env` excluded via `.gitignore`).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → select this
   repo → main file path: `streamlit_app.py`.
3. Under **Advanced settings → Secrets**, add:
   ```
   GROQ_API_KEY = "your-key-here"
   ```
4. Deploy.

---

## Known limitations

- **Language detection** can misclassify very short, informal messages (trained on
  relatively formal/longer text samples).
- **Sentiment classifier** is a general-purpose pretrained model, not fine-tuned on
  customer-support-specific language — occasional misclassification on neutral,
  matter-of-fact questions is expected.
- **RAG knowledge base** is built from one representative example per intent (27
  documents); a production system would use a larger, curated FAQ/policy corpus.
- **No conversation memory** — each message is currently routed and answered
  independently, without multi-turn context.

---

## Tech stack

| Component | Tool |
|---|---|
| Classical ML | scikit-learn (TF-IDF, Naive Bayes, LinearSVC, Logistic Regression) |
| Sentiment | `cardiffnlp/twitter-xlm-roberta-base-sentiment` (transformers) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector search | FAISS |
| LLM generation | Groq (`openai/gpt-oss-20b`) |
| Deployment | Streamlit / Flask |
