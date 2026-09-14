"""
Deployment service for the e-commerce customer-support chatbot.

Wires together the 4 trained modules:
  1. Language Detection   (TF-IDF char n-grams + classical ML)
  2. Sentiment Classifier (fine-tuned DistilBERT)
  3. Intent Classifier    (TF-IDF word n-grams + LinearSVC, bucket-level)
  4. Q&A RAG Pipeline     (FAISS + sentence-transformers + Groq LLM)

Routing logic (see route_message()):
  - human_handoff intent          -> escalate directly, skip the LLM entirely
  - negative sentiment            -> escalate as priority + apologetic framing
  - everything else               -> answer via the RAG pipeline

Run with:
    export GROQ_API_KEY="your-key-here"
    python app.py

Expected artifacts in the working directory (produced by the 4 notebooks):
    language_detection_vectorizer.joblib
    language_detection_model.joblib
    language_detection_label_encoder.joblib
    sentiment_model/                        (saved HF model + tokenizer folder)
    intent_vectorizer.joblib
    intent_bucket_model.joblib
    intent_bucket_label_encoder.joblib
    rag_knowledge_base.index
    rag_knowledge_base.csv
"""

import os
import re
import logging

import joblib
import pandas as pd
import faiss
from flask import Flask, request, jsonify
from transformers import pipeline as hf_pipeline
from sentence_transformers import SentenceTransformer
from groq import Groq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chatbot")

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------
GROQ_MODEL = "openai/gpt-oss-20b"
RAG_TOP_K = 3
INTENT_BUCKETS_REQUIRING_HANDOFF = {"human_handoff"}
NEGATIVE_SENTIMENT_LABEL = "negative"

SYSTEM_PROMPT = """You are a helpful e-commerce customer support assistant.
Answer the customer's question using ONLY the information in the CONTEXT below.
If the context does not contain enough information to answer confidently, say you
don't have that information and offer to connect the customer with a human agent.
Keep answers concise, polite, and to the point. Do not invent policies, prices,
order numbers, or timelines that are not present in the context."""

APOLOGETIC_PREFIX = (
    "I'm really sorry for the trouble you've experienced. "
)

HUMAN_HANDOFF_MESSAGE = (
    "I'm connecting you with a human agent who can help with this right away."
)


# -----------------------------------------------------------------------
# Load all 4 modules once at startup
# -----------------------------------------------------------------------
def load_language_detection():
    vectorizer = joblib.load("language_detection_vectorizer.joblib")
    model = joblib.load("language_detection_model.joblib")
    label_encoder = joblib.load("language_detection_label_encoder.joblib")

    def clean_text(text: str) -> str:
        text = re.sub(r"http\S+|www\.\S+", " ", text)
        text = re.sub(r"&\w+;", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def predict(text: str) -> str:
        features = vectorizer.transform([clean_text(text)])
        pred = model.predict(features)
        return label_encoder.inverse_transform(pred)[0]

    return predict


def load_sentiment_classifier():
    sentiment_pipe = hf_pipeline(
        "text-classification",
        model="mwael399/sentiment-model",
        tokenizer="mwael399/sentiment-model",
    )

    def predict(text: str) -> str:
        result = sentiment_pipe(text, truncation=True, max_length=64)[0]
        return result["label"]

    return predict


def load_intent_classifier():
    vectorizer = joblib.load("intent_vectorizer.joblib")
    model = joblib.load("intent_bucket_model.joblib")
    label_encoder = joblib.load("intent_bucket_label_encoder.joblib")

    def clean_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r"\{\{.*?\}\}", " slot ", text)
        text = re.sub(r"[^a-z0-9\s']", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def predict(text: str) -> str:
        features = vectorizer.transform([clean_text(text)])
        pred = model.predict(features)
        return label_encoder.inverse_transform(pred)[0]

    return predict


def load_rag_pipeline():
    index = faiss.read_index("rag_knowledge_base.index")
    kb_df = pd.read_csv("rag_knowledge_base.csv")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    def retrieve(query: str, top_k: int = RAG_TOP_K) -> pd.DataFrame:
        query_emb = embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        scores, indices = index.search(query_emb, top_k)
        results = kb_df.iloc[indices[0]].copy()
        results["score"] = scores[0]
        return results

    def build_prompt(query: str, retrieved_docs: pd.DataFrame) -> str:
        context = "\n\n".join(
            f"[{row.category} / {row.intent}]\nQ: {row.instruction}\nA: {row.response}"
            for row in retrieved_docs.itertuples()
        )
        return (
            f"CONTEXT:\n{context}\n\n"
            f"CUSTOMER QUESTION:\n{query}\n\n"
            f"Answer the customer's question based on the context above."
        )

    def answer(query: str) -> str:
        retrieved_docs = retrieve(query)
        user_prompt = build_prompt(query, retrieved_docs)
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=300,
        )
        return completion.choices[0].message.content

    return answer


logger.info("Loading models...")
detect_language = load_language_detection()
detect_sentiment = load_sentiment_classifier()
detect_intent_bucket = load_intent_classifier()
rag_answer = load_rag_pipeline()
logger.info("All 4 modules loaded — service ready.")


# -----------------------------------------------------------------------
# Routing logic
# -----------------------------------------------------------------------
def route_message(text: str) -> dict:
    """Run the message through all 3 classifiers, then decide how to respond."""
    language = detect_language(text)
    sentiment = detect_sentiment(text)
    intent_bucket = detect_intent_bucket(text)

    escalate = intent_bucket in INTENT_BUCKETS_REQUIRING_HANDOFF

    if escalate:
        reply = HUMAN_HANDOFF_MESSAGE
        priority = "high" if sentiment == NEGATIVE_SENTIMENT_LABEL else "normal"
    else:
        reply = rag_answer(text)
        if sentiment == NEGATIVE_SENTIMENT_LABEL:
            reply = APOLOGETIC_PREFIX + reply
        priority = "high" if sentiment == NEGATIVE_SENTIMENT_LABEL else "normal"

    return {
        "reply": reply,
        "language": language,
        "sentiment": sentiment,
        "intent_bucket": intent_bucket,
        "escalated_to_human": escalate,
        "priority": priority,
    }


# -----------------------------------------------------------------------
# Flask app
# -----------------------------------------------------------------------
app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/chat", methods=["POST"])
def chat():
    payload = request.get_json(silent=True) or {}
    text = payload.get("message", "").strip()

    if not text:
        return jsonify({"error": "Field 'message' is required and cannot be empty."}), 400

    try:
        result = route_message(text)
    except Exception:
        logger.exception("Error while routing message")
        return jsonify({"error": "Internal error while processing the message."}), 500

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
