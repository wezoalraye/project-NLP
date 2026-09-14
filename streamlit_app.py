"""
Streamlit deployment for the e-commerce customer-support chatbot.

Wires together the 4 trained modules:
  1. Language Detection   (TF-IDF char n-grams + classical ML)
  2. Sentiment Classifier (fine-tuned DistilBERT)
  3. Intent Classifier    (TF-IDF word n-grams + LinearSVC, bucket-level)
  4. Q&A RAG Pipeline     (FAISS + sentence-transformers + Groq LLM)

Same routing logic as app.py (the Flask version):
  - human_handoff intent -> escalate directly, skip the LLM entirely
  - negative sentiment   -> escalate as priority + apologetic framing
  - everything else      -> answer via the RAG pipeline

Run with:
    export GROQ_API_KEY="your-key-here"      # or use a .env file (see below)
    streamlit run streamlit_app.py

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

import joblib
import pandas as pd
import faiss
import streamlit as st
from transformers import pipeline as hf_pipeline
from sentence_transformers import SentenceTransformer
from groq import Groq

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Config

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

APOLOGETIC_PREFIX = "I'm really sorry for the trouble you've experienced. "
HUMAN_HANDOFF_MESSAGE = (
    "I'm connecting you with a human agent who can help with this right away."
)


# Load all 4 modules once, cached across reruns/messages
@st.cache_resource(show_spinner="Loading language detection model...")
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


@st.cache_resource(show_spinner="Loading sentiment model...")
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


@st.cache_resource(show_spinner="Loading intent classifier...")
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


@st.cache_resource(show_spinner="Loading RAG pipeline (embeddings + knowledge base)...")
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
            f"Answer the customer's question based on the context above. "
            f"Reply in the SAME language the customer used in their question."
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


# -----------------------------------------------------------------------
# Routing logic (identical to app.py)
# -----------------------------------------------------------------------
def route_message(text: str, detect_language, detect_sentiment, detect_intent_bucket, rag_answer) -> dict:
    language = detect_language(text)
    sentiment = detect_sentiment(text)
    intent_bucket = detect_intent_bucket(text)

    escalate = intent_bucket in INTENT_BUCKETS_REQUIRING_HANDOFF
    priority = "high" if sentiment == NEGATIVE_SENTIMENT_LABEL else "normal"

    if escalate:
        reply = HUMAN_HANDOFF_MESSAGE
    else:
        reply = rag_answer(text)
        if sentiment == NEGATIVE_SENTIMENT_LABEL:
            reply = APOLOGETIC_PREFIX + reply

    return {
        "reply": reply,
        "language": language,
        "sentiment": sentiment,
        "intent_bucket": intent_bucket,
        "escalated_to_human": escalate,
        "priority": priority,
    }


# -----------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------
st.set_page_config(page_title="Customer Support Chatbot", page_icon="🛒")
st.title("🛒 E-commerce Customer Support Chatbot")
st.caption("Language detection · Sentiment analysis · Intent routing · RAG-grounded answers")

if "GROQ_API_KEY" not in os.environ:
    st.error(
        "GROQ_API_KEY is not set. Add it to a .env file in this folder "
        "(GROQ_API_KEY=your-key-here) or export it before running Streamlit."
    )
    st.stop()

detect_language = load_language_detection()
detect_sentiment = load_sentiment_classifier()
detect_intent_bucket = load_intent_classifier()
rag_answer = load_rag_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "meta" in msg:
            meta = msg["meta"]
            st.caption(
                f"🌐 {meta['language']} · 🙂 {meta['sentiment']} · "
                f"🏷️ {meta['intent_bucket']} · "
                f"{'🚨 escalated to human' if meta['escalated_to_human'] else '🤖 auto-answered'}"
            )

user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = route_message(
                user_input, detect_language, detect_sentiment, detect_intent_bucket, rag_answer
            )
        st.write(result["reply"])
        st.caption(
            f"🌐 {result['language']} · 🙂 {result['sentiment']} · "
            f"🏷️ {result['intent_bucket']} · "
            f"{'🚨 escalated to human' if result['escalated_to_human'] else '🤖 auto-answered'}"
        )

    st.session_state.messages.append(
        {"role": "assistant", "content": result["reply"], "meta": result}
    )
