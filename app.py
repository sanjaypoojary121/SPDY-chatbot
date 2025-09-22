import streamlit as st
import json
import google.generativeai as genai
import numpy as np
import nltk
from nltk.corpus import wordnet
from qdrant_client import QdrantClient

# ---------------- Setup ----------------
# Configure Gemini (for generating answers only)
genai.configure(api_key='AIzaSyAksG3SchiWu_26iSAcLPjwAq6JCGaAeGA')
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# Qdrant Setup
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.O8LVZ3Yqqe1AuIy1h5ZpD6HR7UQgxoGDL1Ieo7A5QY8"
QDRANT_URL = "https://f8ca4079-9b8d-4d2f-a596-924c8054e845.europe-west3-0.gcp.cloud.qdrant.io"
COLLECTION_NAME = "mite-chatbot"

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60.0)

# ---------------- NLP Helpers ----------------
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

def preprocess_with_synonyms(query: str) -> str:
    """Expand query by adding synonyms to improve retrieval."""
    words = query.split()
    expanded = []
    for word in words:
        synonyms = wordnet.synsets(word)
        if synonyms:
            expanded.append(synonyms[0].lemmas()[0].name().replace("_", " "))
        expanded.append(word)
    return " ".join(expanded)

# --- AMIR Classification ---
def classify_query(query: str) -> str:
    query = query.lower()
    if any(kw in query for kw in ["define", "what is", "fees", "duration", "admission", "when"]):
        return "factual"
    elif any(kw in query for kw in ["how", "python", "write code", "syntax", "function"]):
        return "technical"
    elif any(kw in query for kw in ["compare", "difference", "better", "advantage"]):
        return "complex"
    else:
        return "conversational"

# --- Retrieval with Qdrant ---
def retrieve_with_qdrant(query, k=5):
    """
    Retrieve top-k chunks from Qdrant using embeddings.
    """
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("all-MiniLM-L6-v2")  # same model used for upload
    query_vec = embedder.encode(query).tolist()  # embed the user query

    res = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vec,
        limit=k
    )

    results = []
    scores = []
    for hit in res:
        results.append(hit.payload.get("text", ""))
        scores.append(hit.score)

    # Debug: print top results and similarity scores
    print("DEBUG: Retrieved Chunks with Scores:")
    for r, s in zip(results, scores):
        print(f"{s:.3f} -> {r[:60]}...")  # first 60 chars

    return results, scores


# --- No Data Fallback ---
def filter_by_confidence(results, confidences, threshold=0.2):
    """
    Keep results only if max confidence > threshold
    Lowered threshold to 0.2 to avoid missing valid chunks
    """
    if not results:
        return ["No Data Available"]
    if max(confidences) > threshold:
        return results
    return ["No Data Available"]


# --- AMIR Pipeline with fallback ---
def amir_pipeline(query):
    qtype = classify_query(query)
    results, confidences = retrieve_with_qdrant(query)
    filtered = filter_by_confidence(results, confidences, threshold=0.2)

    # 🔄 fallback: if "No Data Available", retry with synonym expansion
    if filtered == ["No Data Available"]:
        expanded_query = preprocess_with_synonyms(query)
        results, confidences = retrieve_with_qdrant(expanded_query)
        filtered = filter_by_confidence(results, confidences, threshold=0.15)

    return filtered

# --- Generate Response with Gemini ---
def generate_response_with_gemini(query, retrieved_chunks):
    if retrieved_chunks == ["No Data Available"]:
        return "❌ No data available to answer this question."

    context = "\n".join(retrieved_chunks)
    prompt = f"""
You are a helpful assistant trained on MITE college information.

Based on the following context, answer the user's question. 
If the answer is not present, say "No Data Available" and do not make anything up.

Context:
{context}

Question:
{query}
"""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Error generating response: {e}"

# ----------------- Streamlit UI --------------------
st.set_page_config(page_title="MITE Chatbot", page_icon="🤖")
st.title("🤖 MITE Chatbot (RAG + AMIR + Synonyms + Qdrant)")

if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.text_input("Ask your question here:")

if st.button("Ask") and user_input:
    st.session_state.messages.append({"role": "user", "text": user_input})
    retrieved_chunks = amir_pipeline(user_input)
    bot_response = generate_response_with_gemini(user_input, retrieved_chunks)
    st.session_state.messages.append({"role": "bot", "text": bot_response})

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"🧑 **You:** {msg['text']}")
    else:
        st.markdown(f"🤖 **Bot:** {msg['text']}")
