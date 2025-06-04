import streamlit as st
import numpy as np
from embeddings import get_sbert_embeddings, get_tfidf_embeddings, get_word2vec_embeddings
from search import build_faiss_index, search

# Load documents
with open("data/docs.txt", encoding="utf-8") as f:
    docs = [line.strip() for line in f if line.strip()]

# Streamlit UI
st.title("🔍 Semantic Search Demo")
query = st.text_input("Enter your search query:")
method = st.selectbox("Choose embedding method:", ["SBERT", "TF-IDF", "Word2Vec"])

# Cache embeddings
@st.cache_resource
def get_cached_embeddings(method):
    if method == "SBERT":
        return get_sbert_embeddings(docs)
    elif method == "TF-IDF":
        return get_tfidf_embeddings(docs)
    elif method == "Word2Vec":
        return get_word2vec_embeddings(docs)

if query:
    doc_embeddings, model_or_vectorizer = get_cached_embeddings(method)

    if method == "SBERT":
        query_embedding = model_or_vectorizer.encode([query])[0]

    elif method == "TF-IDF":
        query_embedding = model_or_vectorizer.transform([query]).toarray()[0]

    elif method == "Word2Vec":
        tokens = query.lower().split()
        valid_tokens = [w for w in tokens if w in model_or_vectorizer.wv]
        if not valid_tokens:
            st.warning("Query contains no recognizable words for Word2Vec.")
            st.stop()
        query_embedding = np.mean([model_or_vectorizer.wv[w] for w in valid_tokens], axis=0)

    # Normalize and search
    index = build_faiss_index(doc_embeddings)
    results, scores = search(index, query_embedding)

    st.subheader("Top Results:")
    for i, score in zip(results, scores):
        question, answer = docs[i].split("?", 1)
        st.markdown(f"**Q:** {question.strip()}?")
        st.markdown(f"**A:** {answer.strip()}")
        st.markdown(f"**Score:** `{score:.2f}`")
        st.text("—" * 40)