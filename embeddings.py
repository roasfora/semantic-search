from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer

# TF-IDF
def get_tfidf_embeddings(corpus):
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(corpus)
    return embeddings.toarray(), vectorizer

# Word2Vec
def get_word2vec_embeddings(corpus):
    tokenized = [doc.split() for doc in corpus]
    model = Word2Vec(sentences=tokenized, vector_size=100, window=5, min_count=1, workers=1)
    embeddings = [
        sum(model.wv[word] for word in doc if word in model.wv) / len(doc)
        for doc in tokenized
    ]
    return embeddings, model

# SBERT
def get_sbert_embeddings(corpus, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(corpus)
    return embeddings, model
