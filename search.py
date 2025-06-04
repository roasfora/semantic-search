import faiss
import numpy as np

def build_faiss_index(embeddings):
    embeddings = np.array(embeddings).astype('float32')
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # IP = Inner Product = Cosine sim (when normalized)
    index.add(embeddings)

    return index

def search(index, query_embedding, top_k=3):
    query_embedding = np.array([query_embedding]).astype('float32')
    faiss.normalize_L2(query_embedding)  # Normalize query as well

    D, I = index.search(query_embedding, top_k)
    return I[0], D[0]