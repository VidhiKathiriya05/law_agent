import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from django.conf import settings

# Base directory (where manage.py lives)
BASE_DIR = settings.BASE_DIR.parent

# Exact files you already have
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "sc_judgments_faiss.index")
DOCS_PATH = os.path.join(BASE_DIR, "sc_judgments_texts.pkl")

# Load FAISS index
index = faiss.read_index(FAISS_INDEX_PATH)

# Load documents
with open(DOCS_PATH, "rb") as f:
    documents = pickle.load(f)

# Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_context(query, k=3):
    query_vec = embedder.encode([query])
    D, I = index.search(np.array(query_vec), k)

    return "\n\n".join([documents[i] for i in I[0]])
