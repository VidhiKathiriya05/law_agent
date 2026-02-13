import chromadb
from sentence_transformers import SentenceTransformer

DB_DIR = "../vector_db"

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_collection(name="ipc_collection")

query = input("Ask your legal question: ")

query_embedding = model.encode(query).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

print("\n🔎 Top Results:\n")

for i in range(len(results["documents"][0])):
    print(f"Result {i+1}")
    print("Document:", results["documents"][0][i][:500])
    print("Metadata:", results["metadatas"][0][i])
    print("-" * 50)