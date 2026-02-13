import os
import json
import chromadb
from sentence_transformers import SentenceTransformer

DATA_DIR = "../data"
DB_DIR = "../vector_db"

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=DB_DIR)

try:
    client.delete_collection("ipc_collection")
except:
    pass

collection = client.get_or_create_collection(name="ipc_collection")

print("Reading JSON files...")

id_counter = 0

for file in os.listdir(DATA_DIR):
    if file.endswith(".json"):
        act_name = file.replace(".json", "").upper()
        print(f"Processing {file}...")

        with open(os.path.join(DATA_DIR, file), "r", encoding="utf-8") as f:
            data = json.load(f)

            if isinstance(data, list):
                for section in data:

                    # Extract possible fields safely
                    section_number = str(
                        section.get("Section") or 
                        section.get("section") or 
                        section.get("id") or ""
                    )

                    title = str(
                        section.get("section_title") or ""
                    )

                    description = str(
                        section.get("section_desc") or 
                        section.get("description") or 
                        section.get("text") or ""
                    )

                    # Combine EVERYTHING
                    full_text = f"""
                    Act: {act_name}
                    Section: {section_number}
                    Title: {title}
                    Content: {description}
                    """

                    # Always insert (even if empty)
                    embedding = model.encode(full_text).tolist()

                    collection.add(
                        ids=[str(id_counter)],
                        embeddings=[embedding],
                        documents=[full_text],
                        metadatas=[{
                            "act": act_name,
                            "section": section_number
                        }]
                    )

                    id_counter += 1

print("✅ Vector database rebuilt successfully!")
print("Total documents inserted:", id_counter)