import json
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# 1. Load extracted pages
with open("data/extracted_output.json", "r", encoding="utf-8") as f:
    pages = json.load(f)

# 2. Clean HTML picture comments and split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = []
for i, page in enumerate(pages, start=1):
    meta = page.get("metadata", {})
    page_num = meta.get("page_number") or meta.get("page") or i
    
    clean_text = (
        page["text"]
        .replace("<!-- Start of picture text -->", "")
        .replace("<!-- End of picture text -->", "")
        .replace("<br>", " ")
    )
    
    for idx, text in enumerate(splitter.split_text(clean_text)):
        chunks.append({
            "id": f"p{page_num}_c{idx}",
            "page": int(page_num),
            "text": text
        })

print(f"Prepared {len(chunks)} clean chunks.")

# 3. Generate embeddings with your local MiniLM model
print("Generating 384D embeddings...")
model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [c["text"] for c in chunks]
embeddings = model.encode(texts, show_progress_bar=True).tolist()

# 4. Create a permanent ChromaDB database on disk inside ./chroma_db
client = chromadb.PersistentClient(path="./chroma_db")

# Delete old collection if re-running so you don't get duplicate ID errors
try:
    client.delete_collection("thermo_ch6")
except Exception:
    pass

collection = client.create_collection(
    name="thermo_ch6",
    metadata={"hnsw:space": "cosine"}  # Tell ChromaDB to use Cosine Similarity
)

# 5. Store IDs, text documents, vectors, and page metadata together
collection.add(
    ids=[c["id"] for c in chunks],
    documents=texts,
    embeddings=embeddings,
    metadatas=[{"page": c["page"]} for c in chunks]
)

print(f"Success! Saved {collection.count()} chunks permanently to ./chroma_db")