import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

with open("data/extracted_output.json", "r", encoding="utf-8") as f:
    pages = json.load(f)

print(f"Loaded {len(pages)} pages from JSON.")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = []
for i, page in enumerate(pages, start=1):
    meta = page.get("metadata", {})
   
    page_num = meta.get("page_number") or meta.get("page") or i
    page_chunks = splitter.split_text(page["text"])
    
    for idx, text in enumerate(page_chunks):
        chunks.append({
            "chunk_id": f"p{page_num}_c{idx}",
            "page": page_num,
            "text": text
        })

print(f"Created {len(chunks)} chunks. Sample chunk 0:\n{chunks[0]}")

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [c["text"] for c in chunks]
chunk_embeddings = model.encode(texts, show_progress_bar=True)

print(f"Embeddings shape: {chunk_embeddings.shape}")

query = "What are maxwell relations?"
query_embedding = model.encode(query)


scores = cos_sim(query_embedding, chunk_embeddings)[0]


top_indices = scores.argsort(descending=True)[:3]

print(f"\nQuery: '{query}'\n" + "=" * 60)
for rank, idx in enumerate(top_indices, start=1):
    idx = int(idx)
    print(f"\nRank #{rank} | Score: {scores[idx]:.4f} | Page {chunks[idx]['page']}")
    print("-" * 40)
    print(chunks[idx]["text"][:450])