import os
import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer

# 1. Load GEMINI_API_KEY from .env and initialize the Gemini client on v1beta
load_dotenv()
gemini_client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(api_version="v1beta")
)

# 2. Connect to your permanent ChromaDB on disk (no re-embedding needed!)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="thermo_ch6")

# 3. Load the same embedding model used to build the database
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
# 4. Ask a question and convert it into a 384D vector
query = "How is shaft work calculated in an adiabatic gas compressor?"
query_vector = embed_model.encode([query]).tolist()

# 5. Retrieve the top 4 closest chunks and their page numbers from ChromaDB
results = collection.query(
    query_embeddings=query_vector,
    n_results=4
)

# 6. Format retrieved chunks with page labels for the LLM
context_blocks = []
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    context_blocks.append(f"[Page {meta['page']}]:\n{doc}")

context_text = "\n\n---\n\n".join(context_blocks)

# 7. Send the question + retrieved context to Gemini with strict grounding rules
response = gemini_client.models.generate_content(
    model="gemini-2.5-flash",
    contents=f"Context:\n{context_text}\n\nQuestion: {query}",
    config=types.GenerateContentConfig(
        system_instruction=(
            "You are a precise engineering assistant. Answer the user's question "
            "using ONLY the provided textbook context. Cite the page number like [Page X] "
            "for every fact or equation. If the answer is not in the context, say "
            "'I cannot find this in the provided document.'"
        ),
        temperature=0.1
    )
)

print(f"\nQuestion: {query}\n" + "=" * 60)
print(response.text)