import json
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from tqdm import tqdm   # progress bar

# ---------- CONFIG ----------
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.O8LVZ3Yqqe1AuIy1h5ZpD6HR7UQgxoGDL1Ieo7A5QY8"
QDRANT_URL = "https://f8ca4079-9b8d-4d2f-a596-924c8054e845.europe-west3-0.gcp.cloud.qdrant.io"
COLLECTION_NAME = "mite-chatbot"
DIMENSION = 384  # embedding size for all-MiniLM-L6-v2
BATCH_SIZE = 100  # reduced batch size to avoid timeout

# ---------- CONNECT ----------
print("🔄 Connecting to Qdrant...")
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=60.0   # ⏱️ increased timeout from default 5s to 60s
)

# ---------- CREATE COLLECTION IF NEEDED ----------
collections = [c.name for c in client.get_collections().collections]
if COLLECTION_NAME not in collections:
    print("📦 Creating collection:", COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=rest.VectorParams(size=DIMENSION, distance=rest.Distance.COSINE)
    )

# ---------- EMBEDDING SETUP ----------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ---------- LOAD & CLEAN CHUNKS ----------
print("📂 Loading chunks from mite_website_chunks.txt...")
with open("mite_website_chunks.txt", "r", encoding="utf-8") as f:
    chunks = [line.strip() for line in f.readlines() if line.strip()]

print(f"✅ Loaded {len(chunks)} chunks")

# ---------- BUILD POINTS ----------
points = []
for i, chunk in enumerate(chunks):
    emb = embedder.encode(chunk).tolist()
    points.append(rest.PointStruct(id=i, vector=emb, payload={"text": chunk}))

# ---------- UPLOAD IN BATCHES ----------
print(f"⬆️ Uploading {len(points)} chunks to Qdrant in batches of {BATCH_SIZE}...")

for i in tqdm(range(0, len(points), BATCH_SIZE), desc="Uploading"):
    batch = points[i: i + BATCH_SIZE]
    client.upsert(collection_name=COLLECTION_NAME, points=batch)

print("🎉 Done! All chunks uploaded to Qdrant.")
