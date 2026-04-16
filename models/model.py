from sentence_transformers import SentenceTransformer

# Load once (important)
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")