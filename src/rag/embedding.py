from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
model = SentenceTransformer(EMBED_MODEL_NAME)

def create_embeddings(chunks):
    return model.encode(chunks)
