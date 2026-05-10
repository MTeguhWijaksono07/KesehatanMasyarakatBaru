from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_similarity(answer1, answer2):

    emb1 = model.encode([answer1])
    emb2 = model.encode([answer2])

    sim = cosine_similarity(emb1, emb2)[0][0]

    return sim
