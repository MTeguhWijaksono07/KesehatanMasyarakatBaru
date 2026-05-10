from sklearn.metrics.pairwise import cosine_similarity

def retrieve(query_embedding, doc_embeddings, documents, top_k=3):

    similarities = cosine_similarity(
        [query_embedding],
        doc_embeddings
    )[0]

    ranked = similarities.argsort()[::-1][:top_k]

    return [documents[i] for i in ranked]
