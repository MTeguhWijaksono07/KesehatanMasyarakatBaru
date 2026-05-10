import numpy as np
from .graph_db import neo4j_handler
from .embedding import model as embed_model
from ..llm.generator import generate_answer
from rank_bm25 import BM25Okapi
import os

_chunk_cache = None
_bm25_index = None
_embedding_matrix = None

def load_resources():
    global _chunk_cache, _bm25_index, _embedding_matrix
    if _chunk_cache is not None:
        return _chunk_cache, _bm25_index, _embedding_matrix
    
    print('Loading chunks dari Neo4j...')
    _chunk_cache = neo4j_handler.get_all_chunks()
    
    if not _chunk_cache:
        print('No chunks found in Neo4j')
        return [], None, None

    # Pre-compute BM25
    tokenized_corpus = [c.get('content', '').lower().split() for c in _chunk_cache]
    _bm25_index = BM25Okapi(tokenized_corpus)
    
    # Pre-compute Embedding Matrix for vectorized search
    embeddings = []
    valid_chunks = []
    for c in _chunk_cache:
        if c.get('embedding'):
            embeddings.append(c['embedding'])
            valid_chunks.append(c)
    
    if embeddings:
        _embedding_matrix = np.array(embeddings, dtype=np.float32)
        _chunk_cache = valid_chunks # Only keep chunks with embeddings for consistency
        
    print(f'{len(_chunk_cache)} chunks di-cache dan index siap')
    return _chunk_cache, _bm25_index, _embedding_matrix

def hyde_transform(query):
    """Buat dokumen hipotetis dari query menggunakan LLM, lalu embed."""
    prompt = (
        'Anda adalah pakar kesehatan masyarakat Indonesia. '
        'Tulis paragraf singkat (3-4 kalimat) yang menjelaskan topik berikut '
        'dari sudut pandang kesehatan masyarakat berbasis bukti:\n\n'
        f'Topik: {query}\n\nParagraf:'
    )
    hypo_doc = generate_answer(prompt)
    
    orig_vec = embed_model.encode(query, normalize_embeddings=True)
    hypo_vec = embed_model.encode(hypo_doc, normalize_embeddings=True)
    
    # Bobot: 70% original query + 30% hypothetical document (adjusted for stability)
    ensemble = 0.7 * orig_vec + 0.3 * hypo_vec
    ensemble /= np.linalg.norm(ensemble)
    return hypo_doc, ensemble

def dense_retrieve(query_vec, top_k=5):
    chunks, _, embedding_matrix = load_resources()
    if embedding_matrix is None:
        return []
    
    # Vectorized cosine similarity: (N, D) @ (D,) -> (N,)
    similarities = np.dot(embedding_matrix, query_vec)
    
    top_indices = np.argsort(similarities)[::-1][:top_k]
    results = []
    for idx in top_indices:
        results.append({**chunks[idx], 'dense_score': float(similarities[idx])})
    return results

def sparse_retrieve(query, top_k=5):
    chunks, bm25, _ = load_resources()
    if bm25 is None:
        return []
    
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({**chunks[idx], 'sparse_score': float(scores[idx])})
    return results

def retrieve_from_graph(query, top_k=3):
    # 1. Get query vector (with optional HyDE)
    try:
        # HyDE can be slow, so we could skip it for very short queries
        if len(query.split()) > 3:
            _, query_vec = hyde_transform(query)
        else:
            query_vec = embed_model.encode(query, normalize_embeddings=True)
    except Exception as e:
        print(f'Retrieval prep error: {e}')
        query_vec = embed_model.encode(query, normalize_embeddings=True)
    
    # 2. Hybrid Search
    dense_results = dense_retrieve(query_vec, top_k=top_k*2)
    sparse_results = sparse_retrieve(query, top_k=top_k*2)
    
    # 3. Simple RRF or Reciprocal Rank Fusion / Score Combination
    # For simplicity, we'll just merge and deduplicate, prioritizing dense but keeping sparse
    combined = {}
    
    for r in dense_results:
        cid = r['chunk_id']
        combined[cid] = r
        combined[cid]['hybrid_score'] = r['dense_score'] * 0.7 # Weighting
        
    for r in sparse_results:
        cid = r['chunk_id']
        if cid in combined:
            combined[cid]['hybrid_score'] += (r['sparse_score'] / (max(sparse_results, key=lambda x: x['sparse_score'])['sparse_score'] + 1e-6)) * 0.3
        else:
            combined[cid] = r
            combined[cid]['hybrid_score'] = (r['sparse_score'] / (max(sparse_results, key=lambda x: x['sparse_score'])['sparse_score'] + 1e-6)) * 0.3

    sorted_results = sorted(combined.values(), key=lambda x: x.get('hybrid_score', 0), reverse=True)
    return sorted_results[:top_k]

def format_rag_context(chunks):
    if not chunks:
        return '[TIDAK ADA KONTEKS RELEVAN DITEMUKAN DALAM DATABASE]'
    
    context_parts = []
    for i, c in enumerate(chunks, 1):
        content = c.get('content', '').strip()
        bab = c.get('chapter', 'N/A')
        # Format lebih terstruktur agar LLM mudah membedakan antar sumber
        part = (
            f"--- DATA REFERENSI {i} ---\n"
            f"KONTEN: {content}\n"
            f"LOKASI: Bab {bab}\n"
            "--------------------------"
        )
        context_parts.append(part)
    
    return "\n\n".join(context_parts)
