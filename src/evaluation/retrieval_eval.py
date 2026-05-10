def evaluate_retrieval(retrieved_chunks, expected_keywords):

    total_found = 0

    for chunk in retrieved_chunks:
        for keyword in expected_keywords:

            if keyword.lower() in chunk.lower():
                total_found += 1

    score = total_found / len(expected_keywords)

    return min(score, 1.0)
