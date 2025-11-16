def calculate_precision_at_k(expected_file, results, k):
    # on calcule la précision comme l'inverse du rang (MRR)
    if not results:
        return 0
    retrieved_files = [filename for filename, score in results[:k]]
    # on retourne 1/rang si le document est trouvé, 0 sinon
    if expected_file in retrieved_files:
        return 1 / (retrieved_files.index(expected_file) + 1)
    return 0

def calculate_recall_at_k(expected_file, results, k):
    # on calcule le recall binaire: 1 si trouvé dans le top-k, 0 sinon
    if not results:
        return 0
    retrieved_files = [filename for filename, score in results[:k]]
    return 1 if expected_file in retrieved_files else 0