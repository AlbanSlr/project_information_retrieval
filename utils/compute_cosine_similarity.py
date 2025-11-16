import math

def compute_cosine_similarity(query_vector, doc_vector):
    # on calcule la similarité cosine entre deux vecteurs (dictionnaires)
    # formule: cos(theta) = (A·B) / (||A|| * ||B||)
    
    # on calcule le produit scalaire en combinant tous les termes
    dot_product = sum(
        query_vector.get(word, 0) * doc_vector.get(word, 0) 
        for word in set(query_vector.keys()) | set(doc_vector.keys())
    )
    
    # on calcule les normes (magnitudes) de chaque vecteur
    query_magnitude = math.sqrt(sum(val ** 2 for val in query_vector.values()))
    doc_magnitude = math.sqrt(sum(val ** 2 for val in doc_vector.values()))
    
    # on évite la division par zéro si un vecteur est nul
    if query_magnitude == 0 or doc_magnitude == 0:
        return 0
    
    # on retourne la similarité cosine normalisée entre 0 et 1
    return dot_product / (query_magnitude * doc_magnitude)