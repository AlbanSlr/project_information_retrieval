"""
fasttext_search_engine_cached.py
--------------------------------
Moteur de recherche sémantique basé sur FastText pour un corpus Wikipédia.
Inclut la mise en cache des vecteurs de documents (gain de temps énorme).

"""

import os
import json
import numpy as np
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# =============================
# 1. CONSTANTES ET CHEMINS
# =============================

DATA_PATH = 'wiki_split_extract_2k/'
QUERIES_FILE = 'requetes.jsonl'
FASTTEXT_PATH = 'cc.fr.300.vec'      # modèle FastText français (à télécharger)
DOC_VECTORS_FILE = 'doc_vectors.npy' # fichier de cache pour les vecteurs
DOC_NAMES_FILE = 'doc_names.json'    # pour retrouver le nom des docs
TOP_K = 5


# =============================
# 2. FONCTIONS DE CHARGEMENT
# =============================

def load_documents(data_path):
    """Charge tous les fichiers texte dans un dictionnaire."""
    docs = {}
    for file_name in tqdm(os.listdir(data_path), desc="Chargement des documents"):
        if file_name.endswith('.txt'):
            with open(os.path.join(data_path, file_name), 'r', encoding='utf-8') as f:
                docs[file_name] = f.read().strip().lower()
    return docs


def load_queries(queries_file):
    """Charge les requêtes et fichiers attendus depuis le JSONL."""
    queries = []
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            for q in data["Queries"]:
                queries.append((q, data["Answer file"]))
    return queries


# =============================
# 3. FASTTEXT : EMBEDDINGS
# =============================

def load_fasttext_model(model_path):
    """Charge le modèle FastText pré-entraîné (format .vec)."""
    print("Chargement du modèle FastText…")
    model = KeyedVectors.load_word2vec_format(model_path)
    print(f"✅ Modèle chargé ({len(model.key_to_index)} mots)")
    return model


def text_to_vector(text, model):
    """Retourne le vecteur moyen FastText d'un texte (liste de mots connus)."""
    words = [w for w in text.split() if w in model]
    if not words:
        return np.zeros(model.vector_size)
    vectors = np.array([model[w] for w in words])
    return np.mean(vectors, axis=0)


def build_or_load_doc_vectors(docs, model, vec_file, names_file):
    """
    Soit charge les vecteurs depuis un cache .npy,
    soit les calcule et les sauvegarde.
    """
    if os.path.exists(vec_file) and os.path.exists(names_file):
        print("Chargement des vecteurs de documents depuis le cache…")
        matrix = np.load(vec_file)
        with open(names_file, 'r', encoding='utf-8') as f:
            doc_names = json.load(f)
        print(f"✅ {len(doc_names)} vecteurs chargés.")
        return doc_names, matrix

    print("Calcul des vecteurs FastText pour tous les documents…")
    doc_names = list(docs.keys())
    matrix = []
    for name in tqdm(doc_names, desc="Vectorisation"):
        vec = text_to_vector(docs[name], model)
        matrix.append(vec)
    matrix = np.vstack(matrix)

    # Sauvegarde du cache
    np.save(vec_file, matrix)
    with open(names_file, 'w', encoding='utf-8') as f:
        json.dump(doc_names, f)
    print(f"✅ Vecteurs sauvegardés dans '{vec_file}' ({len(doc_names)} documents).")

    return doc_names, matrix


# =============================
# 4. RECHERCHE
# =============================

def search_query(query, model, doc_names, matrix, top_k=5):
    """Retourne les top_k documents les plus similaires à la requête."""
    query_vec = text_to_vector(query, model).reshape(1, -1)
    sims = cosine_similarity(query_vec, matrix)[0]
    ranked = sorted(zip(doc_names, sims), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


# =============================
# 5. ÉVALUATION
# =============================

def evaluate_model(queries, model, doc_names, matrix, top_k=5):
    """Évalue la qualité du moteur de recherche avec le fichier de requêtes."""
    total = len(queries)
    correct = 0
    mrr_sum = 0

    for query, answer in tqdm(queries, desc="Évaluation des requêtes"):
        results = search_query(query, model, doc_names, matrix, top_k=top_k)
        ranked_docs = [r[0] for r in results]

        # Précision à 1
        if ranked_docs[0] == answer:
            correct += 1

        # MRR (Mean Reciprocal Rank)
        if answer in ranked_docs:
            rank = ranked_docs.index(answer) + 1
            mrr_sum += 1 / rank

    accuracy = correct / total
    mrr = mrr_sum / total
    return {"accuracy": accuracy, "mrr": mrr, "total_queries": total}


# =============================
# 6. MAIN
# =============================

def main():
    print("=== Moteur de recherche FastText (Français) ===\n")

    # 1. Charger les données
    docs = load_documents(DATA_PATH)
    queries = load_queries(QUERIES_FILE)

    # 2. Charger le modèle FastText
    model = load_fasttext_model(FASTTEXT_PATH)

    # 3. Charger ou créer les vecteurs documents
    doc_names, matrix = build_or_load_doc_vectors(docs, model, DOC_VECTORS_FILE, DOC_NAMES_FILE)

    # 4. Exemple de requête
    example_query = "course à pied"
    print(f"\nExemple de recherche : '{example_query}'\n")
    results = search_query(example_query, model, doc_names, matrix, top_k=TOP_K)
    for name, score in results:
        print(f"{name:25s}  ->  score={score:.4f}")

    # 5. Évaluation globale
    print("\nÉvaluation globale sur les requêtes de test...")
    metrics = evaluate_model(queries, model, doc_names, matrix, top_k=TOP_K)

    # 6. Résumé final
    print("\n=== Résumé de l'évaluation ===")
    print(f"Nombre de requêtes : {metrics['total_queries']}")
    print(f"Précision@1 : {metrics['accuracy']*100:.2f}%")
    print(f"MRR (Mean Reciprocal Rank) : {metrics['mrr']:.4f}")
    print("\n✅ Évaluation terminée.")


# =============================
# POINT D’ENTRÉE
# =============================

if __name__ == "__main__":
    main()
