import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ============================
# 1) Charger documents
# ============================

def load_documents(data_path):
    files = [f for f in os.listdir(data_path) if f.endswith(".txt")]
    docs = {}
    for f in tqdm(files, desc="Chargement des documents"):
        with open(os.path.join(data_path, f), "r", encoding="utf-8") as fd:
            docs[f] = fd.read()
    return docs


# ============================
# 2) Charger SBERT
# ============================

def load_sbert_model():
    print("📥 Chargement du modèle SBERT multilingue…")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    print("✅ Modèle chargé")
    return model


# ============================
# 3) Calcul + cache des embeddings documents
# ============================

def compute_or_load_doc_vectors(model, documents, cache_path="sbert_vectors.pkl"):
    if os.path.exists(cache_path):
        print("📦 Chargement des vecteurs SBERT depuis le cache…")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print("🧠 Calcul des embeddings SBERT pour les documents…")
    doc_names = list(documents.keys())
    contents = [documents[name] for name in doc_names]

    vectors = model.encode(contents, batch_size=32, show_progress_bar=True)

    doc_vectors = {name: vec for name, vec in zip(doc_names, vectors)}

    with open(cache_path, "wb") as f:
        pickle.dump(doc_vectors, f)

    print("✅ Vecteurs sauvegardés dans", cache_path)
    return doc_vectors


# ============================
# 4) Embedding requête
# ============================

def embed_query(model, query):
    return model.encode([query])[0]


# ============================
# 5) Recherche
# ============================

def search(model, doc_vectors, query, top_k=5):
    q_vec = embed_query(model, query)

    doc_names = list(doc_vectors.keys())
    matrix = np.array([doc_vectors[f] for f in doc_names])

    sims = cosine_similarity([q_vec], matrix)[0]

    top_idx = np.argsort(sims)[::-1][:top_k]

    return [(doc_names[i], sims[i]) for i in top_idx]


# ============================
# 6) Charger requêtes
# ============================

def load_queries(path="requetes.jsonl"):
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            answer = obj["Answer file"]
            for q in obj["Queries"]:
                queries.append((q, answer))
    return queries


# ============================
# 7) Évaluation
# ============================

def evaluate(model, doc_vectors, queries):
    ranks = []
    hits = 0

    for q, answer_file in tqdm(queries, desc="Évaluation des requêtes"):
        results = search(model, doc_vectors, q, top_k=2000)
        ranked = [r[0] for r in results]

        if answer_file in ranked:
            rank = ranked.index(answer_file) + 1
            ranks.append(1 / rank)
            if rank == 1:
                hits += 1
        else:
            ranks.append(0)

    precision1 = hits / len(queries)
    mrr = sum(ranks) / len(queries)

    return precision1, mrr


# ============================
# MAIN COMPLET
# ============================

if __name__ == "__main__":
    DATA_PATH = "wiki_split_extract_2k/"

    print("=== Moteur de recherche SBERT Multilingue ===")

    # 1) Charger documents
    documents = load_documents(DATA_PATH)

    # 2) Charger SBERT
    model = load_sbert_model()

    # 3) Embeddings (avec cache)
    doc_vectors = compute_or_load_doc_vectors(model, documents)

    # 4) Test rapide
    query_test = "course à pied"
    print("\nExemple de recherche :", query_test)
    for fname, score in search(model, doc_vectors, query_test):
        print(f"{fname:<25}  score={score:.4f}")

    # 5) Évaluation globale
    queries = load_queries()
    p1, mrr = evaluate(model, doc_vectors, queries)

    print("\n=== Résultats finaux ===")
    print(f"Nombre de requêtes : {len(queries)}")
    print(f"Precision@1 : {p1*100:.2f}%")
    print(f"MRR : {mrr:.4f}")
    print("🏁 Évaluation terminée.")
