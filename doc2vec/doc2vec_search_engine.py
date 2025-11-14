import os
import json
import numpy as np
from tqdm import tqdm
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# ============================
# 1) Chargement des documents
# ============================

def load_documents(data_path):
    files = [f for f in os.listdir(data_path) if f.endswith(".txt")]
    docs = {}
    for f in tqdm(files, desc="Chargement des documents"):
        with open(os.path.join(data_path, f), "r", encoding="utf-8") as fd:
            text = fd.read().lower()
            docs[f] = text
    return docs


# ============================
# 2) Préparer corpus Doc2Vec
# ============================

def build_tagged_documents(documents):
    tagged_docs = []
    for i, (fname, text) in enumerate(documents.items()):
        words = text.split()
        tagged_docs.append(TaggedDocument(words=words, tags=[fname]))
    return tagged_docs


# ============================
# 3) Entraîner Doc2Vec
# ============================

def train_doc2vec(tagged_docs, vector_size=200, epochs=40):
    print("🔧 Entraînement du modèle Doc2Vec…")
    model = Doc2Vec(
        vector_size=vector_size,
        window=10,
        min_count=2,
        workers=4,
        dm=1   # DBOW=0, DM=1 → DM donne de meilleurs résultats sémantiques
    )
    model.build_vocab(tagged_docs)
    model.train(tagged_docs, total_examples=model.corpus_count, epochs=epochs)
    print("✅ Modèle Doc2Vec entraîné")
    return model


# ============================
# 4) Calcul + cache des vecteurs documents
# ============================

def compute_or_load_doc_vectors(model, documents, cache_path="doc_vectors.pkl"):
    if os.path.exists(cache_path):
        print("Chargement des vecteurs documents depuis le cache…")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print("Calcul des vecteurs documents…")
    doc_vectors = {}
    for fname in tqdm(documents.keys()):
        doc_vectors[fname] = model.dv[fname]

    with open(cache_path, "wb") as f:
        pickle.dump(doc_vectors, f)

    print("✅ Vecteurs sauvegardés dans", cache_path)
    return doc_vectors


# ============================
# 5) Embedding de la requête
# ============================

def embed_query(model, query):
    words = query.lower().split()
    return model.infer_vector(words)


# ============================
# 6) Fonction de recherche
# ============================

def search(model, doc_vectors, query, top_k=5):
    q_vec = embed_query(model, query)

    # Matrice des vecteurs documents
    doc_names = list(doc_vectors.keys())
    matrix = np.array([doc_vectors[f] for f in doc_names])

    sims = cosine_similarity([q_vec], matrix)[0]

    # Tri décroissant
    top_idx = np.argsort(sims)[::-1][:top_k]

    return [(doc_names[i], sims[i]) for i in top_idx]


# ============================
# 7) Évaluation
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


def evaluate(model, doc_vectors, queries):
    ranks = []
    hits = 0

    for q, answer_file in tqdm(queries, desc="Évaluation des requêtes"):
        results = search(model, doc_vectors, q, top_k=2000)
        ranked_files = [r[0] for r in results]

        if answer_file in ranked_files:
            rank = ranked_files.index(answer_file) + 1
            ranks.append(1 / rank)
            if rank == 1:
                hits += 1
        else:
            ranks.append(0)

    precision1 = hits / len(queries)
    mrr = sum(ranks) / len(queries)

    return precision1, mrr


# ============================
# MAIN — exécute l’ensemble
# ============================

if __name__ == "__main__":
    DATA_PATH = "wiki_split_extract_2k/"

    print("=== Moteur de recherche Doc2Vec ===")

    # 1) Charger documents
    documents = load_documents(DATA_PATH)

    # 2) Corpus TaggedDocument
    tagged_docs = build_tagged_documents(documents)

    # 3) Entraîner modèle
    model = train_doc2vec(tagged_docs)

    # 4) Charger ou calculer les vecteurs
    doc_vectors = compute_or_load_doc_vectors(model, documents)

    # 5) Test rapide
    test_query = "course à pied"
    print("\nExemple de recherche :", test_query)
    for fname, score in search(model, doc_vectors, test_query):
        print(f"{fname:<25}  score={score:.4f}")

    # 6) Évaluation complète
    queries = load_queries()
    p1, mrr = evaluate(model, doc_vectors, queries)

    print("\n=== Résultats finaux ===")
    print(f"Nombre de requêtes : {len(queries)}")
    print(f"Precision@1 : {p1*100:.2f}%")
    print(f"MRR : {mrr:.4f}")
    print("Evaluation terminée.")
