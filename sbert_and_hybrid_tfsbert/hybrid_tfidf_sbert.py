import os
import json
import pickle
import numpy as np
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

import nltk
from nltk.corpus import stopwords

# ======================================
# 0) Préparer les stopwords français
# ======================================
nltk.download("stopwords")
french_stopwords = stopwords.words("french")

# ======================================
# 1) Charger documents
# ======================================
def load_documents(data_path):
    files = [f for f in os.listdir(data_path) if f.endswith(".txt")]
    docs = {}
    for f in tqdm(files, desc="Chargement des documents"):
        with open(os.path.join(data_path, f), "r", encoding="utf-8") as fd:
            docs[f] = fd.read()
    return docs

# ======================================
# 2) TF-IDF
# ======================================
def load_or_fit_tfidf(documents, cache="tfidf_model.pkl"):
    if os.path.exists(cache):
        print("📦 Chargement TF-IDF depuis le cache…")
        with open(cache, "rb") as f:
            vectorizer, matrix = pickle.load(f)
        return vectorizer, matrix

    print("🧠 Apprentissage TF-IDF…")
    vectorizer = TfidfVectorizer(stop_words=french_stopwords, max_features=50000)
    doc_texts = list(documents.values())
    matrix = vectorizer.fit_transform(doc_texts)

    with open(cache, "wb") as f:
        pickle.dump((vectorizer, matrix), f)

    print("✅ TF-IDF entraîné et sauvegardé.")
    return vectorizer, matrix

# ======================================
# 3) SBERT
# ======================================
def load_sbert_model():
    print("📥 Chargement SBERT…")
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def load_sbert_vectors(cache="sbert_vectors.pkl"):
    with open(cache, "rb") as f:
        return pickle.load(f)

# ======================================
# 4) Recherche hybride TF-IDF + SBERT
# ======================================
def hybrid_search(query, vectorizer, tfidf_matrix, sbert_model, sbert_vectors, doc_names, alpha=0.7, top_k=5):
    # ------ TF-IDF step ------
    q_tfidf = vectorizer.transform([query])
    scores_tfidf = cosine_similarity(q_tfidf, tfidf_matrix)[0]

    # prendre les 200 meilleurs
    top_idx = np.argsort(scores_tfidf)[::-1][:200]
    top_docs = [doc_names[i] for i in top_idx]
    top_scores_tfidf = scores_tfidf[top_idx]

    # ------ SBERT reranking ------
    q_sbert = sbert_model.encode([query])[0]
    sbert_scores = []
    for d in top_docs:
        sbert_scores.append(cosine_similarity([q_sbert], [sbert_vectors[d]])[0][0])
    sbert_scores = np.array(sbert_scores)

    # ------ Fusion des scores ------
    final_scores = alpha * sbert_scores + (1 - alpha) * top_scores_tfidf
    rerank_idx = np.argsort(final_scores)[::-1][:top_k]

    return [(top_docs[i], final_scores[i]) for i in rerank_idx]

# ======================================
# 5) Charger requêtes
# ======================================
def load_queries(path="requetes.jsonl"):
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            answer = obj["Answer file"]
            for q in obj["Queries"]:
                queries.append((q, answer))
    return queries

# ======================================
# 6) Évaluation
# ======================================
def evaluate(vectorizer, tfidf_matrix, sbert_model, sbert_vectors, doc_names):
    queries = load_queries()

    hits = 0
    mrr = 0
    for q, answer in tqdm(queries, desc="Évaluation"):
        results = hybrid_search(q, vectorizer, tfidf_matrix, sbert_model, sbert_vectors, doc_names, top_k=2000)
        ranked = [r[0] for r in results]

        if answer in ranked:
            rank = ranked.index(answer) + 1
            mrr += 1 / rank
            if rank == 1:
                hits += 1

    p1 = hits / len(queries)
    mrr /= len(queries)
    return p1, mrr

# ======================================
# MAIN
# ======================================
if __name__ == "__main__":
    DATA_PATH = "wiki_split_extract_2k/"

    print("=== Moteur de recherche Hybride TF-IDF + SBERT ===")

    # 1) Charger documents
    documents = load_documents(DATA_PATH)
    doc_names = list(documents.keys())

    # 2) TF-IDF
    vectorizer, tfidf_matrix = load_or_fit_tfidf(documents)

    # 3) SBERT
    sbert_model = load_sbert_model()
    sbert_vectors = load_sbert_vectors()

    # 4) Test rapide
    query_test = "course à pied"
    print("\nExemple recherche hybride :", query_test)
    for fname, score in hybrid_search(query_test, vectorizer, tfidf_matrix, sbert_model, sbert_vectors, doc_names):
        print(f"{fname:<25} score={score:.4f}")

    # 5) Évaluation totale
    p1, mrr = evaluate(vectorizer, tfidf_matrix, sbert_model, sbert_vectors, doc_names)

    print("\n=== Résultats finaux ===")
    print(f"Precision@1 : {p1*100:.2f}%")
    print(f"MRR : {mrr:.4f}")
    print("🏁 Évaluation terminée.")
