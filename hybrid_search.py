import os
import sys
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.load_documents import load_documents
from sbert.sbert import load_sbert_model, build_document_vectors, infer_query_vector
from TFIDF.TF_IDF import compute_tf, compute_idf
from utils.text_processing import clean_text
from utils.compute_cosine_similarity import compute_cosine_similarity as compute_cos_sim


class HybridSearchEngine:
    def __init__(self, data_path='wiki_split_extract_2k/', sbert_model_name='paraphrase-multilingual-mpnet-base-v2'):
        print("Initialisation du moteur hybride SBERT + TF-IDF...")
        self.data_path = data_path
        # on charge tous les documents du corpus
        self.documents = load_documents(data_path)
        
        print("\nChargement du modèle SBERT...")
        # on charge le modèle SBERT pré-entraîné et on vectorise tous les documents
        self.sbert_model, self.doc_names, self.sbert_vectors = build_document_vectors(
            data_path=data_path,
            model_name=sbert_model_name
        )
        
        print("\nConstruction de l'index TF-IDF...")
        # on construit l'index TF-IDF pour la recherche lexicale
        self.tfidf_index = self._build_tfidf_index()
        
        print("\nMoteur hybride prêt!")
    
    def _build_tfidf_index(self):
        # on calcule d'abord les fréquences de termes avec normalisation logarithmique
        term_frequencies = compute_tf(self.documents, use_log_tf=True, lemmatize=False, normalize=False)
        # on calcule ensuite l'IDF pour pondérer l'importance des termes
        idf = compute_idf(self.documents, lemmatize=False, normalize=False)
        
        # on construit l'index TF-IDF en multipliant TF par IDF pour chaque terme
        tfidf_index = {}
        for doc_path, tf in term_frequencies.items():
            tfidf_index[doc_path] = {
                word: freq * idf.get(word, 0)
                for word, freq in tf.items()
            }
        return tfidf_index
    
    def _search_sbert(self, query, top_k=10):
        # on transforme la requête en vecteur avec le modèle SBERT
        query_vec = infer_query_vector(self.sbert_model, query).reshape(1, -1)
        # on calcule la similarité cosine entre la requête et tous les documents
        sims = cosine_similarity(query_vec, self.sbert_vectors)[0]
        
        # on stocke les scores de similarité pour chaque document
        results = {}
        for doc_name, score in zip(self.doc_names, sims):
            results[doc_name] = score
        
        return results
    
    def _search_tfidf(self, query, top_k=10):
        # on nettoie la requête de la même manière que les documents indexés
        query_words = clean_text(query, lemmatize=False, normalize=True)
        
        # on retourne un résultat vide si la requête ne contient aucun mot valide
        if not query_words:
            return {}
        
        # on compte les occurrences de chaque mot dans la requête
        word_counts = {}
        for word in query_words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # on crée le vecteur de requête normalisé par la longueur
        query_vector = {word: count / len(query_words) for word, count in word_counts.items()}
        
        # on calcule la similarité cosine avec chaque document de l'index
        results = {}
        for doc_path, doc_vector in self.tfidf_index.items():
            score = compute_cos_sim(query_vector, doc_vector)
            # on ne garde que les documents avec un score positif
            if score > 0:
                doc_name = os.path.basename(doc_path)
                results[doc_name] = score
        
        return results
    
    def search_hybrid(self, query, top_k=10, sbert_weight=0.5, tfidf_weight=0.5):
        # on effectue la recherche avec les deux méthodes en parallèle
        sbert_scores = self._search_sbert(query, top_k)
        tfidf_scores = self._search_tfidf(query, top_k)
        
        # on récupère tous les documents trouvés par au moins une des deux méthodes
        all_docs = set(sbert_scores.keys()) | set(tfidf_scores.keys())
        
        # on normalise les scores pour pouvoir les combiner équitablement
        max_sbert = max(sbert_scores.values()) if sbert_scores else 1.0
        max_tfidf = max(tfidf_scores.values()) if tfidf_scores else 1.0
        
        # on combine les scores normalisés avec les poids spécifiés (50/50 par défaut)
        hybrid_scores = {}
        for doc in all_docs:
            sbert_norm = sbert_scores.get(doc, 0) / max_sbert
            tfidf_norm = tfidf_scores.get(doc, 0) / max_tfidf
            hybrid_scores[doc] = sbert_weight * sbert_norm + tfidf_weight * tfidf_norm
        
        sorted_results = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]
    
    def get_document_content(self, doc_name):
        for doc_path, content in self.documents.items():
            if os.path.basename(doc_path) == doc_name:
                return content
        return None


def console_search_interface():
    print("="*70)
    print("MOTEUR DE RECHERCHE HYBRIDE - SBERT + TF-IDF")
    print("="*70)
    print()
    
    # on initialise le moteur hybride avec chargement des modèles
    engine = HybridSearchEngine()
    
    print("\n" + "="*70)
    print("Interface de recherche prête!")
    print("Tapez 'quit' ou 'exit' pour quitter")
    print("Tapez 'show <numero>' pour afficher le contenu d'un document")
    print("="*70)
    
    # on garde en mémoire les derniers résultats pour la commande 'show'
    last_results = []
    
    while True:
        print("\n" + "-"*70)
        query = input("Recherche: ").strip()
        
        # on gère la commande de sortie
        if query.lower() in ['quit', 'exit', 'q']:
            print("\nAu revoir!")
            break
        
        # on gère la commande d'affichage détaillé d'un document
        if query.lower().startswith('show '):
            try:
                num = int(query.split()[1]) - 1
                if 0 <= num < len(last_results):
                    doc_name, score = last_results[num]
                    content = engine.get_document_content(doc_name)
                    print(f"\n{'='*70}")
                    print(f"Document: {doc_name}")
                    print(f"Score: {score:.4f}")
                    print(f"{'='*70}")
                    print(content[:500] + "..." if len(content) > 500 else content)
                else:
                    print(f"Numéro invalide. Entrez un nombre entre 1 et {len(last_results)}")
            except (ValueError, IndexError):
                print("Format invalide. Utilisez: show <numero>")
            continue
        
        # on ignore les requêtes vides
        if not query:
            continue
        
        print(f"\nRecherche: '{query}'")
        print("-"*70)
        
        results = engine.search_hybrid(query, top_k=10)
        last_results = results
        
        if not results:
            print("Aucun résultat trouvé.")
        else:
            # on récupère le meilleur résultat et on affiche son contenu
            doc_name, score = results[0]
            content = engine.get_document_content(doc_name)
            
            print(f"\nMeilleur résultat:")
            print(f"{'='*70}")
            print(f"Document: {doc_name}")
            print(f"Score: {score:.4f}")
            print(f"{'='*70}")
            if content:
                preview_length = 800
                print(content[:preview_length])
                if len(content) > preview_length:
                    print(f"\n[...{len(content) - preview_length} caractères restants...]")
            else:
                print("Contenu non disponible")
            
            if len(results) > 1:
                print(f"\n{'-'*70}")
                print(f"Autres résultats (tapez 'show <numero>' pour voir):\n")
                for i, (doc_name, score) in enumerate(results[1:], 2):
                    print(f"{i:2d}. {doc_name:<30s}  Score: {score:.4f}")


if __name__ == "__main__":
    console_search_interface()
