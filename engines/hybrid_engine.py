"""
Hybrid Search Engine combining TF-IDF and SBERT.
Uses TF-IDF (log + cleaning) for initial retrieval and SBERT for reranking.
"""
import os
import pickle
import sys
import json
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engines.base_engine import BaseSearchEngine
from engines.tfidf_engine import TFIDFSearchEngine


class HybridSearchEngine(BaseSearchEngine):
    """
    Hybrid search engine combining TF-IDF and SBERT.
    First stage: TF-IDF (log + cleaning) retrieves top 200 candidates
    Second stage: SBERT reranks top candidates
    """
    
    def __init__(self, data_path='wiki_split_extract_2k/', 
                 model_name="paraphrase-multilingual-MiniLM-L12-v2",
                 alpha=0.7,
                 stage1_candidates=200,
                 cache_dir='data/'):
        """
        Initialize hybrid search engine.
        
        Args:
            data_path (str): Path to documents directory
            model_name (str): SBERT model name
            alpha (float): Weight for SBERT score (1-alpha for TF-IDF)
            stage1_candidates (int): Number of candidates from TF-IDF stage
            cache_dir (str): Directory for caching
        """
        super().__init__(data_path)
        self.model_name = model_name
        self.alpha = alpha
        self.stage1_candidates = stage1_candidates
        self.cache_dir = cache_dir
        
        # Initialize TF-IDF engine (optimized: log + cleaning)
        self.tfidf_engine = TFIDFSearchEngine(
            data_path=data_path,
            use_log_tf=True,
            clean_params={
                'remove_punctuation': True,
                'remove_stopwords': True,
                'lemmatize': True
            }
        )
        
        self.sbert_model = None
        self.sbert_vectors = {}
        os.makedirs(cache_dir, exist_ok=True)
    
    def load_documents(self):
        """Load documents from data_path."""
        # Use TF-IDF engine's document loading
        self.tfidf_engine.load_documents()
        self.documents = self.tfidf_engine.documents
        print(f"Loaded {len(self.documents)} documents from TF-IDF engine")
    
    def build_index(self):
        """Build TF-IDF and SBERT indices."""
        sbert_cache = os.path.join(self.cache_dir, 'hybrid_sbert.pkl')
        
        # Stage 1: Build TF-IDF index (optimized with log + cleaning)
        print("Stage 1: Building TF-IDF index (log + cleaning)...")
        self.tfidf_engine.build_index()
        
        # Stage 2: Build SBERT index
        print(f"\nStage 2: Loading SBERT model: {self.model_name}...")
        self.sbert_model = SentenceTransformer(self.model_name)
        
        if os.path.exists(sbert_cache):
            print("Loading SBERT vectors from cache...")
            with open(sbert_cache, "rb") as f:
                self.sbert_vectors = pickle.load(f)
        else:
            print("Computing SBERT embeddings...")
            doc_names = list(self.documents.keys())
            contents = [self.documents[name] for name in doc_names]
            vectors = self.sbert_model.encode(contents, batch_size=32, show_progress_bar=True)
            self.sbert_vectors = {name: vec for name, vec in zip(doc_names, vectors)}
            
            with open(sbert_cache, "wb") as f:
                pickle.dump(self.sbert_vectors, f)
            print("SBERT vectors cached")
        
        self.is_built = True
        print("\nHybrid index built successfully")
    
    def search(self, query, top_k=10):
        """
        Search using hybrid TF-IDF + SBERT approach.
        
        Args:
            query (str): Search query
            top_k (int): Number of results to return
        
        Returns:
            list: List of (filename, score) tuples
        """
        if not self.is_built:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        # Stage 1: TF-IDF retrieval (optimized with log + cleaning)
        tfidf_results = self.tfidf_engine.search(query, top_k=self.stage1_candidates)
        candidates = [doc for doc, _ in tfidf_results]
        
        # Normalize TF-IDF scores to [0, 1]
        tfidf_scores_dict = {doc: score for doc, score in tfidf_results}
        max_tfidf = max(tfidf_scores_dict.values()) if tfidf_scores_dict else 1.0
        normalized_tfidf = {doc: score / max_tfidf for doc, score in tfidf_scores_dict.items()}
        
        # Stage 2: SBERT reranking
        q_sbert = self.sbert_model.encode([query])[0]
        sbert_scores = {}
        for doc in candidates:
            sim = cosine_similarity([q_sbert], [self.sbert_vectors[doc]])[0][0]
            sbert_scores[doc] = sim
        
        # Normalize SBERT scores to [0, 1]
        max_sbert = max(sbert_scores.values()) if sbert_scores else 1.0
        min_sbert = min(sbert_scores.values()) if sbert_scores else 0.0
        normalized_sbert = {
            doc: (score - min_sbert) / (max_sbert - min_sbert) if max_sbert != min_sbert else 0.0
            for doc, score in sbert_scores.items()
        }
        
        # Combine scores: alpha * SBERT + (1-alpha) * TF-IDF
        final_scores = {
            doc: self.alpha * normalized_sbert[doc] + (1 - self.alpha) * normalized_tfidf[doc]
            for doc in candidates
        }
        
        # Sort and return top K
        sorted_results = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return sorted_results
    
    def get_stats(self):
        """Get statistics about the search engine."""
        stats = super().get_stats()
        tfidf_stats = self.tfidf_engine.get_stats() if self.tfidf_engine.is_built else {}
        stats.update({
            'engine_type': 'Hybrid (TF-IDF + SBERT)',
            'tfidf_config': 'log(1+tf) + cleaning (stopwords + lemmatize)',
            'stage1_candidates': self.stage1_candidates,
            'sbert_model': self.model_name,
            'alpha': self.alpha,
            'num_unique_terms': tfidf_stats.get('num_unique_terms', 0),
            'sbert_embeddings': len(self.sbert_vectors)
        })
        return stats
