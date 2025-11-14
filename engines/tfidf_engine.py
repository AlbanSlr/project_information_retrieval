"""
TF-IDF Search Engine for Information Retrieval.
Implements index, reverse index, and document ranking using cosine similarity.
"""
import math
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.text_processing import clean_text
from engines.base_engine import BaseSearchEngine


class TFIDFSearchEngine(BaseSearchEngine):
    """
    TF-IDF based search engine with inverted index.
    
    Attributes:
        documents (dict): Document paths mapped to their content
        index (dict): Forward index - document -> term -> TF-IDF score
        reverse_index (dict): Inverted index - term -> list of (document, TF-IDF score)
        idf (dict): Inverse document frequency for each term
        clean_params (dict): Text cleaning parameters
        use_log_tf (bool): Use log(1 + tf) normalization
    """
    
    def __init__(self, data_path='wiki_split_extract_2k/', clean_params=None, use_log_tf=False):
        """
        Initialize the search engine.
        
        Args:
            data_path (str): Path to directory containing documents
            clean_params (dict): Parameters for text cleaning
            use_log_tf (bool): Use logarithmic term frequency
        """
        super().__init__(data_path)
        self.index = {}  # Forward index: doc -> term -> score
        self.reverse_index = {}  # Inverted index: term -> [(doc, score), ...]
        self.idf = {}
        self.clean_params = clean_params or {
            'remove_punctuation': False,
            'remove_stopwords': False,
            'lemmatize': False
        }
        self.use_log_tf = use_log_tf
    
    def load_documents(self):
        """Load documents from data_path."""
        file_paths = [os.path.join(self.data_path, f) for f in os.listdir(self.data_path) if f.endswith('.txt')]
        
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.documents[file_path] = f.read()
        
        print(f"Loaded {len(self.documents)} documents from {self.data_path}")
    
    def build_index(self):
        """Build the TF-IDF index and reverse index."""
        print("Computing term frequencies...")
        term_frequencies = self._compute_term_frequency()
        
        print("Computing inverse document frequencies...")
        self.idf = self._compute_idf()
        
        print("Building TF-IDF index...")
        # Build forward index
        for doc_path, tf in term_frequencies.items():
            self.index[doc_path] = {
                word: freq * self.idf.get(word, 0) 
                for word, freq in tf.items()
            }
        
        # Build reverse (inverted) index
        print("Building reverse index...")
        for doc_path, term_scores in self.index.items():
            filename = os.path.basename(doc_path)
            for term, score in term_scores.items():
                if term not in self.reverse_index:
                    self.reverse_index[term] = []
                self.reverse_index[term].append((filename, score))
        
        # Sort reverse index by score (descending)
        for term in self.reverse_index:
            self.reverse_index[term].sort(key=lambda x: x[1], reverse=True)
        
        self.is_built = True
        print(f"Index built with {len(self.index)} documents and {len(self.reverse_index)} unique terms")
    
    def _compute_term_frequency(self):
        """Compute term frequency for all documents."""
        term_frequencies = {}
        
        for doc_path, text in self.documents.items():
            words = clean_text(text, **self.clean_params)
            
            if not words:
                term_frequencies[doc_path] = {}
                continue
            
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            if self.use_log_tf:
                term_freq = {word: math.log(1 + count) for word, count in word_counts.items()}
            else:
                total_words = len(words)
                term_freq = {word: count / total_words for word, count in word_counts.items()}
            
            term_frequencies[doc_path] = term_freq
        
        return term_frequencies
    
    def _compute_idf(self):
        """Compute inverse document frequency for all terms."""
        doc_count = len(self.documents)
        word_doc_counts = {}
        
        for doc_path, text in self.documents.items():
            words = set(clean_text(text, **self.clean_params))
            
            for word in words:
                word_doc_counts[word] = word_doc_counts.get(word, 0) + 1
        
        return {
            word: math.log(doc_count / (1 + count)) 
            for word, count in word_doc_counts.items()
        }
    
    def _compute_cosine_similarity(self, query_vector, doc_vector):
        """Compute cosine similarity between query and document vectors."""
        dot_product = sum(
            query_vector.get(word, 0) * doc_vector.get(word, 0) 
            for word in set(query_vector.keys()) | set(doc_vector.keys())
        )
        
        query_magnitude = math.sqrt(sum(val ** 2 for val in query_vector.values()))
        doc_magnitude = math.sqrt(sum(val ** 2 for val in doc_vector.values()))
        
        if query_magnitude == 0 or doc_magnitude == 0:
            return 0
        
        return dot_product / (query_magnitude * doc_magnitude)
    
    def search(self, query, top_k=10):
        """
        Search for documents matching the query.
        
        Args:
            query (str): Search query
            top_k (int): Number of top results to return
        
        Returns:
            list: List of (filename, score) tuples sorted by relevance
        """
        query_words = clean_text(query, **self.clean_params)
        
        if not query_words:
            return []
        
        # Build query vector
        word_counts = {}
        for word in query_words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        query_vector = {word: count / len(query_words) for word, count in word_counts.items()}
        
        # Compute similarity scores for all documents
        scores = []
        for doc_path, doc_vector in self.index.items():
            score = self._compute_cosine_similarity(query_vector, doc_vector)
            if score > 0:
                filename = os.path.basename(doc_path)
                scores.append((filename, score))
        
        # Sort by score (descending) and return top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def get_term_documents(self, term):
        """Get all documents containing a specific term using the reverse index."""
        term_clean = clean_text(term, **self.clean_params)
        if not term_clean:
            return []
        
        return self.reverse_index.get(term_clean[0], [])
    
    def get_stats(self):
        """Get statistics about the search engine."""
        stats = super().get_stats()
        stats.update({
            'num_unique_terms': len(self.reverse_index),
            'avg_terms_per_doc': sum(len(terms) for terms in self.index.values()) / len(self.index) if self.index else 0,
            'clean_params': self.clean_params,
            'use_log_tf': self.use_log_tf
        })
        return stats
