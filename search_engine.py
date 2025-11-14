"""
Search Engine for Information Retrieval using TF-IDF.
Implements index, reverse index, and document ranking using cosine similarity.
"""
import math
import os
from text_processing import clean_text


class SearchEngine:
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
    
    def __init__(self, clean_params=None, use_log_tf=False):
        """
        Initialize the search engine.
        
        Args:
            clean_params (dict): Parameters for text cleaning
            use_log_tf (bool): Use logarithmic term frequency
        """
        self.documents = {}
        self.index = {}  # Forward index: doc -> term -> score
        self.reverse_index = {}  # Inverted index: term -> [(doc, score), ...]
        self.idf = {}
        self.clean_params = clean_params or {
            'remove_punctuation': False,
            'remove_stopwords': False,
            'lemmatize': False
        }
        self.use_log_tf = use_log_tf
    
    def load_documents(self, data_path):
        """
        Load documents from a directory.
        
        Args:
            data_path (str): Path to directory containing text documents
        """
        file_paths = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.txt')]
        
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.documents[file_path] = f.read()
        
        print(f"Loaded {len(self.documents)} documents from {data_path}")
    
    def build_index(self):
        """
        Build the TF-IDF index and reverse index.
        Creates both forward index (document -> terms) and inverted index (term -> documents).
        """
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
        
        print(f"Index built with {len(self.index)} documents and {len(self.reverse_index)} unique terms")
    
    def _compute_term_frequency(self):
        """
        Compute term frequency for all documents.
        
        Returns:
            dict: Document -> term -> frequency
        """
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
                # Logarithmic term frequency: log(1 + count)
                term_freq = {word: math.log(1 + count) for word, count in word_counts.items()}
            else:
                # Normalized term frequency: count / total_words
                total_words = len(words)
                term_freq = {word: count / total_words for word, count in word_counts.items()}
            
            term_frequencies[doc_path] = term_freq
        
        return term_frequencies
    
    def _compute_idf(self):
        """
        Compute inverse document frequency for all terms.
        
        Returns:
            dict: Term -> IDF score
        """
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
        """
        Compute cosine similarity between query and document vectors.
        
        Args:
            query_vector (dict): Query term -> weight
            doc_vector (dict): Document term -> TF-IDF score
        
        Returns:
            float: Cosine similarity score
        """
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
        """
        Get all documents containing a specific term using the reverse index.
        
        Args:
            term (str): Term to search for
        
        Returns:
            list: List of (filename, TF-IDF score) tuples
        """
        term_clean = clean_text(term, **self.clean_params)
        if not term_clean:
            return []
        
        return self.reverse_index.get(term_clean[0], [])
    
    def get_index_stats(self):
        """
        Get statistics about the index.
        
        Returns:
            dict: Index statistics
        """
        return {
            'num_documents': len(self.documents),
            'num_unique_terms': len(self.reverse_index),
            'avg_terms_per_doc': sum(len(terms) for terms in self.index.values()) / len(self.index) if self.index else 0,
            'clean_params': self.clean_params,
            'use_log_tf': self.use_log_tf
        }


def main():
    """
    Interactive search interface.
    """
    # Initialize search engine with text cleaning
    print("Initializing search engine...")
    engine = SearchEngine(
        clean_params={
            'remove_punctuation': True,
            'remove_stopwords': True,
            'lemmatize': True
        },
        use_log_tf=True
    )
    
    # Load documents and build index
    engine.load_documents('wiki_split_extract_2k/')
    engine.build_index()
    
    # Display index statistics
    stats = engine.get_index_stats()
    print("\nIndex Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*70)
    print("Interactive Search - Enter your queries (type 'quit' or 'exit' to stop)")
    print("="*70)
    
    # Interactive search loop
    while True:
        print("\n")
        query = input("Search query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if not query:
            print("Please enter a query.")
            continue
        
        results = engine.search(query, top_k=5)
        
        if not results:
            print("No results found.")
            continue
        
        print(f"\nFound {len(results)} results:\n")
        
        for i, (filename, score) in enumerate(results, 1):
            # Get document path
            doc_path = None
            for path in engine.documents.keys():
                if os.path.basename(path) == filename:
                    doc_path = path
                    break
            
            # Get document content preview
            if doc_path:
                content = engine.documents[doc_path]
                preview = content[:200].replace('\n', ' ')
                if len(content) > 200:
                    preview += "..."
            else:
                preview = "Content not available"
            
            print(f"{i}. {filename} (score: {score:.4f})")
            print(f"   {preview}")
            print()


if __name__ == "__main__":
    main()
