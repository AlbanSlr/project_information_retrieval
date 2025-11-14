"""
Base class for all search engines.
Provides a common interface for different retrieval methods.
"""
from abc import ABC, abstractmethod


class BaseSearchEngine(ABC):
    """
    Abstract base class for search engines.
    All search engines must implement this interface.
    """
    
    def __init__(self, data_path='wiki_split_extract_2k/'):
        """
        Initialize the search engine.
        
        Args:
            data_path (str): Path to directory containing documents
        """
        self.data_path = data_path
        self.documents = {}
        self.is_built = False
    
    @abstractmethod
    def load_documents(self):
        """Load documents from data_path into self.documents."""
        pass
    
    @abstractmethod
    def build_index(self):
        """Build the search index/model."""
        pass
    
    @abstractmethod
    def search(self, query, top_k=10):
        """
        Search for documents matching the query.
        
        Args:
            query (str): Search query
            top_k (int): Number of results to return
        
        Returns:
            list: List of (filename, score) tuples sorted by relevance
        """
        pass
    
    def get_document_content(self, filename):
        """
        Get the content of a document.
        
        Args:
            filename (str): Name of the document
        
        Returns:
            str: Document content or None if not found
        """
        for path, content in self.documents.items():
            if path.endswith(filename) or path == filename:
                return content
        return None
    
    def get_stats(self):
        """
        Get statistics about the search engine.
        
        Returns:
            dict: Statistics dictionary
        """
        return {
            'num_documents': len(self.documents),
            'is_built': self.is_built,
            'engine_type': self.__class__.__name__
        }
