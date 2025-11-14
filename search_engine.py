"""
Main search engine - Uses the best performing configuration.
Based on benchmarks, TF-IDF (log + cleaning) gives optimal performance.
"""
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engines.tfidf_engine import TFIDFSearchEngine


def main():
    """
    Interactive search interface using the best performing engine.
    """
    # Initialize TF-IDF engine with optimal configuration (log + cleaning)
    print("Initializing TF-IDF search engine (log + cleaning)...")
    print("Optimal configuration: log(1+tf) normalization + stopwords removal + lemmatization")
    
    engine = TFIDFSearchEngine(
        clean_params={'remove_punctuation': True, 'remove_stopwords': True, 'lemmatize': True},
        use_log_tf=True
    )
    
    # Load documents and build index
    engine.load_documents()
    engine.build_index()
    
    # Display statistics
    stats = engine.get_stats()
    print("\nSearch Engine Statistics:")
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
            # Get document content preview
            content = engine.get_document_content(filename)
            if content:
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
