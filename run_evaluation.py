"""
Evaluation script for the information retrieval system.
This script tests the search engine from test.py against expected results from evaluation.py
"""

import math
from evaluation import load_queries_from_jsonl


# Import the search engine components from test.py
def compute_cosine_similarity(query_vector, doc_vector):
    """
    Compute cosine similarity between query and document vectors.
    """
    dot_product = sum(query_vector.get(word, 0) * doc_vector.get(word, 0) 
                     for word in set(query_vector.keys()) | set(doc_vector.keys()))
    
    query_magnitude = math.sqrt(sum(val ** 2 for val in query_vector.values()))
    doc_magnitude = math.sqrt(sum(val ** 2 for val in doc_vector.values()))
    
    if query_magnitude == 0 or doc_magnitude == 0:
        return 0
    
    return dot_product / (query_magnitude * doc_magnitude)


def search(query, tf_idf, idf, top_k=10):
    """
    Search for documents matching the query.
    
    Args:
        query (str): Search query
        tf_idf (dict): TF-IDF scores for all documents
        idf (dict): IDF scores for all terms
        top_k (int): Number of top results to return
        
    Returns:
        list: List of (filename, score) tuples sorted by relevance
    """
    query_words = query.lower().split()
    
    # Compute query TF-IDF vector
    query_tf = {}
    for word in query_words:
        query_tf[word] = query_tf.get(word, 0) + 1
    
    query_vector = {word: (count / len(query_words)) * idf.get(word, 0) 
                   for word, count in query_tf.items()}
    
    # Compute similarity scores
    scores = []
    for doc_path, doc_vector in tf_idf.items():
        score = compute_cosine_similarity(query_vector, doc_vector)
        if score > 0:
            # Extract just the filename
            filename = doc_path.split('/')[-1]
            scores.append((filename, score))
    
    # Sort by score (descending)
    scores.sort(key=lambda x: x[1], reverse=True)
    
    return scores[:top_k]


def calculate_metrics(expected_results, search_results, k=10):
    """
    Calculate precision, recall, and F1 score.
    
    Args:
        expected_results (list): List of query-answer pairs
        search_results (dict): Dictionary mapping queries to their search results
        k (int): Number of top results to consider
        
    Returns:
        dict: Dictionary containing metrics
    """
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    successful_queries = 0
    
    for item in expected_results:
        query = item['query']
        expected_file = item['answer']
        
        if query not in search_results:
            continue
            
        results = search_results[query][:k]
        retrieved_files = [filename for filename, score in results]
        
        # Check if expected file is in results
        if expected_file in retrieved_files:
            position = retrieved_files.index(expected_file) + 1
            precision = 1 / position  # Precision at the position where it was found
            recall = 1  # We found it
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        else:
            precision = 0
            recall = 0
            f1 = 0
        
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        successful_queries += 1
    
    if successful_queries == 0:
        return {
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'total_queries': 0
        }
    
    return {
        'precision': total_precision / successful_queries,
        'recall': total_recall / successful_queries,
        'f1': total_f1 / successful_queries,
        'total_queries': successful_queries
    }


def main():
    """
    Main evaluation function.
    """
    print("Loading expected results from evaluation.py...")
    expected_results = load_queries_from_jsonl('requetes.jsonl')
    print(f"Loaded {len(expected_results)} query-answer pairs\n")
    
    print("Loading TF-IDF data from compute_TFIDF.py...")
    # Import the precomputed data from compute_TFIDF.py
    from compute_TFIDF import tf_idf, idf
    print(f"Loaded TF-IDF data for {len(tf_idf)} documents\n")
    
    print("Running search queries...")
    search_results = {}
    for i, item in enumerate(expected_results):
        query = item['query']
        results = search(query, tf_idf, idf, top_k=10)
        search_results[query] = results
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(expected_results)} queries...")
    
    print(f"\nAll {len(expected_results)} queries processed.\n")
    
    # Calculate metrics
    print("Calculating evaluation metrics...")
    metrics = calculate_metrics(expected_results, search_results, k=10)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Total queries evaluated: {metrics['total_queries']}")
    print(f"Average Precision@10:    {metrics['precision']:.4f}")
    print(f"Average Recall@10:       {metrics['recall']:.4f}")
    print(f"Average F1 Score@10:     {metrics['f1']:.4f}")
    print("="*60)
    
    # Show some example results
    print("\nExample search results (first 5 queries):")
    print("-"*60)
    for i, item in enumerate(expected_results[:5]):
        query = item['query']
        expected_file = item['answer']
        results = search_results[query][:5]
        
        print(f"\nQuery {i+1}: '{query}'")
        print(f"Expected: {expected_file}")
        print("Top 5 results:")
        for rank, (filename, score) in enumerate(results, 1):
            marker = " ✓" if filename == expected_file else ""
            print(f"  {rank}. {filename} (score: {score:.4f}){marker}")


if __name__ == "__main__":
    main()
