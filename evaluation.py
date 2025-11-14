"""
Evaluation module for the information retrieval system.
Provides metrics for assessing search engine performance.
"""
import json


def load_queries(filepath):
    """
    Load queries from JSONL file.
    
    Args:
        filepath (str): Path to JSONL file containing queries
    
    Returns:
        list: List of dicts with 'query' and 'answer' keys
    """
    query_answer_pairs = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                answer_file = data['Answer file']
                for query in data['Queries']:
                    query_answer_pairs.append({
                        'query': query,
                        'answer': answer_file
                    })
    return query_answer_pairs


def calculate_precision_at_k(expected_file, results, k):
    """
    Calculate Precision@K using reciprocal rank.
    
    Args:
        expected_file (str): Expected result filename
        results (list): List of (filename, score) tuples
        k (int): Number of top results to consider
    
    Returns:
        float: Precision score (1/rank if found in top-k, else 0)
    """
    if not results:
        return 0
    retrieved_files = [filename for filename, score in results[:k]]
    if expected_file in retrieved_files:
        return 1 / (retrieved_files.index(expected_file) + 1)
    return 0


def calculate_recall_at_k(expected_file, results, k):
    """
    Calculate Recall@K (binary: 1 if found, 0 otherwise).
    
    Args:
        expected_file (str): Expected result filename
        results (list): List of (filename, score) tuples
        k (int): Number of top results to consider
    
    Returns:
        float: 1.0 if expected file is in top-k, 0.0 otherwise
    """
    if not results:
        return 0
    retrieved_files = [filename for filename, score in results[:k]]
    return 1 if expected_file in retrieved_files else 0


def calculate_f1_score(precision, recall):
    """
    Calculate F1 score from precision and recall.
    
    Args:
        precision (float): Precision value
        recall (float): Recall value
    
    Returns:
        float: F1 score
    """
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)


def evaluate_search_engine(search_engine, queries, k=10):
    """
    Evaluate search engine performance on a set of queries.
    
    Args:
        search_engine: SearchEngine instance with search() method
        queries (list): List of query-answer pairs
        k (int): Number of top results to consider
    
    Returns:
        dict: Evaluation metrics (precision, recall, f1, num_queries)
    """
    total_precision = 0
    total_recall = 0
    num_queries = len(queries)
    
    for item in queries:
        query = item['query']
        expected_file = item['answer']
        results = search_engine.search(query, top_k=k)
        
        precision = calculate_precision_at_k(expected_file, results, k)
        recall = calculate_recall_at_k(expected_file, results, k)
        
        total_precision += precision
        total_recall += recall
    
    avg_precision = total_precision / num_queries
    avg_recall = total_recall / num_queries
    avg_f1 = calculate_f1_score(avg_precision, avg_recall)
    
    return {
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': avg_f1,
        'num_queries': num_queries
    }


def print_evaluation_results(metrics, config_name=""):
    """
    Print evaluation results in a formatted way.
    
    Args:
        metrics (dict): Evaluation metrics
        config_name (str): Optional configuration name
    """
    if config_name:
        print(f"\n{'='*70}")
        print(f"Configuration: {config_name}")
        print(f"{'='*70}")
    
    print(f"Queries evaluated: {metrics['num_queries']}")
    print(f"Precision@10:      {metrics['precision']:.4f}")
    print(f"Recall@10:         {metrics['recall']:.4f}")
    print(f"F1@10:             {metrics['f1']:.4f}")


def main():
    """
    Example evaluation script.
    """
    from search_engine import SearchEngine
    
    # Load queries
    print("Loading queries...")
    queries = load_queries('requetes.jsonl')
    print(f"Loaded {len(queries)} queries\n")
    
    # Initialize and build search engine
    print("Building search engine...")
    engine = SearchEngine(
        clean_params={
            'remove_punctuation': True,
            'remove_stopwords': True,
            'lemmatize': True
        },
        use_log_tf=True
    )
    engine.load_documents('wiki_split_extract_2k/')
    engine.build_index()
    
    # Evaluate
    print("\nEvaluating search engine...")
    metrics = evaluate_search_engine(engine, queries, k=10)
    print_evaluation_results(metrics, "TF-IDF with text cleaning and log TF")


if __name__ == "__main__":
    main()
