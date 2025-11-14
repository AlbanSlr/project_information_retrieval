import json
import math
import os
from compute_TFIDF import compute_tf_idf


def load_queries(filepath):
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


def load_tfidf_index(data_path):
    file_paths = [data_path + file for file in os.listdir(data_path) if file.endswith('.txt')]
    tf_idf = compute_tf_idf(file_paths)
    return tf_idf


def compute_cosine_similarity(query_vector, doc_vector):
    dot_product = sum(query_vector.get(word, 0) * doc_vector.get(word, 0) 
                     for word in set(query_vector.keys()) | set(doc_vector.keys()))
    
    query_magnitude = math.sqrt(sum(val ** 2 for val in query_vector.values()))
    doc_magnitude = math.sqrt(sum(val ** 2 for val in doc_vector.values()))
    
    if query_magnitude == 0 or doc_magnitude == 0:
        return 0
    
    return dot_product / (query_magnitude * doc_magnitude)


def search(query, tf_idf, top_k=10):
    query_words = query.lower().split()
    word_counts = {}
    for word in query_words:
        word_counts[word] = word_counts.get(word, 0) + 1
    
    query_vector = {word: count / len(query_words) for word, count in word_counts.items()}
    
    scores = []
    for doc_path, doc_vector in tf_idf.items():
        score = compute_cosine_similarity(query_vector, doc_vector)
        if score > 0:
            filename = doc_path.split('/')[-1]
            scores.append((filename, score))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def calculate_precision_at_k(expected_file, results, k):
    if not results:
        return 0
    retrieved_files = [filename for filename, score in results[:k]]
    if expected_file in retrieved_files:
        return 1 / (retrieved_files.index(expected_file) + 1)
    return 0


def calculate_recall_at_k(expected_file, results, k):
    if not results:
        return 0
    retrieved_files = [filename for filename, score in results[:k]]
    return 1 if expected_file in retrieved_files else 0


def evaluate(queries, tf_idf, k=10):
    total_precision = 0
    total_recall = 0
    num_queries = len(queries)
    
    for item in queries:
        query = item['query']
        expected_file = item['answer']
        results = search(query, tf_idf, top_k=k)
        
        precision = calculate_precision_at_k(expected_file, results, k)
        recall = calculate_recall_at_k(expected_file, results, k)
        
        total_precision += precision
        total_recall += recall
    
    avg_precision = total_precision / num_queries
    avg_recall = total_recall / num_queries
    avg_f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
    
    return {
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': avg_f1,
        'num_queries': num_queries
    }


if __name__ == "__main__":
    print("Loading queries...")
    queries = load_queries('requetes.jsonl')
    print(f"Loaded {len(queries)} queries")
    
    print("\nBuilding TF-IDF index...")
    tf_idf = load_tfidf_index('wiki_split_extract_2k/')
    print(f"Indexed {len(tf_idf)} documents")
    
    print("\nEvaluating...")
    metrics = evaluate(queries, tf_idf, k=10)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Total queries: {metrics['num_queries']}")
    print(f"Precision@10:  {metrics['precision']:.4f}")
    print(f"Recall@10:     {metrics['recall']:.4f}")
    print(f"F1@10:         {metrics['f1']:.4f}")
    print("="*60)
    
    print("\nExample search results (first 5 queries):")
    print("-"*60)
    for i, item in enumerate(queries[:5]):
        query = item['query']
        expected_file = item['answer']
        results = search(query, tf_idf, top_k=5)
        
        print(f"\nQuery {i+1}: '{query}'")
        print(f"Expected: {expected_file}")
        print("Top 5 results:")
        for rank, (filename, score) in enumerate(results, 1):
            marker = " OK" if filename == expected_file else ""
            print(f"  {rank}. {filename} (score: {score:.4f}){marker}")