"""
Feature comparison tool for the information retrieval system.
Compares different configurations and visualizes performance metrics.
"""
import json
import math
import os
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from text_processing import clean_text
from evaluation import load_queries, calculate_precision_at_k, calculate_recall_at_k

DATA_PATH = 'wiki_split_extract_2k/'


def compute_term_frequency(file_paths, clean_params, use_log_tf=False):
    term_frequencies = {}
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        words = clean_text(text, **clean_params)
        
        if not words:
            term_frequencies[file_path] = {}
            continue
            
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        if use_log_tf:
            term_freq = {word: math.log(1 + count) for word, count in word_counts.items()}
        else:
            total_words = len(words)
            term_freq = {word: count / total_words for word, count in word_counts.items()}
        
        term_frequencies[file_path] = term_freq
    return term_frequencies


def compute_idf(file_paths, clean_params):
    doc_count = 0
    word_doc_counts = {}
    for file_path in file_paths:
        doc_count += 1
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        words = set(clean_text(text, **clean_params))
        
        for word in words:
            word_doc_counts[word] = word_doc_counts.get(word, 0) + 1
    return {word: math.log(doc_count / (1 + count)) for word, count in word_doc_counts.items()}


def compute_tf_idf(file_paths, clean_params, use_log_tf=False):
    term_frequencies = compute_term_frequency(file_paths, clean_params, use_log_tf)
    idf = compute_idf(file_paths, clean_params)
    tf_idf = {}
    for file_path, tf in term_frequencies.items():
        tf_idf[file_path] = {word: freq * idf.get(word, 0) for word, freq in tf.items()}
    
    return tf_idf


def compute_cosine_similarity(query_vector, doc_vector):
    dot_product = sum(query_vector.get(word, 0) * doc_vector.get(word, 0) 
                     for word in set(query_vector.keys()) | set(doc_vector.keys()))
    
    query_magnitude = math.sqrt(sum(val ** 2 for val in query_vector.values()))
    doc_magnitude = math.sqrt(sum(val ** 2 for val in doc_vector.values()))
    
    if query_magnitude == 0 or doc_magnitude == 0:
        return 0
    
    return dot_product / (query_magnitude * doc_magnitude)


def search(query, tf_idf, clean_params, top_k=10):
    query_words = clean_text(query, **clean_params)
    
    if not query_words:
        return []
    
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


def evaluate(queries, tf_idf, clean_params, k=10):
    total_precision = 0
    total_recall = 0
    num_queries = len(queries)
    
    for item in queries:
        query = item['query']
        expected_file = item['answer']
        results = search(query, tf_idf, clean_params, top_k=k)
        
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


def search_sklearn(query, vectorizer, tfidf_matrix, filenames, clean_params, top_k=10):
    query_words = clean_text(query, **clean_params)
    if not query_words:
        return []
    
    query_text = ' '.join(query_words)
    query_vec = vectorizer.transform([query_text])
    
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        if similarities[idx] > 0:
            results.append((filenames[idx], similarities[idx]))
    
    return results


def evaluate_sklearn(queries, vectorizer, tfidf_matrix, filenames, clean_params, k=10):
    total_precision = 0
    total_recall = 0
    num_queries = len(queries)
    
    for item in queries:
        query = item['query']
        expected_file = item['answer']
        results = search_sklearn(query, vectorizer, tfidf_matrix, filenames, clean_params, top_k=k)
        
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


def run_comparison():
    print("Loading queries...")
    queries = load_queries('requetes.jsonl')
    print(f"Loaded {len(queries)} queries\n")
    
    file_paths = [DATA_PATH + file for file in os.listdir(DATA_PATH) if file.endswith('.txt')]
    print(f"Found {len(file_paths)} documents\n")
    
    configurations = [
        {
            'name': 'Manual TF-IDF',
            'clean_params': {'remove_punctuation': False, 'remove_stopwords': False, 'lemmatize': False},
            'use_log_tf': False,
            'use_sklearn': False
        },
        {
            'name': 'Manual TF-IDF (log)',
            'clean_params': {'remove_punctuation': False, 'remove_stopwords': False, 'lemmatize': False},
            'use_log_tf': True,
            'use_sklearn': False
        },
        {
            'name': 'Scikit-learn TF-IDF',
            'clean_params': {'remove_punctuation': False, 'remove_stopwords': False, 'lemmatize': False},
            'use_log_tf': False,
            'use_sklearn': True
        },
        {
            'name': 'Manual + Text cleaning',
            'clean_params': {'remove_punctuation': True, 'remove_stopwords': True, 'lemmatize': True},
            'use_log_tf': False,
            'use_sklearn': False
        },
        {
            'name': 'Manual + Text cleaning (log)',
            'clean_params': {'remove_punctuation': True, 'remove_stopwords': True, 'lemmatize': True},
            'use_log_tf': True,
            'use_sklearn': False
        },
        {
            'name': 'Scikit + Text cleaning',
            'clean_params': {'remove_punctuation': True, 'remove_stopwords': True, 'lemmatize': True},
            'use_log_tf': False,
            'use_sklearn': True
        }
    ]
    
    results = []
    
    for config in configurations:
        print(f"Testing: {config['name']}")
        print("Building TF-IDF index...")
        
        if config['use_sklearn']:
            documents = []
            filenames = []
            for file_path in file_paths:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                words = clean_text(text, **config['clean_params'])
                documents.append(' '.join(words))
                filenames.append(file_path.split('/')[-1])
            
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(documents)
            
            print("Evaluating...")
            metrics = evaluate_sklearn(queries, vectorizer, tfidf_matrix, filenames, config['clean_params'], k=10)
        else:
            tf_idf = compute_tf_idf(file_paths, config['clean_params'], config['use_log_tf'])
            
            print("Evaluating...")
            metrics = evaluate(queries, tf_idf, config['clean_params'], k=10)
        
        results.append({
            'name': config['name'],
            'metrics': metrics
        })
        
        print(f"Precision@10: {metrics['precision']:.4f}")
        print(f"Recall@10:    {metrics['recall']:.4f}")
        print(f"F1@10:        {metrics['f1']:.4f}\n")
    
    print("\n" + "="*70)
    print("COMPARISON TABLE")
    print("="*70)
    print(f"{'Configuration':<35} {'Precision@10':<15} {'Recall@10':<15} {'F1@10':<10}")
    print("-"*70)
    for result in results:
        print(f"{result['name']:<35} {result['metrics']['precision']:<15.4f} {result['metrics']['recall']:<15.4f} {result['metrics']['f1']:<10.4f}")
    print("="*70)
    
    names = [r['name'] for r in results]
    precisions = [r['metrics']['precision'] for r in results]
    recalls = [r['metrics']['recall'] for r in results]
    f1s = [r['metrics']['f1'] for r in results]
    
    # Precision graph
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, precisions, color='skyblue')
    ax.set_title('Precision@10', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim([0, 1])
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('precision_comparison.png', dpi=300, bbox_inches='tight')
    print("\nPrecision graph saved as 'precision_comparison.png'")
    plt.close()
    
    # Recall graph
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, recalls, color='lightgreen')
    ax.set_title('Recall@10', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim([0, 1])
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('recall_comparison.png', dpi=300, bbox_inches='tight')
    print("Recall graph saved as 'recall_comparison.png'")
    plt.close()
    
    # F1 graph
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, f1s, color='salmon')
    ax.set_title('F1@10', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim([0, 1])
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('f1_comparison.png', dpi=300, bbox_inches='tight')
    print("F1 graph saved as 'f1_comparison.png'")
    plt.close()


if __name__ == "__main__":
    run_comparison()
