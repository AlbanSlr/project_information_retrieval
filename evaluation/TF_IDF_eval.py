import os
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.load_documents import load_documents
from utils.load_queries import load_queries
from utils.text_processing import clean_text
from utils.evaluation_metrics import calculate_precision_at_k, calculate_recall_at_k
from utils.compute_cosine_similarity import compute_cosine_similarity
from TFIDF.TF_IDF import compute_tf, compute_idf


def build_tfidf_index(documents, use_log_tf=False, lemmatize=False, normalize=False):
    print("Computing term frequencies...")
    term_frequencies = compute_tf(documents, use_log_tf, lemmatize, normalize)
    
    print("Computing inverse document frequencies...")
    idf = compute_idf(documents, lemmatize, normalize)
    
    print("Building TF-IDF index...")
    tfidf_index = {}
    for doc_path, tf in term_frequencies.items():
        tfidf_index[doc_path] = {
            word: freq * idf.get(word, 0)
            for word, freq in tf.items()
        }
    
    print(f"Index built with {len(tfidf_index)} documents")
    return tfidf_index


def search_tfidf(query, tfidf_index, lemmatize=False, normalize=False, top_k=10):
    query_words = clean_text(query, lemmatize, normalize)
    
    if not query_words:
        return []
    
    word_counts = {}
    for word in query_words:
        word_counts[word] = word_counts.get(word, 0) + 1
    
    query_vector = {word: count / len(query_words) for word, count in word_counts.items()}
    
    scores = []
    for doc_path, doc_vector in tfidf_index.items():
        score = compute_cosine_similarity(query_vector, doc_vector)
        if score > 0:
            filename = os.path.basename(doc_path)
            scores.append((filename, score))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def evaluate_tfidf(tfidf_index, queries, lemmatize=False, normalize=False, k=10):
    total_precision = 0
    total_recall = 0
    
    for query_data in queries:
        query_text = query_data['query']
        expected_file = query_data['answer']
        
        results = search_tfidf(query_text, tfidf_index, lemmatize, normalize, top_k=k)
        
        precision = calculate_precision_at_k(expected_file, results, k)
        recall = calculate_recall_at_k(expected_file, results, k)
        
        total_precision += precision
        total_recall += recall
    
    avg_precision = total_precision / len(queries)
    avg_recall = total_recall / len(queries)
    
    return avg_precision, avg_recall


def compare_tfidf_features(queries_file='requetes.jsonl'):
    print("="*70)
    print("TF-IDF FEATURE COMPARISON")
    print("="*70)
    print()
    
    print("Loading queries...")
    queries = load_queries(queries_file)
    print(f"Loaded {len(queries)} queries\n")
    
    print("Loading documents...")
    documents = load_documents('wiki_split_extract_2k/')
    print(f"Loaded {len(documents)} documents\n")
    
    configurations = [
        {
            'name': 'Baseline\n(no features)',
            'use_log_tf': False,
            'lemmatize': False,
            'normalize': False,
            'description': 'TF-IDF without preprocessing'
        },
        {
            'name': 'Normalized TF-IDF \n(no preprocessing)',
            'use_log_tf': True,
            'lemmatize': False,
            'normalize': False,
            'description': 'Normalize TF-IDF without preprocessing'
        },
        {
            'name': 'Baseline\n+ Text Normalization\n+ Lemmatize',
            'use_log_tf': False,
            'lemmatize': True,
            'normalize': True,
            'description': 'TF-IDF with text normalization and lemmatization'
        },
        {
            'name': 'Normalized TF-IDF \n+ Text Normalization\n+ Lemmatize',
            'use_log_tf': True,
            'lemmatize': True,
            'normalize': True,
            'description': 'Normalized TF-IDF with text normalization and lemmatization'
        }
    ]
    
    results = []
    
    for config in configurations:
        print(f"\n{'='*70}")
        print(f"Testing: {config['name'].replace(chr(10), ' ')}")
        print(f"Description: {config['description']}")
        print(f"{'='*70}")
        
        try:
            tfidf_index = build_tfidf_index(
                documents,
                use_log_tf=config['use_log_tf'],
                lemmatize=config['lemmatize'],
                normalize=config['normalize']
            )
            
            print("\nEvaluating...")
            precision, recall = evaluate_tfidf(
                tfidf_index,
                queries,
                lemmatize=config['lemmatize'],
                normalize=config['normalize']
            )
            
            results.append({
                'name': config['name'],
                'precision': precision,
                'recall': recall
            })
            
            print(f"\nResults:")
            print(f"  Precision@10: {precision:.3f}")
            print(f"  Recall@10: {recall:.3f}")
            
        except Exception as e:
            print(f"Error testing {config['name'].replace(chr(10), ' ')}: {e}")
            continue
    
    print(f"\n{'='*70}")
    print("Generating feature comparison graph...")
    print(f"{'='*70}\n")
    
    generate_feature_comparison_graph(results)



def generate_feature_comparison_graph(results):
    if not results:
        print("No results to plot")
        return
    
    feature_names = [r['name'] for r in results]
    precision = [r['precision'] for r in results]
    recall = [r['recall'] for r in results]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(feature_names))
    width = 0.25
    
    colors_precision ='#3498db'
    colors_recall = '#2ecc71'
    
    bars1 = ax.bar(x - width, precision, width, label='Precision@10',
                   color=colors_precision, alpha=0.8)
    bars2 = ax.bar(x, recall, width, label='Recall@10',
                   color=colors_recall, alpha=0.8)
    
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    add_labels(bars1)
    add_labels(bars2)
    
    ax.set_xlabel('Parameters Comparison', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('TF-IDF Parameters Comparison',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, fontsize=10)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    
    plt.tight_layout()
    plt.savefig('tfidf_features_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: tfidf_features_comparison.png")


if __name__ == "__main__":
    compare_tfidf_features()
