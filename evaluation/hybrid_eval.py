import os
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.load_queries import load_queries
from utils.evaluation_metrics import calculate_precision_at_k, calculate_recall_at_k
from hybrid_search import HybridSearchEngine


def evaluate_hybrid(engine, queries, sbert_weight=0.5, tfidf_weight=0.5, k=10):
    total_precision = 0
    total_recall = 0
    
    for query_data in queries:
        query_text = query_data['query']
        expected_file = query_data['answer']
        
        results = engine.search_hybrid(query_text, top_k=k, sbert_weight=sbert_weight, tfidf_weight=tfidf_weight)
        
        precision = calculate_precision_at_k(expected_file, results, k)
        recall = calculate_recall_at_k(expected_file, results, k)
        
        total_precision += precision
        total_recall += recall
    
    avg_precision = total_precision / len(queries)
    avg_recall = total_recall / len(queries)
    
    return avg_precision, avg_recall


def compare_hybrid_weights(queries_file='requetes.jsonl'):
    print("="*70)
    print("HYBRID SEARCH ENGINE - WEIGHT COMPARISON")
    print("="*70)
    print()
    
    print("Loading queries...")
    queries = load_queries(queries_file)
    print(f"Loaded {len(queries)} queries\n")
    
    print("Initializing hybrid search engine...")
    engine = HybridSearchEngine()
    
    configurations = [
        {
            'name': 'TF-IDF Only\n(100% TF-IDF)',
            'sbert_weight': 0.0,
            'tfidf_weight': 1.0,
            'description': 'Pure lexical matching'
        },
        {
            'name': 'TF-IDF Heavy\n(25% SBERT)',
            'sbert_weight': 0.25,
            'tfidf_weight': 0.75,
            'description': 'Lexical focus with semantic boost'
        },
        {
            'name': 'Balanced\n(50% / 50%)',
            'sbert_weight': 0.5,
            'tfidf_weight': 0.5,
            'description': 'Equal semantic and lexical'
        },
        {
            'name': 'SBERT Heavy\n(75% SBERT)',
            'sbert_weight': 0.75,
            'tfidf_weight': 0.25,
            'description': 'Semantic focus with lexical boost'
        },
        {
            'name': 'SBERT Only\n(100% SBERT)',
            'sbert_weight': 1.0,
            'tfidf_weight': 0.0,
            'description': 'Pure semantic matching'
        }
    ]
    
    results = []
    
    for config in configurations:
        print(f"\n{'='*70}")
        print(f"Testing: {config['name'].replace(chr(10), ' ')}")
        print(f"Description: {config['description']}")
        print(f"Weights: SBERT={config['sbert_weight']:.2f}, TF-IDF={config['tfidf_weight']:.2f}")
        print(f"{'='*70}")
        
        try:
            print("\nEvaluating...")
            precision, recall = evaluate_hybrid(
                engine,
                queries,
                sbert_weight=config['sbert_weight'],
                tfidf_weight=config['tfidf_weight']
            )
            
            results.append({
                'name': config['name'],
                'precision': precision,
                'recall': recall,
                'sbert_weight': config['sbert_weight']
            })
            
            print(f"\nResults:")
            print(f"  Precision@10: {precision:.3f}")
            print(f"  Recall@10: {recall:.3f}")
            
        except Exception as e:
            print(f"Error testing {config['name'].replace(chr(10), ' ')}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*70}")
    print("Generating weight comparison graph...")
    print(f"{'='*70}\n")
    
    generate_weight_comparison_graph(results)


def generate_weight_comparison_graph(results):
    if not results:
        print("No results to plot")
        return
    
    weight_names = [r['name'] for r in results]
    precision = [r['precision'] for r in results]
    recall = [r['recall'] for r in results]
    sbert_weights = [r['sbert_weight'] for r in results]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(weight_names))
    width = 0.35
    
    colors_precision = '#3498db'
    colors_recall = '#2ecc71'
    
    bars1 = ax.bar(x - width/2, precision, width, label='Precision@10',
                   color=colors_precision, alpha=0.8)
    bars2 = ax.bar(x + width/2, recall, width, label='Recall@10',
                   color=colors_recall, alpha=0.8)
    
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    add_labels(bars1)
    add_labels(bars2)
    
    ax.set_xlabel('Weight Configuration (SBERT / TF-IDF)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Hybrid : Weight Optimization (SBERT + TF-IDF)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(weight_names, fontsize=10)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('hybrid_weights_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: hybrid_weights_comparison.png")


if __name__ == "__main__":
    compare_hybrid_weights()
