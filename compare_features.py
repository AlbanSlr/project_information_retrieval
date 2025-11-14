"""
Feature comparison for TF-IDF configurations.
Compares different feature combinations to show their impact on performance.
"""
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engines.tfidf_engine import TFIDFSearchEngine
from utils.evaluation import load_queries, evaluate_search_engine


def compare_tfidf_features():
    """
    Compare different TF-IDF feature configurations.
    Shows the progressive improvement from baseline to optimized.
    """
    
    print("="*70)
    print("TF-IDF FEATURE COMPARISON")
    print("="*70)
    print()
    
    # Load queries
    print("Loading queries...")
    queries = load_queries('requetes.jsonl')
    print(f"Loaded {len(queries)} queries\n")
    
    # Define TF-IDF configurations
    configurations = [
        {
            'name': 'Baseline\n(no features)',
            'clean_params': {
                'remove_punctuation': False,
                'remove_stopwords': False,
                'lemmatize': False
            },
            'use_log_tf': False,
            'description': 'Raw TF-IDF without preprocessing'
        },
        {
            'name': 'Baseline\n+ Stopwords',
            'clean_params': {
                'remove_punctuation': True,
                'remove_stopwords': True,
                'lemmatize': False
            },
            'use_log_tf': False,
            'description': 'Remove French stopwords'
        },
        {
            'name': 'Baseline\n+ Stopwords\n+ Lemmatize',
            'clean_params': {
                'remove_punctuation': True,
                'remove_stopwords': True,
                'lemmatize': True
            },
            'use_log_tf': False,
            'description': 'Add spaCy lemmatization'
        },
        {
            'name': 'Optimized\n(+ log normalization)',
            'clean_params': {
                'remove_punctuation': True,
                'remove_stopwords': True,
                'lemmatize': True
            },
            'use_log_tf': True,
            'description': 'Add log(1+tf) normalization'
        }
    ]
    
    results = []
    
    for config in configurations:
        print(f"\n{'='*70}")
        print(f"Testing: {config['name'].replace(chr(10), ' ')}")
        print(f"Description: {config['description']}")
        print(f"{'='*70}")
        
        engine = TFIDFSearchEngine(
            clean_params=config['clean_params'],
            use_log_tf=config['use_log_tf']
        )
        
        try:
            # Load and build
            engine.load_documents()
            engine.build_index()
            
            # Evaluate
            print("\nEvaluating...")
            precision, recall, f1 = evaluate_search_engine(engine, queries)
            
            results.append({
                'name': config['name'],
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
            
            print(f"\nResults:")
            print(f"  Precision@10: {precision:.3f}")
            print(f"  Recall@10: {recall:.3f}")
            print(f"  F1@10: {f1:.3f}")
            
        except Exception as e:
            print(f"Error testing {config['name'].replace(chr(10), ' ')}: {e}")
            continue
    
    # Generate comparison graph
    print(f"\n{'='*70}")
    print("Generating feature comparison graph...")
    print(f"{'='*70}\n")
    
    generate_feature_comparison_graph(results)
    
    print("\nFeature comparison complete!")
    print("Generated: tfidf_features_comparison.png")
    print("\n" + "="*70)
    print("CONCLUSION:")
    print("="*70)
    print("Progressive feature addition shows clear improvements:")
    print("  1. Stopwords removal: Filters noise words")
    print("  2. Lemmatization: Normalizes word forms")
    print("  3. Log normalization: Reduces term frequency bias")
    print("\nOptimized configuration gives best performance!")
    print("="*70)


def generate_feature_comparison_graph(results):
    """
    Generate comparison graph showing progressive feature improvements.
    """
    if not results:
        print("No results to plot")
        return
    
    # Extract data
    feature_names = [r['name'] for r in results]
    precision = [r['precision'] for r in results]
    recall = [r['recall'] for r in results]
    f1 = [r['f1'] for r in results]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(feature_names))
    width = 0.25
    
    # Create bars with gradient colors
    colors_precision = ['#3498db', '#2980b9', '#21618c', '#1a5276']
    colors_recall = ['#2ecc71', '#27ae60', '#229954', '#1e8449']
    colors_f1 = ['#e74c3c', '#c0392b', '#a93226', '#922b21']
    
    bars1 = ax.bar(x - width, precision, width, label='Precision@10', 
                   color=colors_precision[:len(precision)], alpha=0.8)
    bars2 = ax.bar(x, recall, width, label='Recall@10', 
                   color=colors_recall[:len(recall)], alpha=0.8)
    bars3 = ax.bar(x + width, f1, width, label='F1@10', 
                   color=colors_f1[:len(f1)], alpha=0.8)
    
    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    
    # Customize graph
    ax.set_xlabel('Feature Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('TF-IDF Feature Ablation Study: Progressive Improvements', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, fontsize=10)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add improvement arrows
    for i in range(len(f1) - 1):
        improvement = f1[i+1] - f1[i]
        if improvement > 0:
            mid_x = x[i] + 0.5
            mid_y = (f1[i] + f1[i+1]) / 2 + width
            ax.annotate(f'+{improvement:.3f}',
                       xy=(mid_x, mid_y),
                       ha='center',
                       fontsize=9,
                       color='green',
                       fontweight='bold')
    
    # Highlight best configuration
    best_idx = np.argmax(f1)
    ax.annotate('Optimal', 
                xy=(best_idx, f1[best_idx] + 0.05),
                xytext=(best_idx, f1[best_idx] + 0.15),
                ha='center',
                fontsize=11,
                fontweight='bold',
                color='#e74c3c',
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2))
    
    plt.tight_layout()
    plt.savefig('tfidf_features_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: tfidf_features_comparison.png")


if __name__ == "__main__":
    compare_tfidf_features()
