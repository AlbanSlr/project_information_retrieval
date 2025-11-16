import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.load_documents import load_documents
from utils.load_queries import load_queries
from utils.evaluation_metrics import calculate_precision_at_k, calculate_recall_at_k
from doc2vec.doc2vec import train_doc2vec, load_doc2vec_model, infer_query_vector


def build_document_vectors(model, documents):
    print("Extraction des vecteurs de documents...")
    doc_names = []
    doc_vectors = []
    
    for doc_path in documents.keys():
        doc_name = os.path.basename(doc_path)
        vector = model.dv[doc_path]
        doc_names.append(doc_name)
        doc_vectors.append(vector)
    
    doc_vectors_matrix = np.vstack(doc_vectors)
    print(f"{len(doc_names)} vecteurs de documents extraits")
    
    return doc_names, doc_vectors_matrix


def search_doc2vec(query, model, doc_names, doc_vectors_matrix, top_k=10):
    query_vec = infer_query_vector(model, query).reshape(1, -1)
    sims = cosine_similarity(query_vec, doc_vectors_matrix)[0]
    
    results = []
    for doc_name, score in zip(doc_names, sims):
        if score > 0:
            results.append((doc_name, score))
    
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def evaluate_doc2vec(model, doc_names, doc_vectors_matrix, queries, k=10):
    total_precision = 0
    total_recall = 0
    
    for query_data in queries:
        query_text = query_data['query']
        expected_file = query_data['answer']
        
        results = search_doc2vec(query_text, model, doc_names, doc_vectors_matrix, top_k=k)
        
        precision = calculate_precision_at_k(expected_file, results, k)
        recall = calculate_recall_at_k(expected_file, results, k)
        
        total_precision += precision
        total_recall += recall
    
    avg_precision = total_precision / len(queries)
    avg_recall = total_recall / len(queries)
    
    
    return avg_precision, avg_recall


def compare_doc2vec_configurations(queries_file='requetes.jsonl', data_path='wiki_split_extract_2k/'):
    print("="*70)
    print("DOC2VEC CONFIGURATION COMPARISON")
    print("="*70)
    print()
    
    print("Loading queries...")
    queries = load_queries(queries_file)
    print(f"Loaded {len(queries)} queries\n")
    
    print("Loading documents...")
    documents = load_documents(data_path)
    print(f"Loaded {len(documents)} documents\n")
    
    configurations = [
        {
            'name': 'Small\n(vec=100, ep=100)',
            'vector_size': 100,
            'epochs': 100,
            'description': 'Faster training, lower dimensionality'
        },
        {
            'name': 'Medium\n(vec=200, ep=400)',
            'vector_size': 200,
            'epochs': 400,
            'description': 'Balanced training time and quality'
        },
        {
            'name': 'Large\n(vec=300, ep=1500)',
            'vector_size': 300,
            'epochs': 1500,
            'description': 'Better representation, very long training'
        }
    ]
    
    results = []
    
    for config in configurations:
        print(f"\n{'='*70}")
        print(f"Testing: {config['name'].replace(chr(10), ' ')}")
        print(f"Description: {config['description']}")
        print(f"{'='*70}")
        
        try:
            model = train_doc2vec(
                data_path=data_path,
                vector_size=config['vector_size'],
                epochs=config['epochs']
            )
            
            doc_names, doc_vectors_matrix = build_document_vectors(model, documents)
            
            print("\nEvaluating...")
            precision, recall = evaluate_doc2vec(
                model,
                doc_names,
                doc_vectors_matrix,
                queries
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
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*70}")
    print("Generating configuration comparison graph...")
    print(f"{'='*70}\n")
    
    generate_configuration_comparison_graph(results)


def generate_configuration_comparison_graph(results):
    if not results:
        print("No results to plot")
        return
    
    config_names = [r['name'] for r in results]
    precision = [r['precision'] for r in results]
    recall = [r['recall'] for r in results]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(config_names))
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
    
    ax.set_xlabel('Model Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Doc2Vec Configuration Comparison: Vector Size & Training Epochs',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(config_names, fontsize=10)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_ylim(0, 0.5)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    
    plt.tight_layout()
    plt.savefig('doc2vec_configurations_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: doc2vec_configurations_comparison.png")


if __name__ == "__main__":
    compare_doc2vec_configurations()
