"""
Utils package - Utility modules for text processing and evaluation.
"""
from utils.text_processing import clean_text, FRENCH_STOPWORDS
from utils.evaluation import (
    load_queries,
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_f1_score,
    evaluate_search_engine,
    print_evaluation_results
)

__all__ = [
    'clean_text',
    'FRENCH_STOPWORDS',
    'load_queries',
    'calculate_precision_at_k',
    'calculate_recall_at_k',
    'calculate_f1_score',
    'evaluate_search_engine',
    'print_evaluation_results'
]
