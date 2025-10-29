import os
import math

DATA_PATH = 'wiki_split_extract_2k/'

file_paths = [DATA_PATH + file for file in os.listdir(DATA_PATH) if file.endswith('.txt')]

def compute_term_frequency(file_paths):
    term_frequencies = {}
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().lower()
        words = text.split()
        total_words = len(words)
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        term_freq = {word: count / total_words for word, count in word_counts.items()}
        term_frequencies[file_path] = term_freq
    return term_frequencies

def compute_idf(file_paths):
    doc_count = 0
    word_doc_counts = {}
    for file_path in file_paths:
        doc_count += 1
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().lower()
        words = set(text.split())
        for word in words:
            word_doc_counts[word] = word_doc_counts.get(word, 0) + 1
    return {word: math.log(doc_count / (1 + count)) for word, count in word_doc_counts.items()} 


def compute_tf_idf(file_paths):
    term_frequencies = compute_term_frequency(file_paths)
    idf = compute_idf(file_paths)
    tf_idf = {}
    for file_path, tf in term_frequencies.items():
        tf_idf[file_path] = {word: freq * idf.get(word, 0) for word, freq in tf.items()}
    return tf_idf
