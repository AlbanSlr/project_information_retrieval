import math

from utils.text_processing import clean_text

def compute_tf(documents, use_log_tf, lemmatize, normalize):
    # on calcule la fréquence des termes pour chaque document
    term_frequencies = {}
    
    for doc_path, text in documents.items():
        # on nettoie le texte selon les paramètres spécifiés
        words = clean_text(text, lemmatize, normalize)
        
        if not words:
            term_frequencies[doc_path] = {}
            continue
        
        # on compte les occurrences de chaque mot
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # on applique la normalisation logarithmique si demandée, sinon normalisation standard
        if use_log_tf:
            # log(1+tf) réduit l'impact des mots très fréquents
            term_freq = {word: math.log(1 + count) for word, count in word_counts.items()}
        else:
            # tf normalisé par la longueur du document
            total_words = len(words)
            term_freq = {word: count / total_words for word, count in word_counts.items()}
        
        term_frequencies[doc_path] = term_freq
    
    return term_frequencies

def compute_idf(documents, lemmatize, normalize):
    # on calcule l'inverse de la fréquence documentaire pour chaque terme
    doc_count = len(documents)
    word_doc_counts = {}
    
    for doc_path, text in documents.items():
        # on utilise un set pour compter chaque mot une seule fois par document
        words = set(clean_text(text, lemmatize, normalize))
        
        for word in words:
            word_doc_counts[word] = word_doc_counts.get(word, 0) + 1
    
    # on calcule l'IDF avec formule log((N/df) + 1) pour éviter division par zéro
    return {
        word: math.log((doc_count / count) + 1) 
        for word, count in word_doc_counts.items()
    }