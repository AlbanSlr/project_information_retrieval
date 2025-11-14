"""
Text processing utilities for the information retrieval system.
Handles text cleaning, tokenization, and normalization.
"""
import json
import spacy
import unicodedata

# Load spaCy French model for lemmatization
nlp = spacy.load('fr_core_news_sm')

# Load French stopwords from JSON file
# Source: https://countwordsfree.com/stopwords/french
with open('stop_words_french.json', 'r', encoding='utf-8') as f:
    FRENCH_STOPWORDS = set(json.load(f))


def clean_text(text, remove_punctuation=False, remove_stopwords=False, lemmatize=False):
    """
    Clean and normalize text for information retrieval.
    
    Args:
        text (str): Input text to clean
        remove_punctuation (bool): Apply NFD normalization (removes accents)
        remove_stopwords (bool): Filter out French stopwords
        lemmatize (bool): Apply lemmatization using spaCy
    
    Returns:
        list: List of processed tokens
    """
    if lemmatize:
        doc = nlp(text.lower())
        words = [token.lemma_ for token in doc if not token.is_space]
    else:
        words = text.lower().split()
    
    if remove_stopwords:
        words = [word for word in words if word not in FRENCH_STOPWORDS]
    
    if remove_punctuation:
        words = [unicodedata.normalize('NFD', word) for word in words]

    return words
