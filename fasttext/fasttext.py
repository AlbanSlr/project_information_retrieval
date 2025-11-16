import os
import numpy as np
from gensim.models import KeyedVectors

from utils.load_documents import load_documents
from utils.text_processing import clean_text


def load_fasttext_model(model_path='cc.fr.300.vec'):
    # on charge le modèle FastText pré-entraîné au format word2vec
    print(f"Chargement du modèle FastText depuis {model_path}...")
    model = KeyedVectors.load_word2vec_format(model_path)
    print(f"Modèle chargé ({len(model.key_to_index)} mots)")
    return model


def save_fasttext_model(model, model_path='models/fasttext.model'):
    # on sauvegarde le modèle dans le dossier models pour réutilisation
    os.makedirs('models', exist_ok=True)
    model.save(model_path)
    print(f"Modèle sauvegardé dans {model_path}")


def text_to_vector(text, model, lemmatize=False, normalize=False):
    # on nettoie le texte et on filtre les mots connus du modèle
    words = clean_text(text, lemmatize, normalize)
    valid_words = [w for w in words if w in model]
    
    # on retourne un vecteur nul si aucun mot n'est reconnu
    if not valid_words:
        return np.zeros(model.vector_size)
    
    # on calcule la moyenne des vecteurs de mots pour représenter le texte
    vectors = np.array([model[w] for w in valid_words])
    return np.mean(vectors, axis=0)


def build_document_vectors(data_path='wiki_split_extract_2k/', model=None, model_path='cc.fr.300.vec', 
                          lemmatize=False, normalize=False):
    # on charge le modèle FastText si non fourni
    if model is None:
        model = load_fasttext_model(model_path)
    
    # on charge tous les documents du corpus
    documents = load_documents(data_path)
    
    print("Calcul des vecteurs FastText pour les documents...")
    doc_names = []
    doc_vectors = []
    
    # on vectorise chaque document en moyennant les vecteurs de ses mots
    for doc_path, text in documents.items():
        doc_name = os.path.basename(doc_path)
        vector = text_to_vector(text, model, lemmatize, normalize)
        doc_names.append(doc_name)
        doc_vectors.append(vector)
    
    # on empile les vecteurs en une matrice pour calculs vectorisés
    doc_vectors_matrix = np.vstack(doc_vectors)
    print(f"{len(doc_names)} vecteurs de documents créés")
    
    return model, doc_names, doc_vectors_matrix


def infer_query_vector(query, model, lemmatize=False, normalize=False):
    return text_to_vector(query, model, lemmatize, normalize)
