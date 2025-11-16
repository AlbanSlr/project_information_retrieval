import os
import numpy as np
from sentence_transformers import SentenceTransformer

from utils.load_documents import load_documents


def load_sbert_model(model_name='paraphrase-multilingual-mpnet-base-v2'):
    # on charge un modèle SBERT pré-entraîné depuis HuggingFace
    print(f"Chargement du modèle SBERT {model_name}...")
    model = SentenceTransformer(model_name)
    print("Modèle chargé")
    return model


def save_sbert_model(model, model_path='models/sbert'):
    # on sauvegarde le modèle localement pour éviter de le retélécharger
    os.makedirs('models', exist_ok=True)
    model.save(model_path)
    print(f"Modèle sauvegardé dans {model_path}")


def build_document_vectors(data_path='wiki_split_extract_2k/', model=None, model_name='paraphrase-multilingual-mpnet-base-v2'):
    # on charge le modèle SBERT si nécessaire
    if model is None:
        model = load_sbert_model(model_name)
    
    # on charge tous les documents
    documents = load_documents(data_path)
    
    print("Calcul des embeddings SBERT pour les documents...")
    # on prépare les listes de noms et contenus pour l'encodage batch
    doc_names = []
    contents = []
    
    # on extrait les noms et contenus de tous les documents
    for doc_path, text in documents.items():
        doc_name = os.path.basename(doc_path)
        doc_names.append(doc_name)
        contents.append(text)
    
    # on encode tous les documents en batch pour plus d'efficacité
    doc_vectors = model.encode(contents, batch_size=32, show_progress_bar=True)
    doc_vectors_matrix = np.array(doc_vectors)
    
    print(f"{len(doc_names)} vecteurs de documents créés")
    
    return model, doc_names, doc_vectors_matrix


def infer_query_vector(model, query):
    # on encode la requête pour obtenir son vecteur sémantique
    return model.encode([query])[0]
