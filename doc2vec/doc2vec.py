import os
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from utils.load_documents import load_documents
from utils.text_processing import clean_text


def build_tagged_documents(documents):
    # on transforme les documents en TaggedDocuments pour Doc2Vec
    tagged_docs = []
    for fname, text in documents.items():
        words = clean_text(text)
        tagged_docs.append(TaggedDocument(words=words, tags=[fname]))
    return tagged_docs


def train_doc2vec(data_path='wiki_split_extract_2k/', vector_size=200, epochs=40):
    # on charge les documents et on les prépare pour l'entraînement
    documents = load_documents(data_path)
    tagged_docs = build_tagged_documents(documents)
    
    print("Entraînement du modèle Doc2Vec…")
    # on crée un modèle Doc2Vec avec la méthode DM (Distributed Memory)
    model = Doc2Vec(
        vector_size=vector_size,
        window=10,
        min_count=2,
        workers=4,
        dm=1
    )
    model.build_vocab(tagged_docs)
    # on entraîne le modèle sur le corpus
    model.train(tagged_docs, total_examples=model.corpus_count, epochs=epochs)
    
    # on sauvegarde le modèle dans le dossier models
    model_path = os.path.join('models', 'doc2vec.model')
    os.makedirs('models', exist_ok=True)
    model.save(model_path)
    print(f"Modèle sauvegardé dans {model_path}")
    
    return model


def load_doc2vec_model(model_path='models/doc2vec.model'):
    # on charge un modèle Doc2Vec déjà entraîné depuis le disque
    return Doc2Vec.load(model_path)


def infer_query_vector(model, query):
    # on nettoie la requête et on infère son vecteur avec le modèle
    words = clean_text(query)
    # l'inférence calcule un vecteur pour un texte non vu pendant l'entraînement
    return model.infer_vector(words)