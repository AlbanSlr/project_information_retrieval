import os

def load_documents(data_path):
    # on récupère tous les fichiers texte du répertoire
    file_paths = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.txt')]
    
    documents = {}
    
    # on charge le contenu de chaque fichier avec son chemin comme clé
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            documents[file_path] = f.read()
            
    return documents