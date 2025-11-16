# project_information_retrieval
### Alban Sellier & Rémi Geraud

Moteur de recherche optimisé pour Wikipédia français utilisant TF-IDF avec normalisation logarithmique et nettoyage de texte.

Projet de moteur de recherche combinant des approches sémantiques (SBERT, Doc2Vec, FastText) et lexicales (TF-IDF) pour la recherche d'information sur un corpus Wikipedia français.

## Installation

### Prérequis

- Python 3.12
- Conda (Anaconda ou Miniconda)
- Fichiers de données :
  - Dossier `wiki_split_extract_2k/` contenant les documents Wikipedia
  - Fichier `requetes.jsonl` contenant les requêtes d'évaluation

### Étapes d'installation

1. **Placer les fichiers de données à la racine du projet**
   ```
   project_information_retrieval/
   ├── wiki_split_extract_2k/
   │   ├── wiki_XXXXXX.txt
   │   ├── wiki_XXXXXX.txt
   │   └── ...
   └── requetes.jsonl
   ```

2. **Créer l'environnement Conda avec Python 3.12**
   ```powershell
   conda create -n nom_env python=3.12
   ```

3. **Activer l'environnement**
   ```powershell
   conda activate nom_env
   ```

4. **Installer les dépendances**
   ```powershell
   pip install -r requirements.txt
   ```

5. **Télécharger le modèle FastText français**
   
   Téléchargez le modèle pré-entraîné [ici](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.vec.gz).
   

   Puis décompressez le fichier et placez `cc.fr.300.vec` à la racine du projet.

## Structure du projet

```
project_information_retrieval/
├── doc2vec/                      # Module Doc2Vec
│   └── doc2vec.py                # Entraînement et inférence Doc2Vec
├── engines/                      # Moteurs de recherche
│   ├── base_engine.py            # Classe de base abstraite
│   └── TFIDF_engine.py           # Moteur TF-IDF orienté objet
├── evaluation/                   # Scripts d'évaluation
│   ├── doc2vec_eval.py           # Évaluation Doc2Vec
│   ├── fasttext_eval.py          # Évaluation FastText
│   ├── hybrid_eval.py            # Évaluation moteur hybride
│   ├── sbert_eval.py             # Évaluation SBERT
│   └── TF_IDF_eval.py            # Évaluation TF-IDF
├── fasttext/                     # Module FastText
│   └── fasttext.py               # Vectorisation avec FastText
├── models/                       # Modèles entraînés sauvegardés
│   └── doc2vec.model             # (généré après entraînement)
├── sbert/                        # Module SBERT
│   └── sbert.py                  # Embeddings avec SBERT
├── TFIDF/                        # Module TF-IDF
│   └── TF_IDF.py                 # Calcul TF-IDF
├── utils/                        # Utilitaires
│   ├── compute_cosine_similarity.py
│   ├── evaluation_metrics.py     # Métriques Precision@k et Recall@k
│   ├── load_documents.py         # Chargement corpus
│   ├── load_queries.py           # Chargement requêtes
│   └── text_processing.py        # Nettoyage et lemmatisation
├── wiki_split_extract_2k/        # Corpus de documents (2000 fichiers)
├── hybrid_search.py              # MOTEUR DE RECHERCHE PRINCIPAL
├── requetes.jsonl                # Requêtes d'évaluation
├── requirements.txt              # Dépendances Python
└── README.md                     # Ce fichier
```

## Utilisation

### Moteur de recherche hybride (Interface console)

Le moteur hybride combine SBERT (recherche sémantique) et TF-IDF (recherche lexicale) avec un poids 50/50.

**Lancer le moteur :**
```powershell
python hybrid_search.py
```

**Commandes disponibles :**
- Tapez une requête pour rechercher dans le corpus
- `show <numero>` : affiche le contenu complet d'un document des résultats
- `quit` ou `exit` : quitte le programme

**Exemple d'utilisation :**
```
Recherche: Irlande

Recherche: 'Irlande'
----------------------------------------------------------------------

Meilleur résultat:
======================================================================
Document: wiki_039825.txt
Score: 0.8598
======================================================================
Armée républicaine irlandaise provisoire
Pour les articles homonymes , voir Pira , Armée républicaine irlandaise et Óglaigh na hÉireann .
L' Armée républicaine irlandaise provisoire ( irlandais : Óglaigh na hÉireann , anglais : Provisional Irish Republican Army , PIRA ) , devenue l' Armée républicaine irlandaise , est une organisation paramilitaire républicaine irlandaise considérée comme terroriste par l' Irlande , et le Royaume-Uni , , qui , de 1969 à 1997 , militait pour l' indépendance complète de l' Irlande du Nord vis-à-vis de la monarchie du Royaume-Uni , et l' instauration d' un État républicain libre et souverain sur l' ensemble de l' île d' Irlande ( Éire Nua , Nouvelle Irlande ) .
Plus puissante organisation républicaine du conflit nord-irlandais , l' IRA provisoire est soupçonn

[...1474 caractères restants...]

----------------------------------------------------------------------
Autres résultats (tapez 'show <numero>' pour voir):

 2. wiki_095248.txt                 Score: 0.8413
 3. wiki_097731.txt                 Score: 0.6796
 4. wiki_094672.txt                 Score: 0.6548
 5. wiki_045290.txt                 Score: 0.5745
 6. wiki_023659.txt                 Score: 0.5499
 7. wiki_113000.txt                 Score: 0.4777
 8. wiki_041649.txt                 Score: 0.4652
 9. wiki_059562.txt                 Score: 0.4639
10. wiki_116932.txt                 Score: 0.4593
...
```

## Évaluation des modèles

Les scripts d'évaluation comparent différentes configurations de chaque modèle et génèrent des graphiques de performance.

### 1. Évaluation TF-IDF

Compare 4 configurations :
```powershell
python evaluation\TF_IDF_eval.py
```

### 2. Évaluation Doc2Vec

Compare 3 tailles de vecteurs et nombres d'epochs :
```powershell
python evaluation\doc2vec_eval.py
```

### 3. Évaluation FastText

Compare 3 configurations :
```powershell
python evaluation\fasttext_eval.py
```
**Note :** Nécessite le fichier `cc.fr.300.vec` à la racine

### 4. Évaluation SBERT

Compare 2 modèles pré-entraînés (MiniLM L12, MPNet) :
```powershell
python evaluation\sbert_eval.py
```

### 5. Évaluation moteur hybride

Compare 5 configurations de poids SBERT/TF-IDF (0/100, 25/75, 50/50, 75/25, 100/0) :
```powershell
python evaluation\hybrid_eval.py
```