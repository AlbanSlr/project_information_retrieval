# Information Retrieval Search Engine
### Alban Sellier & Rémi Geraud

Moteur de recherche Wikipédia français utilisant TF-IDF et des techniques avancées de traitement de texte.

## Vue d'ensemble du projet

Ce projet implémente un système complet de recherche d'information pour parcourir 2 000 documents de Wikipédia en français. Le moteur utilise TF-IDF (Term Frequency-Inverse Document Frequency) avec diverses techniques d'optimisation incluant :

- **Structures d'index** : Index direct (document vers termes) et index inversé (termes vers documents)
- **Prétraitement de texte** : Suppression des stopwords, lemmatisation, normalisation des accents
- **Normalisation TF** : TF classique et TF logarithmique (log(1+tf))
- **Classement** : Similarité cosinus pour l'appariement document-requête
- **Évaluation** : Métriques Precision@K, Recall@K et F1@K

## Structure du projet

```
project_information_retrieval/
├── search_engine.py          # Moteur de recherche principal avec indexation et recherche
├── evaluation.py             # Métriques d'évaluation et traitement des requêtes
├── text_processing.py        # Utilitaires de nettoyage et normalisation de texte
├── compare_features.py       # Comparaison de configurations et visualisation
├── requetes.jsonl           # Requêtes de test (100 requêtes)
├── wiki_split_extract_2k/   # 2000 documents Wikipédia
├── stop_words_french.json   # Liste de stopwords français
├── Makefile                 # Automatisation de la construction et configuration
└── README.md                # Ce fichier
```

## Démarrage rapide

### 1. Configuration de l'environnement

Le projet nécessite Python 3.12 et plusieurs dépendances. Utilisez le Makefile pour une configuration facile :

```bash
make setup
```

Cela va :
- Créer un environnement virtuel Python
- Installer toutes les dépendances (spaCy, scikit-learn, matplotlib, numpy)
- Télécharger le modèle de langue français (fr_core_news_sm)

### 2. Recherche interactive

Essayez le moteur de recherche avec vos propres requêtes :

```bash
make search
```

Sortie attendue :
```
================================================================
Starting interactive search engine...
================================================================

venv/bin/python search_engine.py
Initializing search engine...
Loaded 2000 documents from wiki_split_extract_2k/
Computing term frequencies...
Computing inverse document frequencies...
Building TF-IDF index...
Building reverse index...
Index built with 2000 documents and 49081 unique terms

Index Statistics:
  num_documents: 2000
  num_unique_terms: 49081
  avg_terms_per_doc: 116.955
  clean_params: {'remove_punctuation': True, 'remove_stopwords': True, 'lemmatize': True}
  use_log_tf: True

======================================================================
Interactive Search - Enter your queries (type 'quit' or 'exit' to stop)
======================================================================


Search query:
```

### 3. Comparaison des configurations

Comparez différentes combinaisons de fonctionnalités et générez des graphiques de performance :

```bash
make compare
```
Sortie attendue :
```
================================================================
Comparing different feature configurations...
================================================================

venv/bin/python compare_features.py
Loading queries...
Loaded 100 queries

Found 2000 documents

Testing: Manual TF-IDF
Building TF-IDF index...
Evaluating...
Precision@10: 0.8753
Recall@10:    0.9700
F1@10:        0.9202

Testing: Manual TF-IDF (log)
Building TF-IDF index...
Evaluating...
Precision@10: 0.9251
Recall@10:    0.9700
F1@10:        0.9470

Testing: Scikit-learn TF-IDF
Building TF-IDF index...
Evaluating...
Precision@10: 0.8978
Recall@10:    0.9700
F1@10:        0.9325

Testing: Manual + Text cleaning
Building TF-IDF index...
Evaluating...
Precision@10: 0.8643
Recall@10:    0.9700
F1@10:        0.9141

Testing: Manual + Text cleaning (log)
Building TF-IDF index...
Evaluating...
Precision@10: 0.9112
Recall@10:    0.9700
F1@10:        0.9397

Testing: Scikit + Text cleaning
Building TF-IDF index...
Evaluating...
Precision@10: 0.8692
Recall@10:    0.9700
F1@10:        0.9168


======================================================================
COMPARISON TABLE
======================================================================
Configuration                       Precision@10    Recall@10       F1@10     
----------------------------------------------------------------------
Manual TF-IDF                       0.8753          0.9700          0.9202    
Manual TF-IDF (log)                 0.9251          0.9700          0.9470    
Scikit-learn TF-IDF                 0.8978          0.9700          0.9325    
Manual + Text cleaning              0.8643          0.9700          0.9141    
Manual + Text cleaning (log)        0.9112          0.9700          0.9397    
Scikit + Text cleaning              0.8692          0.9700          0.9168    
======================================================================

Precision graph saved as 'precision_comparison.png'
Recall graph saved as 'recall_comparison.png'
F1 graph saved as 'f1_comparison.png'

================================================================
Comparison complete - Check generated PNG files
================================================================
```

Cela génère 3 fichiers PNG :
- `precision_comparison.png`
- `recall_comparison.png`
- `f1_comparison.png`

## Implémentation technique

### Calcul TF-IDF

Le moteur de recherche implémente TF-IDF avec deux options de normalisation :

1. **TF classique** : `tf(t,d) = count(t,d) / |d|`
2. **TF logarithmique** : `tf(t,d) = log(1 + count(t,d))`

Formule IDF : `idf(t) = log(N / (1 + df(t)))`

Score final : `tfidf(t,d) = tf(t,d) × idf(t)`

### Traitement de texte

Pipeline de nettoyage de texte :
1. **Minuscules** : Conversion de tout le texte en minuscules
2. **Tokenisation** : Séparation sur les espaces ou utilisation de spaCy
3. **Suppression des stopwords** : Filtrage des mots français courants
4. **Lemmatisation** : Réduction des mots à leur forme de base avec spaCy
5. **Normalisation des accents** : Normalisation NFD (optionnel)

### Classement

Les documents sont classés en utilisant la similarité cosinus :

```
similarity(q, d) = (q · d) / (||q|| × ||d||)
```

Où :
- `q` est le vecteur de requête
- `d` est le vecteur de document
- `·` est le produit scalaire
- `||·||` est la norme L2

### Métriques d'évaluation

- **Precision@K** : Rang réciproque (1/rang) si trouvé dans les top-K, sinon 0
- **Recall@K** : Binaire (1 si trouvé dans les top-K, sinon 0)
- **F1@K** : Moyenne harmonique de Precision@K et Recall@K

## Commandes Makefile

```bash
make help           # Afficher toutes les commandes disponibles
make setup          # Configuration complète (première fois)
make search         # Démo de recherche interactive
make compare        # Comparer les configurations
make clean          # Supprimer les fichiers générés
```