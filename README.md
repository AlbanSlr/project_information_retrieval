# Information Retrieval Search Engine
### Alban Sellier & Rémi Geraud

Moteur de recherche optimisé pour Wikipédia français utilisant TF-IDF avec normalisation logarithmique et nettoyage de texte.

## Vue d'ensemble du projet

Ce projet implémente un système de recherche d'information performant pour parcourir 2 000 documents de Wikipédia en français. Le moteur utilise **TF-IDF optimisé** comme configuration principale, qui s'est révélée être la plus performante après comparaison avec différentes approches (SBERT, Hybrid).

### Moteur principal : TF-IDF (log + cleaning)

**Configuration optimale** :
- ✅ Normalisation log(1+tf) : Réduit le biais des termes très fréquents
- ✅ Suppression des stopwords : Filtre 583 mots français non pertinents
- ✅ Lemmatisation spaCy : Normalise les formes de mots
- ✅ Index inversé : Recherche rapide et efficace

**Performance** :
- Precision@10 : ~0.46
- Recall@10 : ~0.52
- F1@10 : ~0.49

### Études comparatives

Le projet inclut deux études de performance :

1. **Comparaison de features TF-IDF** (`compare_features.py`)
   - Compare 4 configurations progressives
   - Montre l'amélioration de chaque feature
   - Génère `tfidf_features_comparison.png`

2. **Comparaison de modèles** (`compare_models.py`)
   - Compare TF-IDF vs SBERT vs Hybrid
   - Démontre que TF-IDF optimisé est optimal
   - Génère `model_comparison.png`

## Structure du projet

```
project_information_retrieval/
├── engines/                  # Moteurs de recherche
│   ├── base_engine.py       # Classe abstraite BaseSearchEngine
│   ├── tfidf_engine.py      # ⭐ TF-IDF optimisé (moteur principal)
│   └── hybrid_engine.py     # Hybrid TF-IDF + SBERT (expérimental)
├── utils/                   # Utilitaires
│   ├── evaluation.py        # Métriques d'évaluation
│   └── text_processing.py   # Nettoyage de texte
├── search_engine.py         # Point d'entrée principal
├── compare_features.py      # Comparaison features TF-IDF
├── compare_models.py        # Comparaison modèles
├── requetes.jsonl          # 116 requêtes de test
├── wiki_split_extract_2k/  # 2000 documents Wikipédia
├── stop_words_french.json  # Stopwords français
├── Makefile                # Automatisation
└── README.md               # Ce fichier
```

## Démarrage rapide

### 1. Configuration de l'environnement

Le projet nécessite Python 3.12. Utilisez le Makefile :

```bash
make setup
```

Cela va :
- Créer un environnement virtuel Python 3.12
- Installer les dépendances (spaCy, scikit-learn, matplotlib, numpy, sentence-transformers)
- Télécharger le modèle français fr_core_news_sm

**Important** : Décompressez `wiki_split_extract_2k.zip` dans le répertoire racine.

### 2. Recherche interactive

Essayez le moteur TF-IDF optimisé :

```bash
make search
```

Sortie attendue :
```
================================================================
Starting TF-IDF search engine (log + cleaning)
================================================================

Initializing search engine...
Loaded 2000 documents from wiki_split_extract_2k/
Computing term frequencies...
Computing inverse document frequencies...
Building TF-IDF index...
Index built with 2000 documents and ~45000 unique terms

Index Statistics:
  num_documents: 2000
  num_unique_terms: 45123
  use_log_tf: True
  clean_params: {'remove_punctuation': True, 'remove_stopwords': True, 'lemmatize': True}

======================================================================
Interactive Search - Enter your queries (type 'quit' or 'exit' to stop)
======================================================================

Search query: révolution française
```

### 3. Comparaisons de performance

#### Comparer les features TF-IDF

```bash
make compare-features
```

Compare 4 configurations :
- Baseline (pas de features)
- Baseline + Stopwords
- Baseline + Stopwords + Lemmatize
- ⭐ Optimized (+ log normalization)

Génère : `tfidf_features_comparison.png`

#### Comparer les modèles

```bash
make compare-models
```

Compare 3 approches :
- ⭐ TF-IDF (log + cleaning)
- SBERT (multilingual)
- Hybrid (TF-IDF + SBERT)

Génère : `model_comparison.png`

#### Tout comparer

```bash
make compare
```

Lance les deux comparaisons et génère les deux graphiques.

### 4. Nettoyage

```bash
make clean        # Supprime les PNG et __pycache__
make clean-cache  # Supprime aussi data/ (caches SBERT)
make clean-all    # Supprime tout y compris venv/
```

## Détails techniques

### TF-IDF Optimisé

**Formules** :
```
TF (log normalization) = log(1 + count)
IDF = log(N / df)
TF-IDF = TF × IDF
Similarité = cosine_similarity(query_vec, doc_vec)
```

**Prétraitement** :
1. Tokenization (spaCy)
2. Suppression ponctuation
3. Suppression stopwords (583 mots)
4. Lemmatisation (spaCy fr_core_news_sm)
5. Normalisation log(1+tf)

**Structures de données** :
- **Index direct** : `{doc_id: {term: tf-idf}}`
- **Index inversé** : `{term: {doc_id: tf-idf}}`

### Métriques d'évaluation

#### Precision@10 (Reciprocal Rank)
```
Precision@10 = 1 / rank_first_relevant
```
Mesure la position du premier document pertinent.

#### Recall@10 (Binary)
```
Recall@10 = 1 si ≥1 document pertinent dans top 10, sinon 0
```

#### F1@10
```
F1@10 = 2 × (Precision × Recall) / (Precision + Recall)
```

## Résultats et conclusions

### Pourquoi TF-IDF (log + cleaning) est optimal ?

Après comparaison exhaustive, TF-IDF optimisé s'est révélé être le meilleur choix pour ce projet :

**Avantages** :
- ✅ **Performance** : F1@10 ~0.49 (compétitif avec SBERT)
- ✅ **Rapidité** : Construction d'index < 5s, recherche instantanée
- ✅ **Simplicité** : Pas de modèle externe lourd
- ✅ **Interprétabilité** : Scores TF-IDF explicables
- ✅ **Robustesse** : Fonctionne bien sur requêtes par mots-clés

**Comparaison avec alternatives** :
- SBERT : Meilleure sémantique mais 10x plus lent et moins explicable
- Hybrid : Légèrement meilleur F1 mais complexité excessive
- Doc2Vec/FastText : Performances inférieures à TF-IDF optimisé

### Impact des features

| Configuration | Precision@10 | Recall@10 | F1@10 | Amélioration |
|---------------|--------------|-----------|-------|--------------|
| Baseline | 0.342 | 0.415 | 0.375 | - |
| + Stopwords | 0.398 | 0.469 | 0.431 | +0.056 |
| + Lemmatize | 0.421 | 0.495 | 0.456 | +0.025 |
| **+ Log normalization** | **0.458** | **0.523** | **0.488** | **+0.032** |

Chaque feature apporte une amélioration mesurable, avec la normalisation logarithmique donnant le gain final significatif.

## Dépendances

- **Python** : 3.12 (requis pour compatibilité NumPy/spaCy)
- **spaCy** : 3.x avec fr_core_news_sm
- **scikit-learn** : TF-IDF et similarité cosinus
- **matplotlib** : Visualisation des performances
- **numpy** : Opérations matricielles
- **sentence-transformers** : Pour comparaison avec SBERT (optionnel)

## Commandes Makefile

```bash
make help             # Affiche l'aide
make setup            # Configuration complète
make search           # Recherche interactive
make compare-features # Compare features TF-IDF
make compare-models   # Compare modèles
make compare          # Les deux comparaisons
make clean            # Supprime PNG et caches Python
make clean-cache      # Supprime data/ (modèles)
make clean-all        # Supprime tout (venv inclus)
```

## Notes importantes

1. **Première exécution** : Construction de l'index TF-IDF prend ~5 secondes.

2. **Modèles optionnels** : SBERT et Hybrid sont implémentés pour comparaison mais ne sont pas le moteur principal.

3. **Compatibilité Python** : Python 3.12 requis pour éviter les problèmes NumPy/spaCy.

## Contribution

Projet développé par :
- **Alban Sellier** : Architecture, TF-IDF optimisé, évaluation
- **Rémi Geraud** : Exploration SBERT, FastText, Doc2Vec

## Licence

Projet académique - Master 2 TAL (Traitement Automatique des Langues)
