import spacy
import unicodedata

nlp = spacy.load('fr_core_news_sm')


def clean_text(text, lemmatize=False, normalize=False):
    # on lemmatise avec spaCy si demandé, sinon on split simplement
    if lemmatize:
        doc = nlp(text.lower())
        words = [token.lemma_ for token in doc if not token.is_space]
    else:
        words = text.lower().split()
    
    # on normalise en retirant la ponctuation et les accents si demandé
    if normalize:
        words = [
            ''.join(c for c in unicodedata.normalize('NFD', word) 
                   if unicodedata.category(c) not in ['Mn', 'Po', 'Ps', 'Pe', 'Pc', 'Pd', 'Pf', 'Pi'])
            for word in words
        ]
        # on retire les mots vides après normalisation
        words = [word for word in words if word]

    return words