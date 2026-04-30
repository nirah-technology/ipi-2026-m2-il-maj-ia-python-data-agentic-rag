import nltk

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords



def main():
    # Le tokensizer
    nltk.download('punkt')
    nltk.download('punkt_tab')

    # La lemmatisation -> filtrage des mots inutiles
    nltk.download('stopwords')

    # POS Tagging -> Reconnaissance gramaticale
    nltk.download('averaged_perceptron_tagger')
    nltk.download('averaged_perceptron_tagger_eng')

    # Liste des mots qui existent dans le monde.
    nltk.download('wordnet')

    # la Segmentation / Tokenization
    text = "Je dis bonjour à mes étudiants de l'IPI qui sont en Master 2 spé Ingénieurie Logicielle. Je suis ravi d'être avec vous. La moyenne de la classe est de 15.95, c'est pas mal du tout."
    print(sent_tokenize(text))

    tokens = word_tokenize(text)
    print(len(tokens))
    print(tokens)


    # Le Nettoyage (Stopwords)
    stop_words = set(stopwords.words('french'))
    filtered_text = [word for word in tokens if word.lower() not in stop_words]

    print(len(stop_words))
    print(len(filtered_text))
    print(filtered_text)


    # Analyse Morphologique -> POS Tagging
    tags = nltk.pos_tag(tokens)
    print(tags)
