from pathlib import Path

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist

import matplotlib.pyplot as plt


def nltk_discovery():
    # Le tokensizer
    nltk.download('punkt')
    nltk.download('punkt_tab')

    # filtrage des mots inutiles
    nltk.download('stopwords')

    # POS Tagging -> Reconnaissance gramaticale
    nltk.download('averaged_perceptron_tagger')
    nltk.download('averaged_perceptron_tagger_eng')

    # Liste des mots qui existent dans le monde.
    nltk.download('wordnet')

    # la Segmentation / Tokenization
    text = Path("corpus.txt").read_text()
    # print(sent_tokenize(text))

    tokens = word_tokenize(text)
    # print(len(tokens))
    # print(tokens)


    # Le Nettoyage (Stopwords)
    stop_words = set(stopwords.words())
    filtered_text = [word for word in tokens if word.lower() not in stop_words]

    # print(len(stop_words))
    # print(len(filtered_text))
    # print(filtered_text)

    # Analyse Morphologique -> POS Tagging
    tags = nltk.pos_tag(tokens)
    print(tags)

    # Lemmatisation
    nouns = [tag[0] for tag in tags if tag[1] in ['NN', 'NNS', 'NNP', 'NNP']]
    verbs = [tag[0] for tag in tags if tag[1] in ['VBZ', 'FW', 'VB']]

    lemmatizer = WordNetLemmatizer()

    lemmatized_nouns = [lemmatizer.lemmatize(word) for word in nouns]
    # print(nouns)
    # print(lemmatized_nouns)

    lemmatized_verbs = [lemmatizer.lemmatize(word) for word in verbs]
    # print(verbs)
    # print(lemmatized_verbs)

    frequency_distribution = FreqDist(lemmatized_nouns)
    # print(frequency_distribution.most_common(5))

    plt.figure(figsize=(10, 6))
    frequency_distribution.plot(30, cumulative=False, title="Fréquence des lemmes (Noms, verbes)")
    plt.show()

def give_date_time():
    from datetime import datetime
    return datetime.now()


def langchain_discovery():

    from langchain.agents import create_agent
    from langchain_ollama import ChatOllama
    from langchain.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

    sml = ChatOllama(model="gemma3:4b", temperature=0.3)

    agent = create_agent(model=sml, system_prompt=SystemMessage("You are a helpful assistant. Never use emoji. Your answer should be short and conscice, always responds in French. {}"))
    response = agent.invoke(input={
        "messages": [
            # {
            #     "role": "user",
            #     "content": "Je m'appelle Nicolas."
            # }

            HumanMessage(content="Je m'appelle Nicolas. Quelle est la date d'aujourd'hui ?")
        ]
    }, )
    print(response["messages"][-1].content)


def main():
    langchain_discovery()
