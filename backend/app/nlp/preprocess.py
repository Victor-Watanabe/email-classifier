import spacy

nlp = spacy.load("pt_core_news_sm")

def preprocess_text(text: str) -> str:
    """
    Pré-processa texto para o modelo:
    - converte para minúsculo
    - remove stopwords e pontuação
    - aplica lematização
    - mantém nomes próprios e entidades importantes
    """
    doc = nlp(text.lower())

    tokens = []

    for token in doc:
        # ignora stopwords e pontuação
        if token.is_stop or token.is_punct:
            continue

        # mantêm nomes próprios e entidades
        if token.ent_type_ in ["PER", "ORG", "LOC", "MISC"]:
            tokens.append(token.text)  # mantem forma original para entidades
        else:
            tokens.append(token.lemma_)  # lematiza palavras comuns

    return " ".join(tokens)
