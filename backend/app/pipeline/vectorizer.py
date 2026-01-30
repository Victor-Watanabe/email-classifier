from sklearn.feature_extraction.text import TfidfVectorizer

# Vectorizer será carregado externamente (ex: na inicialização da API)
vectorizer: TfidfVectorizer | None = None


def set_vectorizer(trained_vectorizer: TfidfVectorizer):
    """
    Define o vectorizer já treinado para uso no pipeline.
    """
    global vectorizer
    vectorizer = trained_vectorizer


def transform_text(text: str):
    """
    Transforma um texto pré-processado em vetor TF-IDF.

    Parâmetro:
        text (str): texto limpo

    Retorno:
        vetor TF-IDF (sparse matrix)
    """
    if vectorizer is None:
        raise RuntimeError("Vectorizer não foi inicializado.")

    return vectorizer.transform([text])
