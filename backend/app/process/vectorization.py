from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

# Considera as 5000 palavras mais importantes.
vectorizer = TfidfVectorizer(max_features=5000)

def fit_vectorizer(corpus: list):
    """
    Treina o vetorizer TF-IDF com o corpus de emails.

    Parâmetro:
        corpus (list): lista de textos limpos

    Retorno:
        vectorizer: objeto TF-IDF treinado
    """
    global vectorizer
    vectorizer.fit(corpus)

    # Cria diretório se não existir
    os.makedirs("backend/cache", exist_ok=True)

    # Salva o vectorizer
    joblib.dump(vectorizer, "backend/cache/vectorizer.joblib")
    print("Vectorizer treinado e salvo em backend/cache/vectorizer.joblib")
    return vectorizer

def transform_text(text: str):
    """
    Transforma um texto limpo em vetor numérico para o modelo.

    Parâmetro:
        text (str): texto pré-processado

    Retorno:
        vetor TF-IDF (sparse matrix)
    """
    global vectorizer
    return vectorizer.transform([text])

def load_vectorizer(path: str = "backend/cache/vectorizer.joblib"):
    """
    Carrega um vetor TF-IDF previamente treinado.

    Parâmetro:
        path (str): caminho do arquivo .joblib

    Retorno:
        vectorizer carregado
    """
    global vectorizer
    vectorizer = joblib.load(path)
    print(f"Vectorizer carregado de {path}")
    return vectorizer


# =======================
# TESTE RÁPIDO DO VETORIZER
# =======================
if __name__ == "__main__":
    print("=== Testando Vectorization ===")
    
    # Corpus de teste
    corpus = [
        "Preciso do relatório financeiro",
        "Desejo feliz natal",
        "Qual o status da minha solicitação?"
    ]
    
    # Treina o vectorizer
    print("Treinando vectorizer...")
    fit_vectorizer(corpus)
    
    # Transforma um texto novo
    text_novo = "Preciso de atualização do relatório X"
    vector = transform_text(text_novo)
    
    print("\nTexto transformado:", text_novo)
    print("Shape do vetor TF-IDF:", vector.shape)
    print("Vetor esparso (primeiros 10 elementos):", vector.toarray()[0][:10])
