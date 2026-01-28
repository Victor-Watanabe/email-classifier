# backend/app/avaliation/classifier_my_llm.py

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from ..nlp.vectorizer import transform_text, fit_vectorizer, load_vectorizer
from ..nlp.preprocess import preprocess_text
from .gemini_service import query_gemini  # função que consulta Gemini API se necessário

# Inicializa o modelo
model = LogisticRegression()

def train_classifier(corpus: list, labels: list):
    """
    Treina o classificador com exemplos anotados.

    Parâmetros:
        corpus (list): textos limpos
        labels (list): categorias correspondentes ('Produtivo' ou 'Improdutivo')

    Retorno:
        model: modelo treinado
    """
    global model

    # Treina o vectorizer (TF-IDF) se ainda não estiver treinado
    fit_vectorizer(corpus)

    # Converte textos em vetores numéricos
    X_sparse = [transform_text(text) for text in corpus]
    X = [x.toarray()[0] for x in X_sparse]  # transforma sparse matrix em array
    y = labels

    # Divide treino e teste (apenas para validação interna)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treina o modelo
    model.fit(X_train, y_train)

    # Avalia performance rapidamente
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Classificador treinado! Accuracy no teste: {acc:.2f}")

    # Salva o modelo para uso futuro
    joblib.dump(model, "models/classifier.joblib")
    return model


def load_classifier(path: str = "models/classifier.joblib"):
    """
    Carrega modelo treinado do disco.
    """
    global model
    model = joblib.load(path)
    return model


def classify_email(text: str, confidence_threshold: float = 0.8):
    """
    Classifica um email como Produtivo ou Improdutivo.
    Se a confiança for baixa, envia para Gemini API.

    Parâmetros:
        text (str): email bruto
        confidence_threshold (float): limiar mínimo para confiar no modelo

    Retorno:
        dict: {
            'classification': str,
            'suggested_reply': str,
            'justification': str,
            'confidence': float,
            'gemini_used': bool
        }
    """
    global model

    # Pré-processa
    clean_text = preprocess_text(text)

    # Vetoriza
    vector = transform_text(clean_text).toarray()

    # Predição
    pred_prob = model.predict_proba(vector)[0]
    pred_class = model.classes_[pred_prob.argmax()]
    confidence = pred_prob.max()

    # Se confiança baixa, usa Gemini API
    if confidence < confidence_threshold:
        gemini_response = query_gemini(text)
        return {
            "classification": gemini_response.get("classification", "Desconhecido"),
            "suggested_reply": gemini_response.get("suggested_reply", ""),
            "justification": gemini_response.get("justification", ""),
            "confidence": confidence,
            "gemini_used": True
        }

    # Caso a confiança seja alta, retorna apenas a classificação do modelo
    return {
        "classification": pred_class,
        "suggested_reply": "",
        "justification": "",
        "confidence": confidence,
        "gemini_used": False
    }


# --- Teste rápido ---
if __name__ == "__main__":
    print("=== Testando Classificador Mini LLM ===")
    # Dados de exemplo
    examples = [
        "Preciso de atualização do relatório X",
        "Feliz Natal e boas festas!",
        "O pagamento da fatura ainda não foi processado",
        "Obrigado pelo envio das fotos"
    ]
    labels = ["Produtivo", "Improdutivo", "Produtivo", "Improdutivo"]

    train_classifier(examples, labels)

    # Teste de classificação
    teste = "Preciso da atualização do relatório dessa semana."
    resultado = classify_email(teste)
    print("Email:", teste)
    print("Resultado:", resultado)
