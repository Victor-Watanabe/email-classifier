# backend/app/inference/classifier.py

import joblib
import os
from typing import Dict

from app.nlp.preprocess import preprocess_text
from app.nlp.vectorizer import transform_text, set_vectorizer
from app.inference.gemini_service import query_gemini

# ============================
# Paths dos modelos
# ============================
MODEL_PATH = "app/models/classifier.joblib"
VECTORIZER_PATH = "app/models/vectorizer.joblib"

# ============================
# Configurações
# ============================
CONFIDENCE_THRESHOLD = 0.75
MIN_TOKEN_LENGTH = 3  # evita textos vazios ou irrelevantes

# ============================
# Respostas fixas (SEM Gemini)
# ============================
FIXED_REPLIES = {
    "PRODUTIVO": (
        "Olá! Recebemos sua mensagem e ela foi encaminhada para análise. "
        "Nossa equipe retornará com mais informações o mais breve possível."
    ),
    "IMPRODUTIVO": (
        "Olá! Agradecemos o contato. Sua mensagem foi recebida com sucesso."
    )
}

# ============================
# Load models
# ============================
if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError(
        "Modelos ou vectorizer não encontrados. Treine antes de iniciar."
    )

trained_vectorizer = joblib.load(VECTORIZER_PATH)
set_vectorizer(trained_vectorizer)

classifier_model = joblib.load(MODEL_PATH)

# ============================
# Confidence calibration
# ============================
def boost_confidence(confidence: float) -> float:
    """
    Reduz o excesso de cautela da Logistic Regression
    sem alterar a classe prevista.
    """
    if confidence >= 0.5:
        return min(1.0, confidence ** 0.5)
    return confidence

# ============================
# Classificação principal
# ============================
def classify_email(text: str) -> Dict:
    """
    Classifica um email como PRODUTIVO ou IMPRODUTIVO.

    Fluxo:
    - Regra de negócio para textos vazios/curtos
    - Classificação local (TF-IDF + Logistic Regression)
    - Fallback para Gemini em caso de baixa confiança
    """

    print("🔹 Texto original:", text)

    # 1️⃣ Pré-processamento
    clean_text = preprocess_text(text)
    print("🔹 Texto pré-processado:", clean_text)

    # 🚨 REGRA DE NEGÓCIO: texto vazio ou irrelevante
    if not clean_text or len(clean_text.split()) < MIN_TOKEN_LENGTH:
        result = {
            "text": text,
            "prediction": "IMPRODUTIVO",
            "confidence": 0.95,
            "reply": FIXED_REPLIES["IMPRODUTIVO"],
            "source": "rule_based",
            "note": "Texto vazio, curto ou sem conteúdo acionável."
        }
        print("🔹 Resultado final (regra de negócio):", result)
        return result

    # 2️⃣ Vetorização
    vector = transform_text(clean_text).toarray()

    # 3️⃣ Classificação local
    prediction = classifier_model.predict(vector)[0]
    probabilities = classifier_model.predict_proba(vector)[0]
    class_index = list(classifier_model.classes_).index(prediction)
    raw_confidence = probabilities[class_index]
    confidence = boost_confidence(raw_confidence)

    print(
        f"🔹 Predição local: {prediction}, "
        f"Confiança bruta: {raw_confidence:.3f}, "
        f"Confiança ajustada: {confidence:.3f}"
    )

    # 4️⃣ Confiança suficiente → usa IA LOCAL
    if confidence >= CONFIDENCE_THRESHOLD:
        result = {
            "text": text,
            "prediction": prediction,
            "confidence": round(float(confidence), 3),
            "reply": FIXED_REPLIES[prediction],
            "source": "local_model"
        }
        print("🔹 Resultado final (modelo local):", result)
        return result

    # 5️⃣ Confiança baixa → fallback Gemini
    print("⚠️ Confiança abaixo do threshold. Consultando Gemini...")

    gemini_response = query_gemini(text)

    result = {
        "text": text,
        "prediction": gemini_response.get("classification"),
        "reply": gemini_response.get("suggested_reply"),
        "justification": gemini_response.get("justification"),
        "confidence": round(float(confidence), 3),
        "source": "gemini_fallback"
    }

    print("🔹 Resultado final (Gemini):", result)
    return result
