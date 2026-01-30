# backend/app/inference/classifier.py

import joblib
import os
#import csv

from typing import Dict
from app.pipeline.preprocess import preprocess_text
from app.pipeline.vectorizer import transform_text, set_vectorizer
from ..inference.gemini_service import query_gemini

# Paths dos modelos
MODEL_PATH = "app/models/classifier.joblib"
VECTORIZER_PATH = "app/models/vectorizer.joblib"

# Dataset incremental (Gemini)
# DATASET_DIR = "app/training/datasets"
# DATASET_PATH = f"{DATASET_DIR}/gemini_feedback.csv"

# Configurações
CONFIDENCE_THRESHOLD = 0.75
MIN_TOKEN_LENGTH = 3

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

# Load models
if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError(
        "Modelos ou vectorizer não encontrados. Treine antes de iniciar."
    )

trained_vectorizer = joblib.load(VECTORIZER_PATH)
set_vectorizer(trained_vectorizer)

classifier_model = joblib.load(MODEL_PATH)

# Confidence calibration
def boost_confidence(confidence: float) -> float:
    if confidence >= 0.5:
        return min(1.0, confidence ** 0.5)
    return confidence

# 🔮 FUNÇÃO FUTURA (COMENTADA)
# 
# def save_gemini_feedback(email_text: str, classification: str):
#     """
#     FUTURO:
#     Salva classificações realizadas pelo Gemini em um CSV
#     para aprendizado supervisionado incremental.
#
#     ⚠️ Este recurso está DESABILITADO no momento.
#     A proposta é permitir que um humano revise os dados
#     antes de re-treinar o modelo local.
#     """
#
#     os.makedirs(DATASET_DIR, exist_ok=True)
#     file_exists = os.path.exists(DATASET_PATH)
#
#     with open(DATASET_PATH, mode="a", newline="", encoding="utf-8") as file:
#         writer = csv.writer(file)
#
#         if not file_exists:
#             writer.writerow(["email_text", "classification"])
#
#         writer.writerow([email_text, classification])

# Classificação principal
def classify_email(text: str) -> Dict:
    print("🔹 Texto original:", text)

    # 1️⃣ Pré-processamento
    clean_text = preprocess_text(text)
    print("🔹 Texto pré-processado:", clean_text)

    # 🚨 REGRA DE NEGÓCIO
    if not clean_text or len(clean_text.split()) < MIN_TOKEN_LENGTH:
        result = {
            "text": text,
            "prediction": "IMPRODUTIVO",
            "confidence": 0.95,
            "reply": FIXED_REPLIES["IMPRODUTIVO"],
            "source": "rule_based",
            "note": "Texto vazio, curto ou sem conteúdo acionável."
        }
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
        f"Confiança ajustada: {confidence:.3f}"
    )

    # 4️⃣ Confiança suficiente → modelo local
    if confidence >= CONFIDENCE_THRESHOLD:
        return {
            "text": text,
            "prediction": prediction,
            "confidence": round(float(confidence), 3),
            "reply": FIXED_REPLIES[prediction],
            "source": "local_model"
        }

    # 5️⃣ Confiança baixa → Gemini
    print("⚠️ Confiança baixa. Consultando Gemini...")

    gemini_response = query_gemini(text)
    gemini_classification = gemini_response.get("classification")

    # 🧠 Salva para aprendizado futuro
    #if gemini_classification:
    #   save_gemini_feedback(text, gemini_classification)

    return {
        "text": text,
        "prediction": gemini_classification,
        "reply": gemini_response.get("suggested_reply"),
        "justification": gemini_response.get("justification"),
        "confidence": round(float(confidence), 3),
        "source": "gemini_fallback"
    }