# backend/app/inference/classifier.py

import joblib
import os
from typing import Dict
from app.nlp.preprocess import preprocess_text
from app.nlp.vectorizer import transform_text, set_vectorizer

# Paths para os modelos
MODEL_PATH = "app/models/classifier.joblib"
VECTORIZER_PATH = "app/models/vectorizer.joblib"

# Threshold de confiança mínima
CONFIDENCE_THRESHOLD = 0.75

# Carregando vectorizer e modelo na inicialização
if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError("Modelos ou vectorizer não encontrados. Treine antes de iniciar.")

trained_vectorizer = joblib.load(VECTORIZER_PATH)
set_vectorizer(trained_vectorizer)

classifier_model = joblib.load(MODEL_PATH)


def classify_email(text: str) -> Dict:
    """
    Recebe um texto, pré-processa, vetoriza, classifica e retorna resultado.
    """
    print("🔹 Texto original:", text)

    # 1️⃣ Pré-processamento
    clean_text = preprocess_text(text)
    print("🔹 Texto pré-processado:", clean_text)

    # 2️⃣ Vetorização
    vector = transform_text(clean_text).toarray()
    print("🔹 Vetor TF-IDF:", vector)

    # 3️⃣ Classificação
    prediction = classifier_model.predict(vector)[0]
    probabilities = classifier_model.predict_proba(vector)[0]
    class_index = list(classifier_model.classes_).index(prediction)
    confidence = probabilities[class_index]

    print(f"🔹 Predição: {prediction}, Confiança: {confidence:.3f}")

    result = {
        "text": text,
        "prediction": prediction,
        "confidence": round(float(confidence), 3)
    }

    # 4️⃣ Threshold (aqui podemos eventualmente chamar Gemini se desejar)
    if confidence < CONFIDENCE_THRESHOLD:
        result["note"] = "Confiança baixa, considerar validação externa."

    print("🔹 Resultado final:", result)
    return result
