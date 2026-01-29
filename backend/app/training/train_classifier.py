from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, classification_report
import joblib
import os

from app.nlp.preprocess import preprocess_text
from app.nlp.vectorizer import set_vectorizer, transform_text


DATASET_PATH = "app/training/datasets/classifier.txt"
VECTORIZER_PATH = "app/models/vectorizer.joblib"
MODEL_PATH = "app/models/classifier.joblib"


def load_dataset(path: str):
    texts = []
    labels = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line or "|" not in line:
                continue

            text, label = line.split("|", 1)
            texts.append(text.strip())
            labels.append(label.strip().upper())

    return texts, labels


def train_classifier():
    print("📥 Carregando dataset...")
    texts, labels = load_dataset(DATASET_PATH)

    print("🧹 Pré-processando textos...")
    processed_texts = [preprocess_text(text) for text in texts]

    print("📦 Carregando vectorizer treinado...")
    trained_vectorizer = joblib.load(VECTORIZER_PATH)
    set_vectorizer(trained_vectorizer)

    print("🔢 Vetorizando textos...")
    X = [transform_text(text).toarray()[0] for text in processed_texts]
    y = labels

    print("✂️ Separando treino e teste (estratificado)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("🤖 Treinando classificador...")
    model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    C=0.7  # testa 0.3, 0.5, 0.7, 1.0
    )
    
    model.fit(X_train, y_train)

    print("📊 Avaliando modelo...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test,
        y_pred,
        pos_label="PRODUTIVO"
    )

    print(f"✅ Accuracy: {acc:.2f}")
    print(f"🎯 Precision (PRODUTIVO): {precision:.2f}")

    print("\n📄 Relatório completo:")
    print(classification_report(y_test, y_pred))

    print("💾 Salvando classificador...")
    os.makedirs("app/models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"🎉 Modelo salvo em {MODEL_PATH}")


if __name__ == "__main__":
    train_classifier()
