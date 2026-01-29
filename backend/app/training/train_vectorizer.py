import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from app.nlp.preprocess import preprocess_text

# ============================
# Paths absolutos do projeto
# ============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "app",
    "training",
    "datasets",
    "emails.txt"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "app",
    "models",
    "vectorizer.joblib"
)

# ============================
# Dataset
# ============================
def load_texts(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]

# ============================
# Treinamento
# ============================
def train_vectorizer():
    print("Lendo dataset...")
    texts = load_texts(DATASET_PATH)

    print("Pré-processando textos...")
    processed = [preprocess_text(text) for text in texts]

    print("Treinando vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9
    )

    vectorizer.fit(processed)

    print("Salvando modelo...")
    joblib.dump(vectorizer, MODEL_PATH)

    print("Vectorizer treinado com sucesso!")
    print(f"Vocabulário: {len(vectorizer.vocabulary_)} termos")

# ============================
if __name__ == "__main__":
    train_vectorizer()
