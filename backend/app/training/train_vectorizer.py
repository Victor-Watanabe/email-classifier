import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from app.pipeline.preprocess import preprocess_text

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
    """Carrega os textos do dataset, ignorando linhas vazias"""
    with open(path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]

# ============================
# Treinamento
# ============================
def train_vectorizer(
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.9
):
    print("🔢 Lendo dataset...")
    texts = load_texts(DATASET_PATH)
    print(f"📄 Total de emails carregados: {len(texts)}")

    print("✂️ Pré-processando textos...")
    processed = [preprocess_text(text) for text in texts]
    print(f"✅ Pré-processamento concluído, exemplos:")
    for t in processed[:3]:
        print(f"  {t}")

    print("🧠 Treinando vectorizer TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df
    )
    vectorizer.fit(processed)

    print("💾 Salvando modelo...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(vectorizer, MODEL_PATH)

    print("🎉 Vectorizer treinado com sucesso!")
    print(f"📚 Vocabulário: {len(vectorizer.vocabulary_)} termos")
    print(f"🔹 Primeiros 20 termos: {list(vectorizer.vocabulary_.keys())[:20]}")

