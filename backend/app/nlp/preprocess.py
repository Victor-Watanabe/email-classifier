import spacy
import re

# ============================
# Carregamento do modelo spaCy
# ============================
nlp = spacy.load("pt_core_news_sm")

# ============================
# Frases sociais / cordialidades
# ============================
SOCIAL_PHRASES = [
    "bom dia",
    "boa tarde",
    "boa noite",
    "feliz natal",
    "feliz ano novo",
    "boas festas",
    "bom feriado",
    "bom carnaval",
    "espero que esteja bem",
    "espero que esteja tudo bem",
    "como vai",
    "tudo bem",
    "saudações",
    "fim de ano",
    "bom fim de ano"
]

# ============================
# Remove cordialidades sociais
# ============================
def remove_social_phrases(text: str) -> str:
    text = text.lower()
    for phrase in SOCIAL_PHRASES:
        # remove frase inteira respeitando limites de palavra
        text = re.sub(rf"\b{re.escape(phrase)}\b", "", text)
    return text


# ============================
# Pré-processamento principal
# ============================
def preprocess_text(text: str) -> str:
    """
    Pré-processa texto para o modelo:
    - remove cordialidades sociais previsíveis
    - converte para minúsculo
    - remove stopwords e pontuação
    - aplica lematização
    - preserva entidades relevantes (PER, ORG, LOC, MISC)
    """

    # 1️⃣ Remove ruído social ANTES de NLP
    text = remove_social_phrases(text)

    # 2️⃣ Processa com spaCy
    doc = nlp(text)

    tokens = []

    for token in doc:
        # ignora stopwords, pontuação e espaços
        if token.is_stop or token.is_punct or token.is_space:
            continue

        # preserva entidades importantes
        if token.ent_type_ in {"PER", "ORG", "LOC", "MISC"}:
            tokens.append(token.text.lower())
        else:
            # evita tokens muito curtos (ruído)
            if len(token.lemma_) >= 3:
                tokens.append(token.lemma_.lower())

    return " ".join(tokens)
