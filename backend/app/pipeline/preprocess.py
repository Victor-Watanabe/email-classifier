import spacy
import re
import logging

logger = logging.getLogger(__name__)

# ============================
# Carregamento seguro do spaCy
# ============================
try:
    # Carrega modelo completo se disponível
    # Desabilita componentes pesados para melhorar performance
    nlp = spacy.load(
        "pt_core_news_sm",
        disable=["parser", "ner"]
    )
except OSError:
    # Fallback para ambiente sem modelo (ex: Render)
    logger.warning(
        "Modelo 'pt_core_news_sm' não encontrado. "
        "Usando spaCy blank (pt) como fallback."
    )
    nlp = spacy.blank("pt")

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
    """
    Remove frases sociais previsíveis que não agregam
    informação semântica ao classificador.
    """
    text = text.lower()

    for phrase in SOCIAL_PHRASES:
        # Remove frase inteira respeitando limites de palavra
        text = re.sub(
            rf"\b{re.escape(phrase)}\b",
            "",
            text
        )

    return text

# ============================
# Pré-processamento principal
# ============================
def preprocess_text(text: str) -> str:
    """
    Pré-processa texto para o modelo de classificação:

    Etapas:
    - remove cordialidades sociais previsíveis
    - converte para minúsculo
    - remove stopwords, pontuação e espaços
    - aplica lematização
    - preserva tokens relevantes
    - ignora ruídos e tokens muito curtos
    """

    if not text or not text.strip():
        return ""

    # 1️⃣ Remove ruído social ANTES do NLP
    text = remove_social_phrases(text)

    # 2️⃣ Processa com spaCy
    doc = nlp(text)

    tokens = []

    for token in doc:
        # Ignora stopwords, pontuação e espaços
        if token.is_stop or token.is_punct or token.is_space:
            continue

        # Preserva entidades relevantes quando disponíveis
        if token.ent_type_ in {"PER", "ORG", "LOC", "MISC"}:
            tokens.append(token.text.lower())
        else:
            # Evita tokens muito curtos (ruído)
            lemma = token.lemma_.lower()
            if len(lemma) >= 3:
                tokens.append(lemma)

    return " ".join(tokens)
