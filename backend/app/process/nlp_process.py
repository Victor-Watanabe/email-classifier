import spacy

nlp = spacy.load("pt_core_news_sm")

def preprocess_text(text: str) -> str:
    
# converte para minúsculo e cria objeto spaCy
    doc = nlp(text.lower()) 

# lista que vai armazenar palavras relevantes
    tokens = []  

# Passa de palavra em palavra pelo documento/texto
    for token in doc:
       # ignora stop words
        if token.is_stop:  
            continue

       # ignora pontuação
        if token.is_punct:  
            continue

      # adiciona a forma base da palavra
        tokens.append(token.lemma_)  

    # retorna texto limpo
    return " ".join(tokens)  
