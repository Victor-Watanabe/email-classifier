from app.inference import classifier

def test_text_classification():
    texto_teste = "Boa Tarde, desejo feliz natal a toda equipe, boas festas."
    print("\n=== TESTE TEXTO ===")
    resultado = classifier.classify_email(texto_teste)
    print("Texto:", texto_teste)
    print("Resultado:", resultado)

if __name__ == "__main__":
    test_text_classification()
