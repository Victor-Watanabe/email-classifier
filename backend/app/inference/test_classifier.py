from app.inference import classifier

def test_text_classification():
    texto_teste = '''"Assunto: Regularização de divergência em fatura – Contrato nº 45872

Boa tarde,

Espero que estejam bem.

Identificamos uma divergência nos valores apresentados na fatura referente ao contrato nº 45872, com vencimento em 30/01/2026. O valor faturado não corresponde ao que foi previamente acordado na proposta comercial aprovada em 15/12/2025.

Em anexo, seguem:

a cópia da proposta comercial assinada;

a fatura recebida;

o demonstrativo de cálculo que evidencia a diferença de valores.

Solicito, por gentileza, a verificação do ocorrido e o envio de uma fatura corrigida ou esclarecimentos formais sobre o ajuste realizado, para que possamos prosseguir com o pagamento dentro do prazo.

Aproveito para desejar um excelente início de ano a toda a equipe.

Fico no aguardo de um retorno até sexta-feira, para evitar impactos no nosso fluxo financeiro.

Atenciosamente,
Carlos Henrique Almeida
Analista Financeiro
Departamento Financeiro
Tel: (11) 9xxxx-xxxx"'''
    print("\n=== TESTE TEXTO ===")
    resultado = classifier.classify_email(texto_teste)
    print("Texto:", texto_teste)
    print("Resultado:", resultado)

if __name__ == "__main__":
    test_text_classification()
