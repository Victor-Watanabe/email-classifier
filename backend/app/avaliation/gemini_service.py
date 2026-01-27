import google.generativeai as genai
from app.config import GEMINI_API_KEY

# Configura a API com a key
genai.configure(api_key=GEMINI_API_KEY)

# Modelo base
model = genai.GenerativeModel("gemini-2.5-flash")


def test_gemini():
    prompt = """
Você é um sistema automatizado responsável por analisar e classificar
emails recebidos por uma grande empresa do setor financeiro.

Contexto:
A empresa recebe um alto volume diário de emails, incluindo solicitações
de status de processos, envio de documentos e mensagens improdutivas.
O objetivo é automatizar a triagem, liberando tempo da equipe humana.

Tarefa:
Analise o email abaixo e:

1. Classifique o conteúdo como PRODUTIVO ou IMPRODUTIVO.
2. Sugira uma resposta automática adequada ao tipo identificado.

Critérios de Classificação:

PRODUTIVO:
- Solicitações de andamento ou status de processos
- Envio ou solicitação de documentos
- Dúvidas sobre serviços, contratos ou operações
- Mensagens que exigem ação ou resposta da equipe

IMPRODUTIVO:
- Mensagens de felicitação ou cortesia (ex: "feliz natal", "bom dia")
- Agradecimentos sem nova solicitação
- Conteúdo genérico, irrelevante ou sem demanda clara

Formato de Resposta:
Responda EXCLUSIVAMENTE no seguinte formato JSON:

{
  "classification": "PRODUTIVO ou IMPRODUTIVO",
  "suggested_reply": "Resposta automática profissional e objetiva",
  "justification": "Breve explicação da decisão"
}


Texto:
"Olá, Boa tarde a todos, gostaria de começar desejando o feliz natal para toda a equipe aqui presente
espero que todos estejam bem e gostaria de tirar algumas dúvidas sobre o novo sistema implantado recentemente, 
pos está acusando alguns erros agora pela tarde. 

 Obrigado e bom final de semana a todos!"
"""

    response = model.generate_content(prompt)

    return response.text
