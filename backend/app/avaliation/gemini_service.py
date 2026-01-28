# backend/app/avaliation/gemini_service.py

import google.generativeai as genai
from app.config import GEMINI_API_KEY
import json

# Configura a API com a key
genai.configure(api_key=GEMINI_API_KEY)

# Modelo base
model = genai.GenerativeModel("gemini-2.5-flash")


def query_gemini(email_text: str):
    """
    Envia um email para a Gemini API para classificação e sugestão de resposta.

    Parâmetros:
        email_text (str): texto do email recebido

    Retorno:
        dict: {
            "classification": "PRODUTIVO ou IMPRODUTIVO",
            "suggested_reply": str,
            "justification": str
        }
    """
    prompt = f"""
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

{{
  "classification": "PRODUTIVO ou IMPRODUTIVO",
  "suggested_reply": "Resposta automática profissional e objetiva",
  "justification": "Breve explicação da decisão"
}}

Texto:
"{email_text}"
"""

    # Envia prompt para o modelo
    response = model.generate_content(prompt)
    text = response.text.strip()

    # Remove bloco de crases se existir (```json ... ```)
    if text.startswith("```") and text.endswith("```"):
        text_lines = text.split("\n")
        # Remove a primeira e última linha do bloco de crases
        text = "\n".join(text_lines[1:-1])

    # Converte texto em dict
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback caso não seja JSON válido
        return {
            "classification": "Desconhecido",
            "suggested_reply": response.text,
            "justification": "Resposta não padronizada pelo modelo Gemini"
        }
