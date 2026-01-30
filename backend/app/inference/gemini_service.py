import google.generativeai as genai
import json
from app.config import GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash")


def query_gemini(email_text: str) -> dict:
    prompt = f"""
Classifique o email abaixo como PRODUTIVO ou IMPRODUTIVO
e sugira uma resposta automática profissional.

Responda SOMENTE em JSON válido neste formato:

{{
  "classification": "PRODUTIVO ou IMPRODUTIVO",
  "suggested_reply": "Resposta objetiva",
  "justification": "Breve explicação"
}}

Email:
\"\"\"{email_text}\"\"\"
"""

    response = model.generate_content(prompt)
    text = response.text.strip()

    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:-1])

    try:
        parsed = json.loads(text)
        parsed["classification"] = parsed["classification"].upper()
        return parsed
    except Exception:
        return {
            "classification": "IMPRODUTIVO",
            "suggested_reply": (
                "Olá! Recebemos sua mensagem e ela será analisada em breve."
            ),
            "justification": "Falha ao interpretar resposta do Gemini."
        }
