from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.inference import classifier
from app.utils.extract_text_from_pdf import extract_text_from_pdf

app = FastAPI(title="Email Classifier API")

# Configuração CORS
# Permitir todos os domínios por enquanto (pode restringir depois)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todos os domínios
    allow_methods=["*"],  # Permite todos os métodos HTTP (GET, POST, etc.)
    allow_headers=["*"],  # Permite todos os headers
)

# Health check básico
@app.get("/health")
def health_check():
    return {"status": "ok"}

# Endpoint para classificar TEXTO
@app.post("/classify/text")
async def classify_text_endpoint(text: str = Form(...)):
    if not text.strip():
        return JSONResponse(
            status_code=400,
            content={"error": "Informe o texto para classificação."}
        )

    try:
        # Chama o classifier para processar o texto
        result = classifier.classify_email(text)
        return {"result": result}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

# Endpoint para classificar PDF
@app.post("/classify/pdf")
async def classify_pdf_endpoint(file: UploadFile = File(...)):
    if not file:
        return JSONResponse(
            status_code=400,
            content={"error": "Informe um arquivo PDF para classificação."}
        )

    try:
        # Extrai texto do PDF
        pdf_text = extract_text_from_pdf(file)
        if not pdf_text.strip():
            return JSONResponse(
                status_code=400,
                content={"error": "O PDF não contém texto legível."}
            )

        # Chama o classifier para processar o texto extraído
        result = classifier.classify_email(pdf_text)
        return {"result": result}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
