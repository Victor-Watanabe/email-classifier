from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.inference import classifier
from app.utils.extract_text_from_pdf import extract_text_from_pdf
from app.utils.extract_text_from_txt import extract_text_from_txt

app = FastAPI(title="Email Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Quais domínios podem acessar.
    allow_credentials=True,
    allow_methods=["*"], # Metodos aceito POST, GET, ETC.
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/classify/text")
async def classify_text_endpoint(text: str = Form(...)):
    if not text.strip():
        return JSONResponse(
            status_code=400,
            content={"error": "Informe o texto para classificação."}
        )

    result = classifier.classify_email(text)
    return {"result": result}

@app.post("/classify/file")
async def classify_file_endpoint(file: UploadFile = File(...)):
    if not file:
        return JSONResponse(
            status_code=400,
            content={"error": "Informe um arquivo PDF ou TXT."}
        )

    try:
        if file.content_type == "application/pdf":
            text = extract_text_from_pdf(file)

        elif file.content_type == "text/plain":
            text = extract_text_from_txt(file)

        else:
            return JSONResponse(
                status_code=400,
                content={"error": "Formato de arquivo não suportado."}
            )

        if not text.strip():
            return JSONResponse(
                status_code=400,
                content={"error": "O arquivo não contém texto legível."}
            )

        result = classifier.classify_email(text)
        return {"result": result}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
