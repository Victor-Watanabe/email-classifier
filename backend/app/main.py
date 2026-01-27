from fastapi import FastAPI
from app.config import GEMINI_API_KEY
from app.gemini_service import test_gemini


app = FastAPI()


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "gemini_key_loaded": GEMINI_API_KEY is not None
    }

@app.get("/test-gemini")
def test_gemini_route():
    return {
        "gemini_response": test_gemini()
    }