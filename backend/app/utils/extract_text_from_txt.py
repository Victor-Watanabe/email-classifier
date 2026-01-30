from fastapi import UploadFile

def extract_text_from_txt(file: UploadFile) -> str:
    content = file.file.read()
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("latin-1")
