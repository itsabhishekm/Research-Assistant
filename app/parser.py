import pdfplumber
import re
from typing import List

def extract_text_from_pdf(file) -> str:
    try:
        full_text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                full_text += page.extract_text() + "\n"
        return full_text
    except Exception as e:
        raise RuntimeError(f"Failed to parse PDF: {e}")

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = text.replace("- ", "")
    return text.strip()

def split_into_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks
