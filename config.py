import torch
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "EleutherAI/gpt-j-6B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


