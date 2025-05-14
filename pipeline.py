from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from parser import split_into_chunks, clean_text
from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBED_MODEL_NAME
from model import LocalGPTModel

class RAGPipeline:
    def __init__(self, raw_text: str):
        self.text = clean_text(raw_text)
        self.chunks = split_into_chunks(self.text, CHUNK_SIZE, CHUNK_OVERLAP)

        # Initialize embedding model
        self.emb_model = SentenceTransformer(EMBED_MODEL_NAME)

        # Seting up the Chroma vector store
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection("research_chunks")

        self.embed_chunks()

    def embed_chunks(self):
        embeddings = self.emb_model.encode(self.chunks, show_progress_bar=True)

        for i, chunk in enumerate(self.chunks):
            self.collection.add(
                documents=[chunk],
                embeddings=[embeddings[i]],
                ids=[f"chunk_{i}"]
            )

    def retrieve_relevant_chunks(self, query, k=5):
        results = self.collection.query(query_texts=[query], n_results=k, include=["documents", "distances"])
        documents = results["documents"][0]
        distances = results["distances"][0]
        return documents, distances



#llm = LocalT5Model()
#chunks = RAGPipeline.retrieve_relevant_chunks("What is the main contribution?")
#context = " ".join(chunks)

#answer = llm.generate_answer("What is the main contribution?", context)
#print(answer)