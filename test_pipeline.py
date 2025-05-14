from pipeline import RAGPipeline

text = """
We propose a new approach to neural reasoning with transformer-based models.
Our experiments demonstrate state-of-the-art performance on several NLP benchmarks.
The method is simple, effective, and efficient for low-resource environments.
"""

# Createing an instance
rag = RAGPipeline(text)

query = "What is the main contribution?"
chunks = rag.retrieve_relevant_chunks(query)

# results
for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i+1} ---\n{chunk}\n")
