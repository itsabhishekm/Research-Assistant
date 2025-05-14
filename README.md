# Research Assistant

Research Assistant is a local AI assistant that helps users understand academic papers by asking natural language questions. This assistant uses a Retrieval-Augmented Generation (RAG) pipeline powered by GPT-J and vector search to provide grounded, context-aware answers from PDF research papers.

---

##  System Architecture

### Frontend

* I have used **Streamlit** to build the user interface.
* Users can upload a PDF, view the document, and interact with the chatbot in a responsive two-column layout.
* `streamlit-pdf-viewer` is used to render the PDF directly inside the web app.
* Chat input field lets users ask queries related to the paper.

###  Backend

* **Python-based RAG pipeline** handles PDF parsing, text preprocessing, embedding, retrieval, and response generation.
* When a user submits a question:

  1. The system retrieves the most relevant text chunks from the PDF using vector similarity search.
  2. A prompt combining the context and the question is generated.
  3. The prompt is passed to a language model to generate an answer.

###  Vector Storage

* **ChromaDB** is used as a lightweight local vector database.
* All extracted text chunks are embedded using `SentenceTransformer` and stored in ChromaDB.
* At query time, ChromaDB returns the top-k semantically relevant chunks based on the input question.

### Language Model

* **EleutherAI/gpt-j-6B** is used as the primary LLM for generating answers.
* It supports long context (up to 2048 tokens), making it suitable for dense academic content.
* Prompts are pre-truncated if necessary to stay within the model’s token limits.
* For lower-resource environments, models like `flan-t5-base` or `gpt2-medium` can be used instead.

### Chunking and Embedding

* PDFs are parsed using `pdfplumber`, and the extracted text is cleaned.
* The text is split into overlapping chunks to maintain context coherence.
* Each chunk is embedded using a pre-trained `SentenceTransformer`.
* These embeddings are stored alongside the original text for fast retrieval.

---

## Directory Structure

```
Research-Assistant/
├── app/
│   ├── main.py           # Streamlit app entry point
│   ├── parser.py         # PDF parsing and text cleaning
│   ├── pipeline.py       # Chunking, embedding, retrieval logic
│   ├── model.py          # GPT-J 
│   ├── config.py         # Declaring Constants for chunk size, model name, etc.
|   ├── ui.py             # UI designed using streamlit
│
├── requirements.txt     # Python dependencies
├── README.md            # You're here yay!!
├── demo/                # screenshots of the demo
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/itsabhishekm/Research-Assistant.git
cd Research-Assistant
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
streamlit run main.py
```

>  Note: `gpt-j-6B` requires a GPU with at least 16GB VRAM. If you're on CPU, consider using `flan-t5-base` or `gpt2-medium` instead.

---

## How It Works

1. **PDF Upload**: The user uploads a research paper.
2. **Text Extraction**: The PDF is parsed into clean text using `pdfplumber`.
3. **Chunking & Embedding**: The paper is split into overlapping text chunks and embedded using `SentenceTransformer`.
4. **Storage**: Chunks are stored in ChromaDB for semantic search.
5. **Question Input**: The user types a question.
6. **Retrieval**: Top-k relevant chunks are retrieved from ChromaDB.
7. **Prompting**: A prompt is constructed as:

   ```
   Context:
   <retrieved chunks>

   Question: <your question>
   Answer:
   ```
8. **Generation**: The prompt is passed to GPT-J to generate an answer.
9. **Display**: The answer is shown on the right while the PDF remains on view.

---

## Requirements

* Python 3.8+
* torch
* transformers
* chromadb
* sentence-transformers
* pdfplumber
* streamlit
* streamlit-pdf-viewer

> Do Add `accelerate` and `bitsandbytes` if using quantized models or optimizing memory.

---

## 📸 Screenshots

![image](https://github.com/user-attachments/assets/5a6b93b3-c5ed-433b-907a-fba6ce9687cc)


---

## Future scope

The current version is the Phase 1 of the project and in the phase 2 I am working on to builing a actual agent tailored to the specific use case and also palnned to shift my frontend to reac.js and also add some 3d elements using tree.js.

## Referrence

* Hugging Face Transformers (language models including GPT-J): https://huggingface.co/transformers
* GPT-J (EleutherAI) model card: https://huggingface.co/EleutherAI/gpt-j-6B
* ChromaDB (vector storage engine): https://www.trychroma.com or https://docs.trychroma.com
* Sentence Transformers (embeddings): https://www.sbert.net
* pdfplumber (PDF text extraction): https://github.com/jsvine/pdfplumber
* Streamlit (frontend framework): https://streamlit.io
* streamlit-pdf-viewer: https://pypi.org/project/streamlit-pdf-viewer
* Inspired by our team faced issues while researching new fields
