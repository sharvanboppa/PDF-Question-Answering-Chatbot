# PDF Question Answering Chatbot using RAG

## Project Overview
An AI-powered PDF Question Answering Chatbot that allows users to upload PDF documents and ask questions. It uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from documents and generate context-based answers using the Gemini API.

## Technologies Used
- Python
- Streamlit
- PyPDF
- Sentence Transformers
- FAISS
- Google Gemini API
- NumPy

## How It Works
1. Upload a PDF document through the Streamlit interface.
2. Extract text from the PDF using PyPDF.
3. Split the extracted text into smaller chunks.
4. Generate vector embeddings using Sentence Transformers.
5. Store embeddings in a FAISS vector index.
6. Retrieve relevant chunks using semantic and keyword-based search.
7. Pass the retrieved context to Gemini.
8. Generate and display an answer based on the retrieved information.

## Project Architecture

PDF Upload → Text Extraction → Chunking → Embeddings → FAISS Vector Database → Hybrid Retrieval → Gemini API → Answer Generation

## Installation and Execution

### 1. Clone the repository
```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate into the project
```bash
cd PDF_Question_Answering_Chatbot
```

### 3. Create a virtual environment
```bash
python -m venv venv
```

### 4. Activate the environment (Windows)
```bash
venv\Scripts\activate
```

### 5. Install dependencies
```bash
pip install -r requirements.txt
```

### 6. Configure the API key

Create a `.env` file in the project directory:

```env
GEMINI_API_KEY=your_gemini_api_key
```

### 7. Run the application
```bash
streamlit run app.py
```

## Key Features
- PDF document upload
- Automatic text extraction
- Text chunking with overlap
- Semantic search using vector embeddings
- FAISS-based similarity search
- Hybrid keyword and semantic retrieval
- Context-grounded AI responses
- Retrieved context and relevance scores

## Future Enhancements
- Support for multiple PDF documents
- Improved retrieval using reranking
- Chat history
- Source citations
- Enhanced user interface

## Author
Sharvan Boppa

AI / ML | Generative AI | Python