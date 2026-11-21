
import os
import re
import time

import streamlit as st
import numpy as np
import faiss

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from google import genai


# ==================================================
# 1. CONFIGURATION
# ==================================================

load_dotenv()

st.set_page_config(
    page_title="PDF Question Answering Chatbot",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF Question Answering Chatbot")

st.write(
    "Upload a PDF and ask questions about its content "
    "using Retrieval-Augmented Generation (RAG)."
)


# ==================================================
# 2. GEMINI API CONFIGURATION
# ==================================================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error(
        "Gemini API key not found. "
        "Please add GEMINI_API_KEY to your .env file."
    )
    st.stop()

client = genai.Client(api_key=api_key)

# Keep the model name that you tested in your environment.
GEMINI_MODEL = "gemini-3.8-flash"


# ==================================================
# 3. LOAD SENTENCE TRANSFORMER MODEL
# ==================================================

@st.cache_resource
def load_embedding_model():

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return model


with st.spinner("Loading embedding model..."):
    embedding_model = load_embedding_model()


# ==================================================
# 4. EXTRACT TEXT FROM PDF
# ==================================================

def extract_text_from_pdf(pdf_file):

    reader = PdfReader(pdf_file)

    extracted_text = ""

    for page_number, page in enumerate(reader.pages):

        page_text = page.extract_text()

        if page_text:

            extracted_text += (
                f"\n{page_text}\n"
            )

    return extracted_text, len(reader.pages)


# ==================================================
# 5. SPLIT TEXT INTO CHUNKS
# ==================================================

def split_text_into_chunks(
    text,
    chunk_size=250,
    overlap=50
):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk_words = words[start:end]

        chunk_text = " ".join(chunk_words)

        if chunk_text.strip():

            chunks.append(chunk_text)

        start += chunk_size - overlap

    return chunks


# ==================================================
# 6. CREATE EMBEDDINGS
# ==================================================

def create_embeddings(chunks):

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    embeddings = embeddings.astype(
        "float32"
    )

    # Normalize vectors for cosine similarity
    faiss.normalize_L2(embeddings)

    return embeddings


# ==================================================
# 7. CREATE FAISS VECTOR DATABASE
# ==================================================

def create_vector_database(embeddings):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(embeddings)

    return index


# ==================================================
# 8. TEXT NORMALIZATION
# ==================================================

def tokenize_text(text):

    return set(
        re.findall(
            r"[a-z0-9]+",
            text.lower()
        )
    )


# ==================================================
# 9. HYBRID RETRIEVAL
# ==================================================

def retrieve_relevant_chunks(
    question,
    index,
    chunks,
    top_k=5
):

    if not chunks:

        return []

    # ----------------------------------------------
    # A. SEMANTIC SEARCH USING FAISS
    # ----------------------------------------------

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(
        question_embedding
    )

    scores, indices = index.search(
        question_embedding,
        len(chunks)
    )

    # ----------------------------------------------
    # B. KEYWORD MATCHING
    # ----------------------------------------------

    stopwords = {
        "what", "is", "the", "a", "an",
        "and", "or", "why", "does", "do",
        "how", "to", "of", "in", "for",
        "it", "this", "that", "are", "was",
        "were", "explain", "describe", "tell",
        "me", "about", "can", "you", "with"
    }

    question_words = tokenize_text(
        question
    ) - stopwords

    ranked_chunks = []

    # ----------------------------------------------
    # C. COMBINE SEMANTIC AND KEYWORD SCORES
    # ----------------------------------------------

    for semantic_score, chunk_index in zip(
        scores[0],
        indices[0]
    ):

        if chunk_index == -1:
            continue

        chunk = chunks[chunk_index]

        chunk_words = tokenize_text(
            chunk
        )

        matched_words = (
            question_words.intersection(
                chunk_words
            )
        )

        if question_words:

            keyword_score = (
                len(matched_words)
                / len(question_words)
            )

        else:

            keyword_score = 0.0

        # Exact phrase bonus for multiword technical terms
        phrase_bonus = 0.0

        normalized_question = question.lower()
        normalized_chunk = chunk.lower()

        if len(question_words) >= 2:

            if normalized_question in normalized_chunk:

                phrase_bonus = 0.25

        # Combined relevance score
        final_score = (
            0.65 * float(semantic_score)
            + 0.35 * keyword_score
            + phrase_bonus
        )

        ranked_chunks.append(
            (
                final_score,
                float(semantic_score),
                keyword_score,
                chunk
            )
        )

    # ----------------------------------------------
    # D. SORT BY FINAL RELEVANCE SCORE
    # ----------------------------------------------

    ranked_chunks.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return ranked_chunks[:top_k]


# ==================================================
# 10. GENERATE ANSWER USING GEMINI
# ==================================================

def generate_answer(
    question,
    retrieved_chunks,
    max_retries=2
):

    if not retrieved_chunks:

        return (
            "No relevant information was retrieved "
            "from the uploaded PDF."
        )

    # Extract text from ranked results
    context_parts = []

    for item in retrieved_chunks:

        score, semantic_score, keyword_score, chunk = item

        context_parts.append(chunk)

    context = "\n\n".join(
        context_parts
    )

    prompt = f"""
You are an accurate PDF question-answering assistant.

Your task is to answer the user's question using
the information provided in the retrieved context.

IMPORTANT RULES:

1. Use the retrieved context as your source.
2. Do not invent or assume information.
3. If the answer is not present in the context,
   clearly say that the information was not found
   in the retrieved sections of the PDF.
4. Explain technical terms clearly.
5. Give a structured answer when appropriate.

RETRIEVED CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""

    # ----------------------------------------------
    # GEMINI API WITH RETRIES
    # ----------------------------------------------

    for attempt in range(max_retries + 1):

        try:

            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )

            if response.text:

                return response.text

            return "Gemini returned an empty response."

        except Exception as e:

            error_message = str(e)

            # Temporary server overload
            if (
                "503" in error_message
                or "UNAVAILABLE" in error_message
            ):

                if attempt < max_retries:

                    time.sleep(
                        2 ** attempt
                    )

                    continue

                return (
                    "Gemini is temporarily experiencing "
                    "high demand. Please try again later. "
                    "Your PDF retrieval has already completed."
                )

            # API rate limit
            if (
                "429" in error_message
                or "RESOURCE_EXHAUSTED" in error_message
            ):

                return (
                    "Gemini API rate limit reached. "
                    "Please wait and try again."
                )

            # Model unavailable or invalid
            if (
                "404" in error_message
                or "NOT_FOUND" in error_message
            ):

                return (
                    "Gemini model was not found or is "
                    "not available to your API key. "
                    "Check the model name in Google AI Studio."
                )

            # API key or permission issue
            if (
                "401" in error_message
                or "403" in error_message
                or "PERMISSION_DENIED" in error_message
            ):

                return (
                    "Gemini authentication or permission "
                    "failed. Check your API key and access."
                )

            return (
                f"Gemini API error: {error_message}"
            )

    return "Unable to generate an answer."


# ==================================================
# 11. SESSION STATE
# ==================================================

if "chunks" not in st.session_state:

    st.session_state.chunks = []

if "index" not in st.session_state:

    st.session_state.index = None

if "processed" not in st.session_state:

    st.session_state.processed = False

if "file_name" not in st.session_state:

    st.session_state.file_name = None

if "total_pages" not in st.session_state:

    st.session_state.total_pages = 0


# ==================================================
# 12. PDF UPLOAD
# ==================================================

uploaded_file = st.file_uploader(
    "Upload your PDF document",
    type=["pdf"]
)


# ==================================================
# 13. PROCESS PDF
# ==================================================

if uploaded_file is not None:

    if st.button(
        "Process PDF",
        type="primary"
    ):

        try:

            # Reset previous document
            st.session_state.chunks = []
            st.session_state.index = None
            st.session_state.processed = False
            st.session_state.file_name = None
            st.session_state.total_pages = 0

            with st.spinner(
                "Extracting and processing PDF..."
            ):

                # Extract PDF text
                text, total_pages = extract_text_from_pdf(
                    uploaded_file
                )

                if not text.strip():

                    st.error(
                        "No text extracted. "
                        "The PDF may be scanned or image-based."
                    )

                    st.stop()

                # Create smaller chunks
                chunks = split_text_into_chunks(
                    text,
                    chunk_size=250,
                    overlap=50
                )

                if not chunks:

                    st.error(
                        "No text chunks could be created."
                    )

                    st.stop()

                # Generate embeddings
                embeddings = create_embeddings(
                    chunks
                )

                # Build FAISS index
                index = create_vector_database(
                    embeddings
                )

                # Save processed information
                st.session_state.chunks = chunks

                st.session_state.index = index

                st.session_state.processed = True

                st.session_state.file_name = uploaded_file.name

                st.session_state.total_pages = total_pages

            st.success(
                "PDF processed successfully!"
            )

            st.write(
                f"Total pages: {total_pages}"
            )

            st.write(
                f"Total chunks created: {len(chunks)}"
            )

        except Exception as e:

            st.error(
                f"PDF processing failed: {e}"
            )


# ==================================================
# 14. ASK QUESTIONS
# ==================================================

if st.session_state.processed:

    st.divider()

    st.subheader(
        "💬 Ask Questions About Your PDF"
    )

    st.caption(
        f"Loaded document: {st.session_state.file_name}"
    )

    question = st.text_input(
        "Enter your question:"
    )

    if st.button(
        "Get Answer",
        type="primary"
    ):

        if question.strip():

            with st.spinner(
                "Searching the PDF and generating answer..."
            ):

                # Hybrid retrieval
                retrieved_chunks = retrieve_relevant_chunks(
                    question,
                    st.session_state.index,
                    st.session_state.chunks,
                    top_k=5
                )

                # Generate Gemini answer
                answer = generate_answer(
                    question,
                    retrieved_chunks
                )

            st.subheader("🤖 Answer")

            st.write(answer)

            # --------------------------------------
            # SHOW RETRIEVED CONTEXT
            # --------------------------------------

            with st.expander(
                "View Retrieved Context and Scores"
            ):

                for i, item in enumerate(
                    retrieved_chunks,
                    start=1
                ):

                    (
                        final_score,
                        semantic_score,
                        keyword_score,
                        chunk
                    ) = item

                    st.markdown(
                        f"### Retrieved Chunk {i}"
                    )

                    st.caption(
                        f"Combined score: {final_score:.3f} | "
                        f"Semantic score: {semantic_score:.3f} | "
                        f"Keyword score: {keyword_score:.3f}"
                    )

                    st.write(chunk)

                    st.divider()

        else:

            st.warning(
                "Please enter a question."
            )


# ==================================================
# 15. FOOTER
# ==================================================

st.divider()

st.caption(
    "Built using Python, Streamlit, PyPDF, "
    "Sentence Transformers, FAISS and Gemini API."
)
