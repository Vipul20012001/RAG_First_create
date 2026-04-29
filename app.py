from dotenv import load_dotenv
import os
import pickle
from pathlib import Path
import numpy as np
import google.genai as genai
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
load_dotenv()

DATA_DIR = Path("data")
INDEX_PATH = Path("vector_store.pkl")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def read_pdf(file_bytes):
    reader = PdfReader(file_bytes)
    text = []
    for page in reader.pages:
        text.append(page.extract_text() or "")
    return "\n\n".join(text)


def read_text_file(file_bytes):
    return file_bytes.read().decode("utf-8", errors="ignore")


def chunk_text(text, chunk_size=500, overlap=100):
    tokens = text.split()
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = " ".join(tokens[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


class SimpleVectorStore:
    def __init__(self, embeddings=None, metadatas=None):
        self.embeddings = np.array(embeddings) if embeddings is not None else np.zeros((0, 0), dtype=np.float32)
        self.metadatas = metadatas or []

    def add(self, texts, metadatas, model):
        new_embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        if self.embeddings.size == 0:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        self.metadatas.extend(metadatas)

    def search(self, query, model, top_k=4):
        if self.embeddings.size == 0:
            return []
        query_embedding = model.encode([query], convert_to_numpy=True, show_progress_bar=False)
        scores = cosine_similarity(query_embedding, self.embeddings)[0]
        indexes = np.argsort(scores)[::-1][:top_k]
        return [
            {
                "score": float(scores[i]),
                "text": self.metadatas[i]["text"],
                "source": self.metadatas[i]["source"],
            }
            for i in indexes
        ]

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"embeddings": self.embeddings, "metadatas": self.metadatas}, f)

    @classmethod
    def load(cls, path):
        if not path.exists():
            return cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(embeddings=data.get("embeddings"), metadatas=data.get("metadatas"))


def build_metadata(source, text):
    return {"source": source, "text": text}


def ingest_uploaded_files(files, store, model):
    imported_count = 0
    for uploaded_file in files:
        name = uploaded_file.name
        if name.lower().endswith(".pdf"):
            text = read_pdf(uploaded_file)
        elif name.lower().endswith(".txt"):
            text = read_text_file(uploaded_file)
        else:
            st.warning(f"Skipping unsupported file type: {name}")
            continue

        chunks = chunk_text(text)
        store.add(chunks, [build_metadata(name, chunk) for chunk in chunks], model)
        imported_count += len(chunks)

    return imported_count


def format_sources(results):
    output = []
    for idx, item in enumerate(results, 1):
        output.append(f"### Result {idx} (score: {item['score']:.3f})\nSource: {item['source']}\n\n{item['text']}")
    return "\n\n".join(output)


def generate_answer_with_gemini(query, contexts, gemini_api_key):

# Configure with your API key
# List all models and print their names
    
    # key = gemini_api_key
    # if not key:
    #     return None
    prompt = (
        "Use the provided context to answer the question concisely. "
        "If the answer cannot be found in the context, say you could not find a confident answer.\n\n"
        f"Context:\n{contexts}\n\nQuestion: {query}\nAnswer:"
    )
    try:
        client = genai.Client()
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.4,
                max_output_tokens=80,
            ),
        )
        return response.text.strip()
    except Exception as exc:
        st.error(f"Gemini request failed: {exc}")
        return None


def main():
    st.set_page_config(page_title="Streamlit RAG System", layout="wide")
    st.title("📚 Streamlit RAG System")
    st.write(
        "Upload documents, build a local semantic index, and ask questions with retrieval-augmented responses."
    )

    model = load_embedding_model()
    store = SimpleVectorStore.load(INDEX_PATH)

    with st.sidebar:
        st.header("Index Management")
        st.write("Upload `.txt` or `.pdf` files and persist the semantic index locally.")
        uploaded_files = st.file_uploader("Upload documents", type=["txt", "pdf"], accept_multiple_files=True)
        ingest_button = st.button("Ingest uploaded files")
        if ingest_button and uploaded_files:
            with st.spinner("Embedding documents..."):
                count = ingest_uploaded_files(uploaded_files, store, model)
                store.save(INDEX_PATH)
            st.success(f"Ingested {count} text chunks into vector store.")
        elif ingest_button and not uploaded_files:
            st.warning("Select files to upload before ingesting.")

        if INDEX_PATH.exists():
            if st.button("Clear stored index"):
                INDEX_PATH.unlink(missing_ok=True)
                store = SimpleVectorStore()
                st.success("Local vector store cleared.")

        st.markdown("---")
        st.header("Gemini Settings")
        gemini_api_key = st.text_input("Gemini API Key", type="password", placeholder="AIza...", help="Optional. Needed for answer generation.")
        st.caption("If not provided, only retrieved document chunks will be shown.")

    st.header("Ask a question")
    query = st.text_input("Enter your question")
    top_k = st.slider("Number of retrieved chunks", min_value=1, max_value=6, value=4)

    if st.button("Search"):
        if store.embeddings.size == 0:
            st.warning("No vector store available. Ingest documents first.")
            return
        if not query:
            st.warning("Enter a question before searching.")
            return

        with st.spinner("Retrieving relevant context..."):
            results = store.search(query, model, top_k=top_k)
            contexts = "\n\n".join([f"Source: {item['source']}\n{item['text']}" for item in results])

        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Answer")
            answer = generate_answer_with_gemini(query, contexts, gemini_api_key)
            if answer:
                st.write(answer)
            else:
                st.info("Gemini API key is not configured or answer generation failed. Showing retrieved context below.")

        with col2:
            st.subheader("Retrieved chunks")
            for idx, item in enumerate(results, 1):
                st.markdown(f"**{idx}. Source:** {item['source']}  \n**Score:** {item['score']:.3f}")
                st.write(item['text'])

    st.markdown("---")
    st.subheader("Usage notes")
    st.markdown(
        "- Upload text or PDF documents in the sidebar.\n"
        "- Ingest creates a local semantic index in `vector_store.pkl`.\n"
        "- Ask a question and retrieve the most relevant chunks.\n"
        "- If you provide a Gemini API key, Streamlit will use it to generate an answer from the retrieved contexts."
    )


if __name__ == "__main__":
    main()
