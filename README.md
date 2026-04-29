# Streamlit RAG System

A simple Retrieval-Augmented Generation (RAG) application built with Streamlit.

## Features

- Upload `.txt` or `.pdf` documents
- Create a local semantic vector store
- Retrieve relevant document chunks for a query
- Generate answers using OpenAI GPT when an API key is provided

## Setup
1. Create a .env file with the name the code as follows
   ```python
   GEMINI_API_KEY =
   ```

3. Create a Python environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

4. Run the app:

```powershell
streamlit run app.py
```

## Usage

- Upload files using the sidebar.
- Click `Ingest uploaded files` to build the index.
- Enter a question and click `Search`.
- Optionally paste your `OPENAI_API_KEY` in the sidebar for answer generation.

## Notes

- The vector store is persisted to `vector_store.pkl`.
- If no OpenAI key is provided, the app will still show retrieved chunks.
