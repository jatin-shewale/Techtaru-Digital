# Chat with your PDF

Upload a text-based PDF, ask questions about it, and get answers grounded in the document with page references. This project uses a retrieval-augmented generation (RAG) pipeline: embeddings and search run locally, while Groq generates the final answer from retrieved passages.

## Features

- Extracts text page by page from PDFs.
- Splits text into overlapping chunks and keeps each chunk's page number.
- Creates local embeddings with Sentence Transformers and stores them in an in-memory ChromaDB collection.
- Retrieves the most relevant passages using cosine similarity.
- Answers with Groq using retrieved text only, and displays the source pages.
- Refuses to call the language model when no passage meets the relevance threshold.

## Requirements

- Python 3.10 or newer
- A Groq API key
- Internet access the first time the local Sentence Transformers model is downloaded; later runs can use the cached model.

## Setup on Windows PowerShell

Run these commands from the repository root:

```powershell
cd Task1
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `Task1/.env` and replace the placeholder with your Groq key:

```dotenv
GROQ_API_KEY=gsk-your-key-here
```

Start the app from the `Task1` directory:

```powershell
python -m streamlit run app.py
```

Streamlit prints the local URL in the terminal. Open it in a browser, upload a PDF from the sidebar, wait for indexing to finish, then ask a question in the chat box.

## Configuration

These optional values can be set in `.env`:

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | Required | Authenticates answer-generation requests to Groq. |
| `CHAT_MODEL` | `llama-3.3-70b-versatile` | Groq chat model used to generate answers. |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local Sentence Transformers model used for document and question embeddings. |
| `MIN_SIMILARITY` | `0.25` | Minimum cosine similarity (0 to 1) for a passage to be included as context. Raise it to be more selective; lower it to retrieve more broadly. |

Keep `.env` private. It is listed in `.gitignore`; never commit API keys.

## How it works

### 1. Read the PDF

`pypdf` extracts text from each page. Whitespace is normalised, and pages without extractable text are skipped. A PDF with no extractable text is rejected with an OCR hint; scanned pages are not OCRed by this app.

### 2. Split text into chunks

Each page is divided into chunks of about 800 characters with a 150-character overlap. Keeping page numbers in chunk metadata lets the UI show which page supplied each passage.

### 3. Embed and index locally

Sentence Transformers embeds the chunks locally. The vectors are normalised, then stored with their text and page metadata in an ephemeral ChromaDB collection configured for cosine distance. The collection exists only for the current app session; upload the PDF again after restarting the app.

### 4. Retrieve relevant passages

The question is embedded locally, then ChromaDB returns up to four closest chunks. Chroma's cosine distance is converted to similarity with `similarity = 1 - distance`. Chunks below `MIN_SIMILARITY` are discarded.

### 5. Generate a grounded answer

If no chunks pass the threshold, the app returns a fixed “couldn't find that” response without contacting Groq. Otherwise, it sends the question and retrieved passages to the configured Groq chat model. The prompt asks the model to use only that context, cite pages, and abstain when the answer is missing.

Embeddings stay on the machine. Retrieved passage text and the question are sent to Groq for answer generation.

## Limits and error handling

- Questions must contain text and be no longer than 2,000 characters.
- PDFs must contain a text layer. Scanned-image OCR is not included.
- Groq rate limits, timeouts, and connection errors are retried up to five times with exponential backoff.
- The vector collection is in memory and is not persisted between app sessions.
- `GROQ_API_KEY` must be present in `.env` before the app starts.

## Project files

| File | Purpose |
|---|---|
| `app.py` | Streamlit chat interface, PDF upload and source display. |
| `rag_core.py` | PDF extraction, chunking, local embedding, retrieval, and Groq answer generation. |
| `requirements.txt` | Python dependencies for this project. |
| `.env.example` | Environment variable template. |

## Troubleshooting

**`ModuleNotFoundError` when starting the app**: activate the virtual environment created in `Task1`, then run `python -m pip install -r requirements.txt` from the `Task1` directory.

**`GROQ_API_KEY is not set`**: check that `Task1/.env` exists and contains `GROQ_API_KEY` with a valid key. Restart Streamlit after editing it.

**No extractable text found**: the PDF is likely scanned. OCR the PDF first, then upload the searchable version.

**Answers say no relevant content was found**: check that the document contains the requested information. You can also lower `MIN_SIMILARITY` in `.env` and restart the app.
