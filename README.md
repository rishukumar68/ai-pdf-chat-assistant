# AI PDF Chat Assistant

A small GenAI project I built to explore how **Retrieval-Augmented Generation (RAG)** works in practice. Upload a text-based PDF, ask a question, and the app searches for relevant passages before asking Gemini to answer from that evidence.

The goal is to make answers easier to verify: the app displays the passages and PDF page numbers used for each response.

## What it can do

- Upload one PDF at a time.
- Extract text page by page with `pypdf`.
- Split the text into overlapping chunks and retain page numbers.
- Create embeddings with Gemini and index them in FAISS.
- Retrieve the most similar passages for a question.
- Ask Gemini to answer using the retrieved passages rather than general knowledge.
- View the retrieved passages alongside the answer.
- Keep chat history for the current Streamlit session.
- Show useful errors for common PDF and API problems.

> **Current limitations:** The app works with PDFs that contain selectable text. It does not include OCR for scanned documents, persistent storage, user authentication, or production-grade rate limiting. RAG can still retrieve the wrong passage or produce an incorrect answer, so check the cited source text for important information.

## How it works

```text
PDF upload
   |
   v
Extract text and page numbers (pypdf)
   |
   v
Split text into overlapping chunks
   |
   v
Create Gemini embeddings for each chunk
   |
   v
Store normalized vectors in FAISS
   |
   v
Embed the user's question and retrieve top-k chunks
   |
   v
Send the question and retrieved passages to Gemini
   |
   v
Show the answer and source passages
```

### Why these components?

- **pypdf** extracts selectable text and page numbers from a PDF.
- **Chunking** breaks long documents into smaller passages. Overlap helps retain context around chunk boundaries.
- **Gemini embeddings** turn text into vectors that can be compared by semantic similarity.
- **FAISS** searches the vectors. This project uses `IndexFlatIP`; because vectors are L2-normalized, inner product is equivalent to cosine similarity.
- **Gemini generation** writes a response from the retrieved evidence. The prompt asks the model to say when the evidence does not contain the answer.
- **Streamlit** provides the browser-based interface without needing a separate frontend.

## Project structure

```text
ai_pdf_chat_assistant/
├── app.py                  # Streamlit UI, upload flow, and chat history
├── rag_engine.py           # PDF extraction, chunking, embeddings, retrieval, and answers
├── requirements.txt        # Python dependencies
├── .env.example            # Example environment configuration (no real key)
├── .gitignore              # Excludes local secrets and environment files
├── README.md
└── .streamlit/
    └── config.toml         # Streamlit theme and upload settings
```

## Requirements

- Windows 10 or 11
- Python 3.12 (64-bit recommended)
- VS Code with the Python extension
- Internet access for Gemini API requests
- A Gemini API key

## Get a Gemini API key

1. Open [Google AI Studio – API keys](https://aistudio.google.com/apikey).
2. Sign in and create an API key for the Gemini API.
3. Store the key privately. Do not put it in Python source code, screenshots, or GitHub.

Model access and usage limits depend on the current Google AI Studio/API account settings. The model names below match the configuration used while developing this project; if Google changes availability, use a model currently supported by your account.

## Run locally on Windows

### 1. Clone the repository

Open PowerShell in VS Code and run:

```powershell
git clone https://github.com/rishukumar68/ai-pdf-chat-assistant.git
cd ai-pdf-chat-assistant
code .
```

If you already have the project folder, open that folder instead and skip the clone commands.

### 2. Create and activate a virtual environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, allow it for the current terminal only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

You can also use the virtual environment's Python executable directly without activating it.

### 3. Create your local `.env`

```powershell
Copy-Item .env.example .env
code .env
```

Set the values in `.env`:

```dotenv
GEMINI_API_KEY=your_real_api_key_here
GEMINI_MODEL=gemini-3.8-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
EMBED_BATCH_SIZE=64
```

Replace the API-key placeholder with your own key. Keep `.env` local; it should be excluded by `.gitignore`. Never commit the real key.

### 4. Start the app

```powershell
python -m streamlit run app.py
```

Open the local URL shown in the terminal (usually `http://localhost:8501`).

To check that the main packages import correctly:

```powershell
python -c "import streamlit, pypdf, faiss, numpy; from google import genai; print('Imports OK')"
```

## Using the app

1. Upload a text-based PDF from the sidebar.
2. Wait while the app extracts text, creates chunks and embeddings, and builds the FAISS index.
3. Ask a question in the chat box.
4. Read the answer and open **View retrieved source passages** to inspect the evidence.
5. Uploading another PDF replaces the current document and clears the chat.
6. Use **Clear chat** to clear conversation messages while keeping the current document indexed.

The current version indexes one PDF at a time. Uploaded documents and the vector index are held in the active Streamlit session and are not persisted across a server restart.

## Configuration

These settings can be placed in `.env`:

| Variable | Purpose | Default |
|---|---|---|
| `GEMINI_API_KEY` | Gemini API credential | Required |
| `GEMINI_MODEL` | Text-generation model | `gemini-3.8-flash` |
| `GEMINI_EMBEDDING_MODEL` | Embedding model | `gemini-embedding-001` |
| `CHUNK_SIZE` | Approximate chunk length in characters | `1000` |
| `CHUNK_OVERLAP` | Overlap between adjacent chunks, in characters | `150` |
| `EMBED_BATCH_SIZE` | Number of texts sent per embedding request batch | `64` |

These chunk settings use **characters, not tokens**.

## Troubleshooting

### `ModuleNotFoundError`

Activate `.venv`, then run:

```powershell
python -m pip install -r requirements.txt
```

Also check that VS Code is using the `.venv` interpreter.

### Gemini returns `401` or rejects the key

- Check that `.env` is beside `app.py` and is named `.env`, not `.env.txt`.
- Confirm `GEMINI_API_KEY` contains a valid key.
- Restart Streamlit after changing `.env`.
- Never post the key in a GitHub issue or screenshot.

### Gemini returns `404` / model not found

The configured model may not be available to your API account. Check the current model names available to your key and update `GEMINI_MODEL` or `GEMINI_EMBEDDING_MODEL` in `.env`. Restart Streamlit after making the change.

### Gemini returns `429` or a quota error

Check your API quota and usage limits. Try a smaller PDF or a lower `EMBED_BATCH_SIZE`, and wait before retrying if you have hit a rate limit.

### The PDF has no extractable text

The file may be scanned or image-only. This version does not perform OCR; use a searchable PDF or add an OCR step as a future improvement.

### FAISS installation fails on Windows

Upgrade packaging tools and retry:

```powershell
python -m pip install --upgrade pip setuptools wheel
pip install faiss-cpu
```

Use 64-bit Python. If installation still fails, try a clean virtual environment with a Python version supported by the available `faiss-cpu` wheel.

### Port 8501 is busy

```powershell
python -m streamlit run app.py --server.port 8502
```

## Security and deployment notes

- Keep `.env` out of Git. The repository should contain `.env.example` only, with a placeholder value.
- If a real key was ever committed, revoke it and create a replacement. Removing it from the latest file does not remove it from earlier Git history.
- The PDF text and questions are sent to Google's Gemini API for embeddings and generation. Do not upload confidential or regulated documents unless you have permission and have reviewed the relevant data-handling terms.
- Before making the app publicly accessible, consider upload/page limits, usage controls, rate limiting, error handling, and a clear data-retention policy.
- For Streamlit Community Cloud, configure `GEMINI_API_KEY`, `GEMINI_MODEL`, and `GEMINI_EMBEDDING_MODEL` in the app's Secrets settings. Do not upload `.env` to GitHub.

## Checks I use before calling it ready

- [ ] A question answered by the PDF returns a relevant response.
- [ ] Source passages and page numbers support the response.
- [ ] A question not covered by the PDF is handled without inventing an answer.
- [ ] Empty, encrypted, or scanned PDFs show an understandable error.
- [ ] Missing/invalid API credentials and quota errors are handled.
- [ ] `.env` and `.venv` are not present in the GitHub repository.
- [ ] The app works from a fresh environment using the README instructions.

## What I learned / interview explanation

This project helped me practice the main parts of a RAG workflow. The app extracts text from a PDF, splits it into overlapping chunks, creates embeddings for those chunks, and indexes the vectors in FAISS. When a user asks a question, it embeds the query and retrieves the most similar chunks. Those passages are sent to Gemini with the question, and the UI displays the answer together with the source text so the user can verify it.

### Questions to be ready for

- **What is RAG?** Retrieval-Augmented Generation retrieves relevant external content and gives it to a language model as context before generating an answer.
- **What is an embedding?** A numerical vector representation of text that can be used for semantic similarity search.
- **Why use FAISS?** It provides vector similarity search without requiring a separate database service for this small demo.
- **Why normalize vectors?** With L2-normalized vectors, inner product is equivalent to cosine similarity.
- **What does top-k mean?** The number of retrieved chunks sent to the model. A larger value can provide more context but may also add irrelevant text.
- **Does RAG eliminate hallucinations?** No. It can make answers more grounded, but retrieval and generation can still be wrong. The source passages should be checked.
- **What would you improve next?** OCR for scanned PDFs, hybrid keyword/vector retrieval, reranking, evaluation tests, persistent storage, authentication, and usage monitoring.

---

Built as a learning and portfolio project. Contributions and practical suggestions are welcome.
