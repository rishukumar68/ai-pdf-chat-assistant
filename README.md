# AI PDF Chat Assistant

A beginner-friendly Retrieval-Augmented Generation (RAG) app built with Python, Streamlit, pypdf, FAISS, and Google's Gemini API.

## Features

- Upload a text-based PDF and extract text page by page.
- Split extracted text into overlapping chunks while retaining page numbers.
- Generate document and query embeddings with Gemini.
- Store normalized embeddings in a FAISS inner-product index (cosine similarity).
- Retrieve relevant passages for a question and ask Gemini to answer using those passages only.
- Display source passages and page numbers next to the answer.
- Handle empty, unreadable, encrypted, or scanned/image-only PDFs and common API failures.
- Keep the API key in `.env`; `.gitignore` excludes it from Git.

> **Important:** This is an interview/demo project, not a guarantee of perfect answers. Validate important claims against the displayed source passages. Scanned PDFs need OCR, which is not included.

## 1. Requirements

- Windows 10/11
- Python 3.12 (tested design target; Python 3.12 is a good choice for this stack)
- VS Code with the Python extension
- Internet access for Gemini API requests

## 2. Get a Gemini API key

1. Open [Google AI Studio API keys](https://aistudio.google.com/apikey).
2. Sign in to your Google account.
3. Create an API key for the Gemini API.
4. Copy it and keep it private. Do not put it directly in Python source code or commit it to Git.
5. Usage limits, model access, and billing terms depend on Google's current account and API policies.

## 3. Set up in VS Code on Windows

Open PowerShell in VS Code and run these commands. If you already have a project folder, choose a new folder name or `cd` into it.

```powershell
mkdir ai_pdf_chat_assistant
cd ai_pdf_chat_assistant
code .
```

Put the project files from this repository/folder in the opened directory. Then create and activate a virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, you can run this once for the current terminal only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Or skip activation and use the virtual environment executable explicitly:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Create your `.env` file by copying the example:

```powershell
Copy-Item .env.example .env
```

Open `.env` in VS Code and replace `paste_your_gemini_api_key_here` with your real key. Do not add quotes unless your key itself requires them. Never share or commit `.env`.

Run the app:

```powershell
python -m streamlit run app.py
```

If using the explicit venv executable:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit prints a local URL, normally `http://localhost:8501`. Open it in your browser.

## 4. How to use it

1. Upload a PDF in the sidebar.
2. Wait for extraction, chunking, embedding, and FAISS indexing to finish.
3. Ask a question in the chat box.
4. Read the answer and expand **View retrieved source passages** to inspect evidence.
5. Upload a different PDF to replace the active document. The chat resets for the new document.
6. Use **Clear chat** to keep the document indexed but remove conversation history.

The current app indexes one PDF at a time. It does not persist uploaded PDFs or indexes after the Streamlit process restarts.

## 5. Project structure

```text
ai_pdf_chat_assistant/
├── app.py                  # Streamlit UI and session/chat state
├── rag_engine.py           # PDF extraction, chunking, embeddings, FAISS, answer generation
├── requirements.txt        # Python dependencies
├── .env.example            # Safe template for local configuration
├── .gitignore              # Excludes secrets and virtual environment
├── README.md               # Setup, usage, troubleshooting, interview notes
└── .streamlit/
    └── config.toml         # Theme and upload-size configuration
```

## 6. Architecture: the RAG pipeline

```text
PDF upload
   ↓
pypdf extracts text + original page numbers
   ↓
Chunking with overlap
   ↓
Gemini embedding model → numeric vectors
   ↓
FAISS IndexFlatIP (normalized vectors = cosine similarity)
   ↓
User question → query embedding
   ↓
FAISS retrieves top-k chunks
   ↓
Question + retrieved passages → Gemini generation model
   ↓
Answer with source labels + expandable source passages
```

### Why RAG?

A language model's internal knowledge is not the source of truth for a specific PDF. RAG first retrieves relevant text from the uploaded document, then supplies that evidence to the model. This makes responses more grounded and lets the UI show what text informed the answer. It reduces, but does not eliminate, hallucinations.

### Why FAISS?

FAISS efficiently searches vectors by similarity. This demo uses `IndexFlatIP`, an exact inner-product index. Because vectors are L2-normalized first, inner product corresponds to cosine similarity. For a small interview demo, it is simple and does not require a separate database server.

### Why chunks overlap?

Chunking creates manageable text passages for retrieval. Overlap helps preserve information when a sentence or idea crosses a chunk boundary. Here, sizes are character-based for simplicity, not model tokens.

## 7. Configuration

You can optionally add these to `.env`:

```dotenv
GEMINI_MODEL=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
EMBED_BATCH_SIZE=64
```

Use model IDs that are currently available to your API key. If a model is unavailable, update the relevant environment variable to a supported model from Google's documentation.

## 8. Common errors and fixes

### `ModuleNotFoundError`
Activate `.venv`, then run `python -m pip install -r requirements.txt`. Confirm VS Code selected `.venv` as the Python interpreter.

### `faiss` install fails on Windows
Update pip and try again:

```powershell
python -m pip install --upgrade pip setuptools wheel
pip install faiss-cpu
```

Use 64-bit Python 3.12. If your environment has no compatible wheel, create a fresh Python 3.11 or 3.12 virtual environment and retry. Do not install a package named `faiss`—the PyPI package is `faiss-cpu`, imported in code as `faiss`.

### `GEMINI_API_KEY` missing / rejected
Confirm the file is named `.env` (not `.env.txt`), is beside `app.py`, and contains `GEMINI_API_KEY=your-real-key`. Restart Streamlit after changing it. Make sure the key is active and has access to the Gemini API.

### Model not found / 404
The configured model name may not be available for your API key or endpoint. Update `GEMINI_MODEL` or `GEMINI_EMBEDDING_MODEL` in `.env` to a currently supported model. Check Google's official Gemini API model documentation.

### 429 / quota / rate limit
Check the API usage and rate limits in Google AI Studio. Wait before retrying, reduce document size, or lower `EMBED_BATCH_SIZE`. Embeddings are generated in batches to avoid sending every chunk in one request.

### PDF has no text
Some PDFs are scans or images. This project uses text extraction only and does not include OCR. Use a searchable PDF or add an OCR pipeline (for example, Tesseract) as a future enhancement.

### The answer says it cannot find something
This is intentional when the retrieved evidence does not contain enough information. Try a more specific question, check the source passages, or upload a more relevant document. Do not remove the grounding instruction just to force an answer.

### Port 8501 is already in use
Run `python -m streamlit run app.py --server.port 8502`.

### PowerShell cannot activate `.venv`
Run the process-scoped execution-policy command in the setup section, or call `.venv\Scripts\python.exe` directly.

## 9. Security and privacy notes

- `.env` is excluded by `.gitignore`; never commit or screenshot your key.
- The PDF text and questions are sent to Google's Gemini API for embeddings and generation. Do not upload confidential or regulated documents unless you have authorization and have reviewed the provider's applicable terms.
- This demo keeps the active index and chat in Streamlit session state; it does not implement authentication, per-user isolation beyond the session, persistent storage, or production monitoring.
- Before production use, add access controls, file-size/page limits, logging without secrets, rate limiting, and a data-retention policy.

## 10. Interview explanation (60–90 seconds)

> I built a PDF question-answering assistant using Retrieval-Augmented Generation. When a user uploads a PDF, pypdf extracts text page by page. I split the text into overlapping chunks and preserve each chunk's page number as metadata. Gemini's embedding model converts each chunk into a vector, and I normalize those vectors and store them in a FAISS inner-product index. When the user asks a question, I embed the query, retrieve the top-k most similar chunks, and pass only those passages plus the question to Gemini. The prompt instructs the model to answer only from the evidence and say when the PDF doesn't contain the answer. The UI shows both the answer and the retrieved source passages, which improves transparency. I also handle invalid PDFs, empty/scanned documents, missing API keys, and API errors.

### Interview questions to practice

**1. What is RAG?**  
RAG combines retrieval with generation. It finds relevant external information first, then gives that information to a language model as context for its response.

**2. What is an embedding?**  
An embedding is a numeric vector representing semantic information in text. Similar meanings tend to be near each other in vector space.

**3. Why normalize vectors?**  
With L2-normalized vectors, the inner product equals cosine similarity, so FAISS `IndexFlatIP` can rank chunks by semantic similarity.

**4. What is top-k?**  
The number of retrieved chunks sent to the model. A small k is cheaper and focused; a larger k can provide more context but may add noise.

**5. How do you reduce hallucination?**  
Retrieve relevant evidence, instruct the model to use only that evidence, require it to admit when evidence is missing, and show source passages so people can verify the answer. This reduces risk but cannot guarantee correctness.

**6. What are the limitations?**  
Scanned PDFs need OCR; retrieval can miss relevant chunks; chunk sizes and top-k need tuning; API quotas and latency apply; and answers still need validation.

**7. How could you improve it?**  
Add OCR, hybrid keyword/vector search, reranking, document-level filters, persistent vector storage, streaming answers, evaluation datasets, feedback, authentication, and monitoring.

## 11. Smoke test

After installing dependencies, verify imports:

```powershell
python -c "import streamlit, pypdf, faiss, numpy; from google import genai; print('Imports OK')"
```

Then start the app and upload a small text-based PDF. Ask a question whose answer is clearly present, then ask one whose answer is absent. Inspect the retrieved source passages for both.
