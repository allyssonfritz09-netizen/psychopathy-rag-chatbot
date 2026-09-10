# Psychopathy RAG Chatbot

A domain-specific Retrieval-Augmented Generation chatbot, grounded in three
peer-reviewed sources on the etiology, clinical assessment (PCL-R), and
media portrayal of psychopathy. Built on the pipeline developed and tested
in `Allysson_Fritz_RAG_Chatbot_Training_Testing.ipynb`.

Pipeline: **PDF loading → chunking (500/50) → all-MiniLM-L6-v2 embeddings →
ChromaDB → Groq `openai/gpt-oss-20b`**

---

## 1. Run it locally (in VS Code)

1. Open this folder in VS Code (`File → Open Folder`).
2. Open a terminal (`` Ctrl+` ``) and create a virtual environment:
   ```bash
   python -m venv venv
   ```
   Activate it:
   ```bash
   # Windows
   venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```
   If VS Code prompts you to select this venv as the interpreter, say yes
   (or set it manually via the Python extension, bottom-right corner).
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```
   This opens the app in your browser at `http://localhost:8501`, with
   hot-reload on save.
5. Paste your Groq API key into the sidebar box when prompted. It's kept
   only in that browser session — never written to disk.

The first run will take a bit longer while it downloads the embedding
model and builds the vector index from the 3 PDFs in `data/`. After that
it's cached for the rest of the session.

---

## 2. Deploy it for free on Streamlit Community Cloud

**Step 1 — Push to GitHub**
```bash
git init
git add .
git commit -m "Psychopathy RAG chatbot"
```
Create a new repo on [github.com](https://github.com), then:
```bash
git remote add origin https://github.com/<your-username>/<repo-name>.git
git branch -M main
git push -u origin main
```
(You can also do all of this from VS Code's built-in Source Control panel
instead of the terminal.)

> `.gitignore` already excludes `venv/` and `.streamlit/secrets.toml`, so
> your API key will never be pushed.

**Step 2 — Deploy**
1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   GitHub.
2. Click **New app**, select your repo, branch `main`, and set the main
   file path to `app.py`.
3. Before (or right after) deploying, go to **Advanced settings → Secrets**
   (or **App settings → Secrets** once it's live) and paste:
   ```toml
   GROQ_API_KEY = "gsk_your_actual_key"
   ```
4. Deploy. The app reads `st.secrets["GROQ_API_KEY"]` automatically — no
   code changes needed, and the sidebar key-input box won't even appear
   since a key is already found.

Your Groq key is the same free-tier key you used in the Colab notebook —
30 requests/minute, 1,000/day, no credit card required.

---

## 3. What's in the sidebar

- **Source documents** — the three PDFs with full citations.
- **Advanced settings** — sliders for temperature and top-k, and a toggle
  for the fallback ("I cannot answer based on the provided domain data")
  rule. These map directly to the Task 2/3 hallucination stress-test from
  the lab notebook, so you can demo grounded vs. ungrounded behavior live
  in the deployed app.

## 4. Project structure

```
streamlit_app/
├── app.py                        # chat interface + RAG pipeline
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── secrets.toml.example      # copy to secrets.toml for local secrets (git-ignored)
└── data/
    ├── etiology_patrick2022.pdf
    ├── assessment_pclr_metaanalysis.pdf
    └── media_portrayal_lopera2022.pdf
```
