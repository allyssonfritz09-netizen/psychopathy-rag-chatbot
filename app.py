"""
Domain-Specific RAG Chatbot — Understanding Psychopathy
Etiology, Clinical Assessment, and Media Representation

Built on top of the RAG pipeline developed and tested in:
Allysson_Fritz_RAG_Chatbot_Training_Testing.ipynb

Pipeline: PDF Loading -> Chunking -> all-MiniLM-L6-v2 Embeddings ->
          ChromaDB -> Groq (openai/gpt-oss-20b)
"""

import os
import re
import glob
import uuid
from pathlib import Path

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# Load custom CSS (works locally and on Streamlit Cloud)
def load_css():
    css_file = Path(__file__).parent / "assets" / "styles.css"
    with open(css_file, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()


# --------------------------------------------------------------------------
# Fixed pipeline configuration (baseline values from the notebook)
# --------------------------------------------------------------------------
DATA_DIR = "data"
MODEL_NAME = "openai/gpt-oss-20b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 50
DEFAULT_K = 3
DEFAULT_TEMPERATURE = 0.0

FALLBACK_LINE = (
    "If at least one retrieved passage contains information relevant to the "
    "question, answer using that information, even if the passages do not "
    "provide enough detail for a complete answer. Clearly state any "
    "information that is not provided in the retrieved passages. "
    "Only if NONE of the retrieved passages contain information relevant "
    "to the question, reply exactly: "
    "'I cannot answer based on the provided domain data.'\n"
)

SYSTEM_PROMPT_TEMPLATE = (
    "You are a specialized AI assistant for a knowledge base about psychopathy "
    "— covering its clinical etiology, professional assessment methods (e.g., "
    "the PCL-R), and how accurately it is portrayed in media.\n"
    "Base your answer only on the passages provided below (no outside "
    "knowledge). Each passage is labeled with its source document. The "
    "passages are retrieved from three different papers, so some will "
    "usually be unrelated to this specific question -- that is expected, "
    "not a sign the question is unanswerable. Ignore irrelevant passages "
    "and synthesize an answer from whichever ones are relevant, even if "
    "they only partially cover the question or come from a single "
    "document.\n"
    "{fallback_line}"
    "\nContext:\n{context}"
)

SOURCES = [
    {
        "filename": "etiology_patrick2022.pdf",
        "citation": "Patrick, C. J. (2022). Psychopathy: Current Knowledge and "
                    "Future Directions. Annual Review of Clinical Psychology, 18.",
        "covers": "Genetic, neurobiological, and environmental factors contributing "
                  "to the development of psychopathic traits.",
    },
    {
        "filename": "assessment_pclr_metaanalysis.pdf",
        "citation": "Holper, S. et al. (2025). Criterion Validity of the Psychopathy "
                    "Checklist in Legal Contexts: An Updated Meta-Analysis. Journal "
                    "of Personality Assessment.",
        "covers": "How the Hare Psychopathy Checklist-Revised (PCL-R) is used to "
                  "clinically and legally assess psychopathy.",
    },
    {
        "filename": "media_portrayal_lopera2022.pdf",
        "citation": "Lopera-Mármol, M. et al. (2022). Aesthetic Representation of "
                    "Antisocial Personality Disorder in British Coming-of-Age TV "
                    "Series. Social Sciences, 11(3), 133.",
        "covers": "Critical analysis of how antisocial personality traits are "
                  "depicted in television media.",
    },
]

# Filename -> citation, so the LLM's context labels carry the same
# author/year identity as SOURCES (currently only shown in the sidebar).
# Without this, a passage is labeled only "[Source: etiology_patrick2022.pdf]"
# -- the model has no textual basis for confirming that file is "Patrick
# (2022)" when a question is phrased that way (e.g. "According to Patrick
# (2022), ..."), and under the strict fallback rule it can decline to answer
# a question it can't verify the attribution for, even when the retrieved
# passage is squarely on topic.
CITATION_BY_FILENAME = {src["filename"]: src["citation"] for src in SOURCES}


# --------------------------------------------------------------------------
# Retrieval fix: exclude the trailing bibliography/boilerplate section of
# each PDF from the indexed chunks (see notebook Cell 7 / Additional
# Exploration A & B, where vague or broad queries repeatedly pulled back
# "Author Contributions" boilerplate and reference-list text instead of
# substantive content). All three source papers place a run of
# non-substantive sections -- Author Contributions, Funding, Disclosure
# Statement, Acknowledgments, ORCID, Data Availability, and finally
# References/Bibliography/Literature Cited -- at the very end, and once
# one of these begins, every subsequent page of that same PDF is boilerplate
# through to the last page. We detect the first page where this run starts
# and drop that page and everything after it, per source file, before the
# chunks are embedded. Chunking, embeddings, vector store, and retrieval
# logic are otherwise unchanged.
# --------------------------------------------------------------------------
_LIGATURE_MAP = str.maketrans({
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl",
})

BOILERPLATE_SECTION_RE = re.compile(
    r"(?im)^[ \t]*(references|bibliography|literature\s+cited|works\s+cited|"
    r"author\s+contributions?|acknowledge?ments?|conflicts?\s+of\s+interest|"
    r"disclosure\s+statement|funding|data\s+availability\s+statement|"
    r"open\s+scholarship|orcid)[ \t]*(:|\n|$)"
)


def _find_boilerplate_cutoff_pages(raw_documents):
    """For each source PDF, find the earliest page containing a boilerplate
    section heading (References, Author Contributions, Acknowledgments,
    etc.), using the full, untruncated text of each page.

    This must run on whole pages -- not on the 500-char chunks produced by
    the splitter. A regex anchored to line/string start or end (^, $) can
    false-match on a chunk: an ordinary sentence that merely happens to be
    cut off at a chunk boundary (e.g. "...calling for funding" ending a
    chunk) looks identical, to the regex, to a real standalone heading. On
    the etiology paper this produced a false match on an early page, which
    caused nearly the entire document to be dropped as "post-boilerplate".
    Whole-page text has no such artificial boundaries.
    """
    cutoff_page = {}
    for doc in raw_documents:
        source = doc.metadata.get("source")
        page = doc.metadata.get("page")
        if source is None or page is None:
            continue
        normalized = doc.page_content.translate(_LIGATURE_MAP)
        if BOILERPLATE_SECTION_RE.search(normalized):
            if page < cutoff_page.get(source, float("inf")):
                cutoff_page[source] = page
    return cutoff_page


def _exclude_boilerplate_tail(chunks, cutoff_page):
    """Drop chunks on or after each source's boilerplate cutoff page.

    Content before the heading -- i.e. the actual paper body -- is left
    untouched.
    """
    return [
        chunk for chunk in chunks
        if chunk.metadata.get("source") not in cutoff_page
        or chunk.metadata.get("page") is None
        or chunk.metadata.get("page") < cutoff_page[chunk.metadata.get("source")]
    ]


# A table-of-contents entry in these PDFs is typeset as a heading followed
# by a run of leader dots and a page number, e.g.
#   "Neurobiological Research....................................... 404"
# This pattern is a reliable, self-contained signal -- unlike a heading-name
# regex, it does not depend on where a chunk boundary happens to fall, so it
# is safe to check directly against individual 500-char chunks.
TOC_ENTRY_RE = re.compile(r"\.{4,}\s*\d{1,4}")


def _exclude_toc_chunks(chunks, min_entries=1):
    """Drop chunks that are Table-of-Contents listings rather than body
    text. These retrieve well for broad/topical queries (they're literally
    a list of section titles) but contain no substantive discussion --
    e.g. a TOC chunk naming "Neurobiological Research" outranking the
    actual neurobiological-research paragraphs for a neurobiology query.

    min_entries=1 (not 2): the 500-char splitter can cut a chunk boundary
    between the second-to-last and last TOC line, leaving only the final
    entry -- e.g. "Neurobiological Research....... 404" -- alone at the
    top of the next chunk, immediately followed by the next section's
    heading and opening body text. A 2-entry threshold lets this single
    stray heading line through, and because it verbatim-matches the exact
    topic being asked about, it can outrank the real discussion for
    queries on that topic. TOC_ENTRY_RE's dot-leader pattern doesn't occur
    in ordinary prose, so a single match is already a reliable signal.
    """
    return [
        chunk for chunk in chunks
        if len(TOC_ENTRY_RE.findall(chunk.page_content)) < min_entries
    ]


# --------------------------------------------------------------------------
# API key resolution: Streamlit secrets (deployed) -> sidebar input (local)
# --------------------------------------------------------------------------
def get_api_key():
    key = st.secrets.get("GROQ_API_KEY") if hasattr(st, "secrets") else None
    if key:
        return key
    return st.session_state.get("groq_api_key_input", "")


# --------------------------------------------------------------------------
# Cached pipeline setup — runs once per app process
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Reading source documents and building the index...")
def build_vectorstore():
    pdf_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.pdf")))
    raw_documents = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        raw_documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(raw_documents)
    cutoff_page = _find_boilerplate_cutoff_pages(raw_documents)
    chunks = _exclude_boilerplate_tail(chunks, cutoff_page)
    chunks = _exclude_toc_chunks(chunks)

    # Diagnostics: what actually landed in the index, per source. Lets us
    # see -- live, in the running app -- whether a source's cutoff page and
    # indexed chunk count look sane, instead of only being able to check
    # this offline. Purely informational; does not affect retrieval.
    diagnostics = []
    for src in SOURCES:
        source_path = os.path.join(DATA_DIR, src["filename"])
        raw_pages = sum(1 for d in raw_documents if d.metadata.get("source") == source_path)
        indexed_chunks = sum(1 for c in chunks if c.metadata.get("source") == source_path)
        diagnostics.append({
            "file": src["filename"],
            "raw_pages": raw_pages,
            "cutoff_page": cutoff_page.get(source_path, "none detected"),
            "indexed_chunks": indexed_chunks,
        })

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    # A fresh, uniquely-named collection every time this function actually
    # runs. Without this, Chroma's default collection name ("langchain")
    # combined with its process-level client caching can cause a rebuild to
    # silently reuse or append to a stale collection from a previous run --
    # e.g. one built under an earlier, buggier version of this filter --
    # instead of indexing the current chunks from scratch.
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=f"psychopathy-{uuid.uuid4().hex}",
    )

    return vectorstore, diagnostics

def get_llm(api_key: str, temperature: float) -> ChatGroq:
    return ChatGroq(
        model_name=MODEL_NAME,
        temperature=temperature,
        groq_api_key=api_key,
        max_tokens=3072,
    )

def _retrieval_queries(query: str, filename: str):
    queries = [query]
    q = query.lower()

    if filename == "etiology_patrick2022.pdf":
        if any(term in q for term in [
            "genetic",
            "gene",
            "heritab",
            "familial",
            "twin",
            "adoption",
        ]):
            queries.extend([
                "psychopathy heritability",
                "genetic influences psychopathy",
                "disinhibition substantially heritable",
            ])

        if any(term in q for term in [
            "neurobiolog",
            "brain",
            "physiolog",
            "neural",
            "amygdala",
            "prefrontal",
        ]):
            queries.extend([
                "physiological correlates PPI Factor 1 psychopathy",
                "low threat sensitivity psychopathy",
                "weak frontal-inhibitory control psychopathy",
                "amygdala deficits sensitivity threat distress cues psychopathy",
                "role of the amygdala psychopathy",
                "deficient neural response fearful face stimuli psychopathy",
            ])

    return queries


def answer_query(query: str, vectorstore, api_key: str, k: int,
                 temperature: float, fallback_enabled: bool):

    retrieved_docs = []
    seen = set()

    for src in SOURCES:
        source_path = os.path.join(DATA_DIR, src["filename"])

        per_source_retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": k,
                "filter": {"source": source_path},
            }
        )

        retrieval_queries = _retrieval_queries(
            query,
            src["filename"]
        )

        for retrieval_query in retrieval_queries:
            results = vectorstore.similarity_search_with_score(
                retrieval_query,
                k=k,
                filter={"source": source_path},
            )

            for doc, distance in results:

                key = (
                    doc.metadata.get("source"),
                    doc.metadata.get("page"),
                    doc.page_content,
                )

                if key not in seen:
                    seen.add(key)
                    retrieved_docs.append((distance, doc))

        # Keep the most relevant chunks instead of taking the first 5
    # in retrieval order. This prevents earlier results from one
    # query/source from pushing out better matches from later queries.
    scored_docs = []

    retrieved_docs.sort(key=lambda item: item[0])

    retrieved_docs = [
        doc for distance, doc in retrieved_docs[:5]
    ]

    def _label(doc):
        fname = os.path.basename(
            doc.metadata.get("source", "unknown")
        )
        return CITATION_BY_FILENAME.get(fname, fname)

    context = "\n\n".join(
        f"[Source: {_label(doc)}]\n{doc.page_content}"
        for doc in retrieved_docs
    )

    fallback_line = FALLBACK_LINE if fallback_enabled else ""

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        fallback_line=fallback_line,
        context=context
    )

    llm = get_llm(api_key, temperature)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ]

    response = llm.invoke(messages)
    return response.content, retrieved_docs


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.set_page_config(page_title="Psychopathy RAG Chatbot", page_icon="🧠")

# --------------------------------------------------------------------------
# Embedded Lucide icons (local, static SVG path data — no CDN, no JS).
# Each constant holds only the inner <path>/<rect>/<circle> markup; icon()
# wraps it in a fresh <svg> sized for its call site.
# --------------------------------------------------------------------------
ICON_BRAIN = (
    '<path d="M12 18V5" /><path d="M15 13a4.17 4.17 0 0 1-3-4 4.17 4.17 0 0 1-3 4" />'
    '<path d="M17.598 6.5A3 3 0 1 0 12 5a3 3 0 1 0-5.598 1.5" />'
    '<path d="M17.997 5.125a4 4 0 0 1 2.526 5.77" /><path d="M18 18a4 4 0 0 0 2-7.464" />'
    '<path d="M19.967 17.483A4 4 0 1 1 12 18a4 4 0 1 1-7.967-.517" />'
    '<path d="M6 18a4 4 0 0 1-2-7.464" /><path d="M6.003 5.125a4 4 0 0 0-2.526 5.77" />'
)
ICON_BOOK_OPEN = (
    '<path d="M12 5v16" />'
    '<path d="M20.001 19A2 2 0 0022 17V5a2 2 0 00-1.999-2L16 3.002A5 5 0 0012 5a5 5 0 00-4-2'
    'H4a2 2 0 00-2 2v12a2 2 0 001.999 2H8a5 5 0 014 2 5 5 0 014-2z" />'
)
ICON_DNA = (
    '<path d="m10 16 1.5 1.5" /><path d="m14 8-1.5-1.5" />'
    '<path d="M15 2c-1.798 1.998-2.518 3.995-2.807 5.993" /><path d="m16.5 10.5 1 1" />'
    '<path d="m17 6-2.891-2.891" /><path d="M2 15c6.667-6 13.333 0 20-6" />'
    '<path d="m20 9 .891.891" /><path d="M3.109 14.109 4 15" /><path d="m6.5 12.5 1 1" />'
    '<path d="m7 18 2.891 2.891" /><path d="M9 22c1.798-1.998 2.518-3.995 2.807-5.993" />'
)
ICON_CLIPBOARD_CHECK = (
    '<rect width="8" height="4" x="8" y="2" rx="1" ry="1" />'
    '<path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2" />'
    '<path d="m9 14 2 2 4-4" />'
)
ICON_TV = '<path d="m17 2-5 5-5-5" /><rect width="20" height="15" x="2" y="7" rx="2" />'
ICON_KEY_ROUND = (
    '<path d="M2.586 17.414A2 2 0 0 0 2 18.828V21a1 1 0 0 0 1 1h3a1 1 0 0 0 1-1v-1a1 1 0 0 1 '
    '1-1h1a1 1 0 0 0 1-1v-1a1 1 0 0 1 1-1h.172a2 2 0 0 0 1.414-.586l.814-.814a6.5 6.5 0 1 0-4-4z" />'
    '<circle cx="16.5" cy="7.5" r=".5" fill="currentColor" />'
)
ICON_CHECK_CIRCLE = '<circle cx="12" cy="12" r="10" /><path d="m16 9-5.5 5.5L8 12" />'
ICON_BAR_CHART = (
    '<path d="M3 3v16a2 2 0 0 0 2 2h16" /><path d="M18 17V9" /><path d="M13 17V5" />'
    '<path d="M8 17v-3" />'
)
ICON_FILE_TEXT = (
    '<path d="M6 22a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h8a2.4 2.4 0 0 1 1.704.706l3.588 3.588'
    'A2.4 2.4 0 0 1 20 8v12a2 2 0 0 1-2 2z" /><path d="M14 2v5a1 1 0 0 0 1 1h5" />'
    '<path d="M10 9H8" /><path d="M16 13H8" /><path d="M16 17H8" />'
)

SOURCE_ICONS = [ICON_DNA, ICON_CLIPBOARD_CHECK, ICON_TV]

# Suggested-questions marquee: (chip key slug, question text). The icon for
# each is applied purely in CSS (see .st-key-chip_N selectors), since a
# native st.button label can't carry raw HTML/SVG.
SUGGESTED_QUESTIONS = [
    ("chip_0", "What genetic and neurobiological factors underlie psychopathy?"),
    ("chip_1", "How is the PCL-R used to assess psychopathy?"),
    ("chip_2", "How accurately does TV portray antisocial personality traits?"),
    ("chip_3", "What does reduced threat sensitivity look like neurologically?"),
    ("chip_4", "How is the PCL-R used in legal and courtroom settings?"),
    ("chip_5", "What separates psychopathy from ordinary antisocial behavior?"),
]


def icon(svg_inner: str, size: int = 15) -> str:
    """Wrap embedded path data in a sized <svg> tag. Presentation-only helper."""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="{size}" '
        f'height="{size}" fill="none" stroke="currentColor" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round">{svg_inner}</svg>'
    )


def card_header(svg_inner: str, title: str) -> None:
    """Icon badge + title row, used atop the markdown-built sidebar cards."""
    st.markdown(
        f'<div class="card-header"><span class="icon-badge">{icon(svg_inner)}</span>'
        f'<span class="card-title">{title}</span></div>',
        unsafe_allow_html=True,
    )


def field_label(mask_class: str, text: str) -> None:
    """Small icon + caption line, for spots where the native widget (slider,
    expander) can't take raw HTML in its own label."""
    st.markdown(
        f'<span class="icon-mask {mask_class}"></span>'
        f'<span class="card-body-text">{text}</span>',
        unsafe_allow_html=True,
    )


def render_source_chunks(docs) -> None:
    """Shared rendering for the 'Sources retrieved for this answer' expander,
    used both for history replay and the just-generated answer."""
    for i, doc in enumerate(docs, start=1):
        fname = os.path.basename(doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page")
        label = f"Chunk {i} — {fname}" + (f" (page {page})" if page is not None else "")
        st.markdown(
            f'<div class="source-chunk"><div class="source-chunk-label">'
            f'{icon(ICON_FILE_TEXT, 13)}{label}</div></div>',
            unsafe_allow_html=True,
        )
        st.text(doc.page_content[:400] + ("..." if len(doc.page_content) > 400 else ""))


# --------------------------------------------------------------------------
# Hero
# --------------------------------------------------------------------------
st.markdown(
    f'''
    <div class="hero-wrap">
      <div class="hero-inner">
        <div class="hero-icon">{icon(ICON_BRAIN, 24)}</div>
        <div>
          <p class="hero-title">Understanding Psychopathy</p>
          <p class="hero-subtitle">Answers are grounded strictly in three peer-reviewed
          sources on the etiology, clinical assessment, and media portrayal of
          psychopathy — a RAG chatbot with citations, not general knowledge.</p>
        </div>
      </div>
    </div>
    ''',
    unsafe_allow_html=True,
)

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

with st.container(key="marquee_viewport"):
    with st.container(key="marquee_track"):
        for slug, question in SUGGESTED_QUESTIONS:
            if st.button(question, key=slug):
                st.session_state.pending_question = question

with st.sidebar:
    with st.container(key="about_card"):
        card_header(ICON_BOOK_OPEN, "About this chatbot")
        st.markdown(
            '<p class="card-body-text">This assistant answers <b>only</b> from three '
            "academic sources — it will not draw on general knowledge or media "
            "stereotypes about psychopathy. Ask an out-of-domain question and it "
            "will say so.</p>",
            unsafe_allow_html=True,
        )

        st.divider()
    with st.expander("Advanced settings"):
        field_label(
            "icon-mask--sliders",
            "Mirrors the Task 2/3 experiments from the notebook — try raising "
            "temperature and disabling the fallback rule to see hallucination "
            "behavior live.",
        )
        temperature = st.slider("Temperature", 0.0, 1.0, DEFAULT_TEMPERATURE, 0.1)
        k = st.slider("Top-k retrieved chunks", 1, 10, DEFAULT_K, 1)
        fallback_enabled = st.checkbox(
            "Enable fallback rule (refuse out-of-domain questions)",
            value=True,
        )

    st.subheader("Source documents")
    for i, src in enumerate(SOURCES):
        with st.container(key=f"source_card_{i}"):
            card_header(SOURCE_ICONS[i], src["filename"])
            with st.expander(src["filename"]):
                st.markdown(
                    f'<p class="source-citation">{src["citation"]}</p>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<p class="source-covers">{src["covers"]}</p>',
                    unsafe_allow_html=True,
                )

    st.divider()
    st.subheader("Groq API key")
    with st.container(key="apikey_card"):
        card_header(ICON_KEY_ROUND, "Groq API key")
        api_key = get_api_key()
        if not api_key:
            st.text_input(
                "Paste your Groq API key",
                type="password",
                key="groq_api_key_input",
                help="Not saved anywhere — only kept for this browser session. "
                     "On the deployed version, this is set once via app secrets.",
            )
            api_key = get_api_key()
        else:
            st.markdown(
                f'<span class="apikey-status apikey-status--ok">'
                f'{icon(ICON_CHECK_CIRCLE, 14)}API key loaded</span>',
                unsafe_allow_html=True,
            )



# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    avatar = "🧠" if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources retrieved for this answer"):
                render_source_chunks(msg["sources"])


typed_query = st.chat_input("Ask about psychopathy's causes, assessment, or media portrayal...")
user_query = st.session_state.pending_question or typed_query
if st.session_state.pending_question:
    st.session_state.pending_question = None

if not api_key:
    st.info("Enter your Groq API key in the sidebar to start chatting.")
    st.stop()

# -------------------------------------------------------------
# Render the chat UI first, then initialize the knowledge base
# -------------------------------------------------------------

# Placeholders for content that depends on the vector index
diagnostics_placeholder = st.empty()
loading_placeholder = st.empty()

with loading_placeholder.container():
    with st.status("Initializing knowledge base...", expanded=False):
        vectorstore, index_diagnostics = build_vectorstore()

loading_placeholder.empty()

with diagnostics_placeholder.container():
    with st.sidebar:
        with st.expander("Indexing diagnostics"):
            field_label(
                "icon-mask--bar",
                "What actually made it into the index after filtering."
            )
            for d in index_diagnostics:
                st.markdown(
                    f'<div class="diag-row"><b>{d["file"]}</b> — '
                    f'{d["raw_pages"]} pages, cutoff {d["cutoff_page"]}, '
                    f'{d["indexed_chunks"]} chunks</div>',
                    unsafe_allow_html=True,
                )


if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant", avatar="🧠"):
        with st.spinner("Retrieving context and generating an answer..."):
            try:
                answer, retrieved_docs = answer_query(
                    user_query, vectorstore, api_key, k, temperature, fallback_enabled
                )
            except Exception as e:
                answer = f"Something went wrong calling Groq: {e}"
                retrieved_docs = []
        st.markdown(answer)
        if retrieved_docs:
            with st.expander("Sources retrieved for this answer"):
                render_source_chunks(retrieved_docs)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": retrieved_docs}
    )