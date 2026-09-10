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

st.title("🧠 Understanding Psychopathy — RAG Chatbot")
st.caption(
    "Answers are grounded strictly in three peer-reviewed sources on the "
    "etiology, clinical assessment, and media portrayal of psychopathy."
)

with st.sidebar:
    st.header("About this chatbot")
    st.write(
        "This assistant answers **only** from three academic sources — it "
        "will not draw on general knowledge or media stereotypes about "
        "psychopathy. Ask an out-of-domain question and it will say so."
    )

    st.subheader("Source documents")
    for src in SOURCES:
        with st.expander(src["filename"]):
            st.write(src["citation"])
            st.caption(src["covers"])

    st.divider()
    st.subheader("Groq API key")
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
        st.success("API key loaded.")

    st.divider()
    with st.expander("Advanced settings"):
        st.caption(
            "Mirrors the Task 2/3 experiments from the notebook — try "
            "raising temperature and disabling the fallback rule to see "
            "hallucination behavior live."
        )
        temperature = st.slider("Temperature", 0.0, 1.0, DEFAULT_TEMPERATURE, 0.1)
        k = st.slider("Top-k retrieved chunks", 1, 10, DEFAULT_K, 1)
        fallback_enabled = st.checkbox(
            "Enable fallback rule (refuse out-of-domain questions)",
            value=True,
        )

if not api_key:
    st.info("Enter your Groq API key in the sidebar to start chatting.")
    st.stop()

# Build (or fetch cached) vectorstore
vectorstore, index_diagnostics = build_vectorstore()

with st.sidebar:
    with st.expander("Indexing diagnostics"):
        st.caption("What actually made it into the index, per source, after "
                   "boilerplate/reference filtering.")
        for d in index_diagnostics:
            st.caption(
                f"**{d['file']}** — {d['raw_pages']} pages loaded, "
                f"cutoff at page {d['cutoff_page']}, "
                f"{d['indexed_chunks']} chunks indexed"
            )

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources retrieved for this answer"):
                for i, doc in enumerate(msg["sources"], start=1):
                    fname = os.path.basename(doc.metadata.get("source", "Unknown"))
                    page = doc.metadata.get("page")
                    label = f"Chunk {i} — {fname}" + (f" (page {page})" if page is not None else "")
                    st.markdown(f"**{label}**")
                    st.text(doc.page_content[:400] + ("..." if len(doc.page_content) > 400 else ""))

user_query = st.chat_input("Ask about psychopathy's causes, assessment, or media portrayal...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
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
                for i, doc in enumerate(retrieved_docs, start=1):
                    fname = os.path.basename(doc.metadata.get("source", "Unknown"))
                    page = doc.metadata.get("page")
                    label = f"Chunk {i} — {fname}" + (f" (page {page})" if page is not None else "")
                    st.markdown(f"**{label}**")
                    st.text(doc.page_content[:400] + ("..." if len(doc.page_content) > 400 else ""))

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": retrieved_docs}
    )