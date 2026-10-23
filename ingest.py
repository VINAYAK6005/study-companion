import re
from pathlib import Path

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from store import get_vector_store
from collections import Counter

DATA_DIR = Path("data/notes")
PAGE_LINE = re.compile(r"Page \d+")
BULLET_CHARS = re.compile(r"[\x7f\ue000-\uf8ff]")
DOT_LEADER = re.compile(r"(?:\.\s+){5,}")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def clean_page(text: str, header: str) -> str:
    """Remove the repeated title header and 'Page N' lines from a page."""
    kept = []
    for line in text.splitlines():
        stripped = BULLET_CHARS.sub("-", line).strip()
        if stripped == header or PAGE_LINE.fullmatch(stripped):
            continue
        kept.append(stripped)
    return "\n".join(kept).strip()

def is_table_of_contents(text: str) -> bool:
    """A contents page has many lines of dot leaders like '. . . . . 3'."""
    return len(DOT_LEADER.findall(text)) >= 3


def find_header(pages_text: list[str]) -> str:
    """Return the first line repeated on most pages, or '' if there isn't one."""
    first_lines = [t.strip().splitlines()[0].strip() for t in pages_text if t.strip()]
    if not first_lines:
        return ""
    line, count = Counter(first_lines).most_common(1)[0]
    return line if count > len(first_lines) / 2 else ""

def load_pdf(path: Path) -> list[Document]:
    """Turn one PDF into a list of cleaned Documents, one per page."""
    reader = PdfReader(path)
    pages_text = [page.extract_text() or "" for page in reader.pages]

    header = find_header(pages_text)
    title = header or path.stem.replace("_", " ")

    docs = []
    for page_number, raw_text in enumerate(pages_text, start=1):
        text = clean_page(raw_text, header)
        if text and not is_table_of_contents(text):
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": path.name, "page": page_number, "title": title},
                )
            )
    return docs


def load_documents() -> list[Document]:
    """Load every PDF in the notes folder."""
    docs = []
    for path in sorted(DATA_DIR.glob("*.pdf")):
        docs.extend(load_pdf(path))
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    """Split pages into overlapping chunks, each labelled with its document title."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    for chunk in chunks:
        chunk.page_content = f"{chunk.metadata['title']}\n\n{chunk.page_content}"
    return chunks

def build_vector_store(chunks: list[Document]):
    """Embed all chunks and store them, replacing any previous contents."""
    store = get_vector_store()
    store.reset_collection()
    store.add_documents(chunks)
    return store


if __name__ == "__main__":
    docs = load_documents()
    chunks = split_documents(docs)
    print(f"Loaded {len(docs)} pages, split into {len(chunks)} chunks")

    print("Embedding and storing chunks (this can take a minute on CPU)...")
    store = build_vector_store(chunks)
    print(f"Stored {len(store.get()['ids'])} chunks in Chroma")