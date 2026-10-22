import re
from pathlib import Path

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path("data/notes")
PAGE_LINE = re.compile(r"Page \d+")
BULLET_CHARS = re.compile(r"[\x7f\ue000-\uf8ff]")
CHUNK_SIZE = 500
CHUNK_OVERLAP=50


def clean_page(text: str, header: str) -> str:
    """Remove the repeated title header and 'Page N' lines from a page."""
    kept = []
    for line in text.splitlines():
        stripped = BULLET_CHARS.sub("-", line).strip()
        if stripped == header or PAGE_LINE.fullmatch(stripped):
            continue
        kept.append(stripped)
    return "\n".join(kept).strip()


def load_pdf(path: Path) -> list[Document]:
    """Turn one PDF into a list of cleaned Documents, one per page."""
    reader = PdfReader(path)
    pages_text = [page.extract_text() or "" for page in reader.pages]

    # The first line of page 1 is the header repeated on every page
    first_page = pages_text[0].strip()
    header = first_page.splitlines()[0].strip() if first_page else ""

    docs = []
    for page_number, raw_text in enumerate(pages_text, start=1):
        text = clean_page(raw_text, header)
        if text:
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": path.name, "page": page_number},
                )
            )
    return docs


def load_documents() -> list[Document]:
    """Load every PDF in the notes folder."""
    docs = []
    for path in sorted(DATA_DIR.glob("*.pdf")):
        docs.extend(load_pdf(path))
    return docs


def split_documents(docs: list[Document])-> list[Document]:
    """split pages into overlapping chunks. Metadata is copied to each chunk. """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
    )

    return splitter.split_documents(docs)

if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} pages")

    chunks = split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    lengths = [len(c.page_content) for c in chunks]
    print(f"Chunk length: min={min(lengths)}, max={max(lengths)}, "
          f"avg={sum(lengths) // len(lengths)}")

    print("\nFirst chunk:")
    print("---", chunks[0].metadata)
    print(chunks[0].page_content)