import re
from pathlib import Path

from pypdf import PdfReader
from langchain_core.documents import Document

DATA_DIR = Path("data/notes")
PAGE_LINE = re.compile(r"Page \d+")
PRIVATE_USE = re.compile(r"[\ue000-\uf8ff]")


def clean_page(text: str, header: str) -> str:
    """Remove the repeated title header and 'Page N' lines from a page."""
    kept = []
    for line in text.splitlines():
        stripped = PRIVATE_USE.sub("-", line).strip()
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


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} pages")

    total_chars = sum(len(d.page_content) for d in docs)
    print(f"Average characters per page: {total_chars // len(docs)}")

    sample = docs[5]
    print("\nMetadata:", sample.metadata)
    print("\nCleaned text:\n")
    print(sample.page_content[:300])