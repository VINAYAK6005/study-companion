from pathlib import Path

from pypdf import PdfReader
from langchain_core.documents import Document

DATA_DIR = Path("data/notes")

def load_pdf(path: Path) -> list[Document]:
    """Turn one pdf into a list of documents, one per page. """

    reader = PdfReader(path)
    docs = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.split():
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": path.name, "page": page_number},
                )
            )
    return docs


def load_documents() -> list[Document]:
    """Load every pdf in the notes folder. """

    docs = []
    for path in sorted(DATA_DIR.glob("*pdf")):
        docs.extend(load_pdf(path))
    return docs

if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} pages")

    total_chars = sum(len(d.page_content) for d in docs)
    print(f"Average character per page: {total_chars // len(docs)}")
    
    first = docs[0]
    print("\nMetadata of first page:", first.metadata)
    print("\nFirst 500 characters:\n")
    print(first.page_content[:500])