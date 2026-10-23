from collections import Counter
from pathlib import Path

from pypdf import PdfReader

DATA_DIR = Path("data/notes")

pdf_paths = sorted(DATA_DIR.glob("*.pdf"))
if not pdf_paths:
    print(f"No PDFs found in {DATA_DIR}/")

for path in pdf_paths:
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    non_empty = [t.strip() for t in pages if t.strip()]
    lengths = [len(t) for t in pages]

    print(f"\n=== {path.name} ===")
    print(f"Pages: {len(pages)}, empty pages: {len(pages) - len(non_empty)}")
    print(f"Characters per page: min={min(lengths)}, max={max(lengths)}, "
          f"avg={sum(lengths) // len(lengths)}")

    first_lines = Counter(t.splitlines()[0] for t in non_empty)
    last_lines = Counter(t.splitlines()[-1] for t in non_empty)
    print("Most common first lines:", first_lines.most_common(3))
    print("Most common last lines:", last_lines.most_common(3))

    unusual = sorted({hex(ord(c)) for t in pages for c in t if ord(c) >= 127})
    print("Unusual characters:", unusual[:20])

    sample = pages[min(5, len(pages) - 1)]
    print("Sample page (raw):", repr(sample[:300]))