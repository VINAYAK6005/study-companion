import sys

from ingest import load_documents, split_documents

source_prefix = sys.argv[1]
page = int(sys.argv[2])

chunks = split_documents(load_documents())
lengths = [len(c.page_content) for c in chunks]
print(f"Total: {len(chunks)} chunks, min={min(lengths)}, "
      f"max={max(lengths)}, avg={sum(lengths) // len(lengths)}")

for chunk in chunks:
    meta = chunk.metadata
    if meta["source"].startswith(source_prefix) and meta["page"] == page:
        print(f"\n--- {meta}  ({len(chunk.page_content)} chars)")
        print(chunk.page_content)