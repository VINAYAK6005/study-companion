from store import get_vector_store
import readline  # enables arrow keys and history in input()

store = get_vector_store()

while True:
    question = input("\nQuestion (or 'quit): ").strip()
    if question.lower() in {"quit", "exit", "bye"}:
        break
    if not question:
        continue

    result = store.similarity_search_with_score(question, k=3)

    for rank, (doc, distance) in enumerate(result, start=1):
        meta = doc.metadata
        print(f"\n#{rank}  distance={distance:.3f}  |  {meta['source']}, page {meta['page']}")
        print(doc.page_content[:200])

