import math

from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

# Part 1: what does an embedding look like?
vector = embeddings.embed_query("LangGraph uses conditional edges for routing.")
print(f"Vector length: {len(vector)}")
print(f"First 5 numbers: {vector[:5]}")


# Part 2: does it capture meaning?
def cosine_similarity(a: list[float], b: list[float]) -> float:
    """1.0 = same direction (similar meaning), near 0 = unrelated."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


question = "How does an agent decide which path to take next?"
sentences = [
    "Conditional edges route the graph to different nodes based on state.",
    "Chunk overlap repeats text between neighbouring chunks.",
    "Paris is the capital of France.",
]

question_vector = embeddings.embed_query("search_query: " + question)
sentence_vectors = embeddings.embed_documents(
    ["search_document: " + s for s in sentences]
)

print(f"\nQuestion: {question}\n")
for sentence, sentence_vector in zip(sentences, sentence_vectors):
    score = cosine_similarity(question_vector, sentence_vector)
    print(f"{score:.3f}  {sentence}")