from generative_topological_maps.embedding.sentence_embedder import SentenceEmbedder


class Qwen3Embedder(SentenceEmbedder):
    def __init__(self, device=None):
        super().__init__("Qwen/Qwen3-Embedding-8B", device)


if __name__ == "__main__":
    embedder = Qwen3Embedder()
    sentence = "This is a test sentence."
    embedding = embedder.embed_text(sentence)
    print("Embedding for sentence:", embedding[:50], "[...]")
    print("Embedding length:", len(embedding))
