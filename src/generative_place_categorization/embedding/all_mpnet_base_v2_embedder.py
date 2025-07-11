from generative_place_categorization.embedding.sentence_embedder import SentenceEmbedder


class AllMpnetBaseV2Embedder(SentenceEmbedder):
    def __init__(self, device=None):
        super().__init__("sentence-transformers/all-mpnet-base-v2", device)


if __name__ == "__main__":
    embedder = AllMpnetBaseV2Embedder()
    sentence = "This is a test sentence."
    embedding = embedder.embed_text(sentence)
    print("Embedding for sentence:", embedding[:50], "[...]")
    print("Embedding length:", len(embedding))
