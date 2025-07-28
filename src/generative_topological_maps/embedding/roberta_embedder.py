from generative_topological_maps.embedding.word_embedder import WordEmbedder


class RoBERTaEmbedder(WordEmbedder):
    def __init__(self, device=None):
        super().__init__('roberta-base', device)


if __name__ == "__main__":
    embedder = RoBERTaEmbedder()
    sentence = "RoBERTa is an optimized version of BERT."
    embedding = embedder.embed_text(sentence, pooling='mean')
    print("Embedding for sentence:", embedding[:50], "[...]")
    print("Embedding length:", len(embedding))
