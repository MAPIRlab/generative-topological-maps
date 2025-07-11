

from generative_place_categorization.embedding.word_embedder import WordEmbedder


class BERTEmbedder(WordEmbedder):
    def __init__(self, device=None):
        super().__init__('bert-base-uncased', device)


if __name__ == "__main__":
    embedder = BERTEmbedder()
    sentence = "BERT is a powerful transformer model."
    embedding = embedder.embed_text(sentence, pooling='mean')
    print("Embedding for sentence:", embedding[:50], "[...]")
    print("Embedding length:", len(embedding))
