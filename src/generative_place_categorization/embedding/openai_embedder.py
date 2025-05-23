import os

from openai import OpenAI


class OpenAIEmbedder:
    def __init__(self, model_name='text-embedding-3-small', api_key=None):
        """
        Initializes the OpenAI embedding model.

        Args:
            model_name (str): Name of the OpenAI embedding model. Defaults to 'text-embedding-3-small'.
            api_key (str, optional): OpenAI API key. If not provided, it uses the OPENAI_API_KEY environment variable.
        """
        self.model_name = model_name
        self.api_key = api_key if api_key else os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set the OPENAI_API_KEY environment variable or pass it as an argument.")

    def embed_text(self, text):
        """
        Embeds a given text using OpenAI's API.

        Args:
            text (str): A string (word or sentence) to be embedded.

        Returns:
            list: A list representing the text embedding.
        """
        client = OpenAI(api_key=self.api_key)
        response = client.embeddings.create(
            input=text,
            model=self.model_name
        )

        return response.data[0].embedding


# Example usage:
if __name__ == "__main__":
    embedder = OpenAIEmbedder()
    sentence = "OpenAI embeddings are powerful for NLP tasks."
    embedding = embedder.embed_text(sentence)
    print("Embedding length:", len(embedding))
