import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class SentenceBERTEmbedder:
    def __init__(self, model_id="sentence-transformers/all-mpnet-base-v2", device=None):
        """
        Initializes the SentenceBERTEmbedder with lazy loading for the model and tokenizer.

        Args:
            model_name (str): Name of the pretrained Sentence-BERT model.
            device (str, optional): Device to run the model on ('cuda' or 'cpu'). Defaults to None.
        """
        self.device = device if device else (
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model_id = model_id
        self.tokenizer = None
        self.model = None

    def _initialize_model_and_tokenizer(self):
        """Lazy initialization of the tokenizer and model."""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.model is None:
            self.model = AutoModel.from_pretrained(
                self.model_id).to(self.device)
            self.model.eval()

    def mean_pooling(self, model_output, attention_mask):
        """
        Applies mean pooling to get a fixed-size sentence embedding.

        Args:
            model_output (torch.Tensor): The model output containing token embeddings.
            attention_mask (torch.Tensor): The attention mask to account for padding.

        Returns:
            torch.Tensor: Sentence embedding after mean pooling.
        """
        token_embeddings = model_output.last_hidden_state  # Extract token embeddings
        input_mask_expanded = attention_mask.unsqueeze(
            -1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def embed_text(self, sentence: str, normalize=True):
        """
        Computes sentence embeddings with optional normalization.

        Args:
            sentences (str): A single sentence or a list of sentences.
            normalize (bool): Whether to normalize embeddings (L2 norm). Defaults to True.

        Returns:
            list: A list of sentence embeddings.
        """
        self._initialize_model_and_tokenizer()

        # Tokenize input
        encoded_input = self.tokenizer(
            [sentence], padding=True, truncation=True, return_tensors='pt').to(self.device)

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Apply mean pooling
        sentence_embeddings = self.mean_pooling(
            model_output, encoded_input['attention_mask'])

        # Normalize embeddings if required
        if normalize:
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings.cpu().numpy().tolist()[0]


if __name__ == "__main__":

    embedder = SentenceBERTEmbedder(model_id="Qwen/Qwen3-Embedding-8B")
    sentence = "This is a test sentence."
    embedding = embedder.embed_text(sentence)
    print(f"Embedding for '{sentence}': {embedding}")
