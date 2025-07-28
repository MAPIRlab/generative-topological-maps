import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class SentenceEmbedder:
    def __init__(self, model_id, device=None):
        """Initializes the SentenceEmbedder with lazy loading for the model and tokenizer."""
        self.device = device or (
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

    def _mean_pooling(self, model_output, attention_mask):
        """Applies mean pooling to get a fixed-size sentence embedding."""
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(
            -1).expand(token_embeddings.size()).float()
        return (token_embeddings * input_mask_expanded).sum(dim=1) / torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

    def embed_text(self, sentence: str, normalize=True):
        """Computes sentence embeddings with optional normalization."""
        self._initialize_model_and_tokenizer()
        encoded = self.tokenizer(
            [sentence], padding=True, truncation=True, return_tensors='pt').to(self.device)

        with torch.no_grad():
            model_output = self.model(**encoded)

        sentence_embedding = self._mean_pooling(
            model_output, encoded['attention_mask'])

        if normalize:
            sentence_embedding = F.normalize(sentence_embedding, p=2, dim=1)

        return sentence_embedding.cpu().numpy().tolist()[0]
