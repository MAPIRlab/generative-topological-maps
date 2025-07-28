import torch
from transformers import AutoModel, AutoTokenizer


class WordEmbedder:
    def __init__(self, model_name, device=None):
        """Initializes the WordEmbedder with lazy loading for the model and tokenizer."""
        self.device = device or (
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

    def _initialize_model_and_tokenizer(self):
        """Lazy initialization of the tokenizer and model."""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.model is None:
            self.model = AutoModel.from_pretrained(
                self.model_name).to(self.device)
            self.model.eval()

    def embed_text(self, text, pooling='cls'):
        """Computes embeddings for the input text using the specified pooling method."""
        self._initialize_model_and_tokenizer()
        tokens = self.tokenizer(text, return_tensors='pt',
                                padding=True, truncation=True)
        input_ids = tokens['input_ids'].to(self.device)
        attention_mask = tokens['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)

        last_hidden_state = outputs.last_hidden_state

        if pooling == 'mean':
            embedding = (last_hidden_state *
                         attention_mask.unsqueeze(-1)).sum(dim=1)
            embedding /= attention_mask.sum(dim=1, keepdim=True)
        elif pooling == 'cls':
            embedding = last_hidden_state[:, 0, :]
        elif pooling == 'max':
            embedding = last_hidden_state.masked_fill(
                attention_mask.unsqueeze(-1) == 0, float('-inf')).max(dim=1)[0]
        else:
            raise ValueError(
                "Invalid pooling method. Choose from 'mean', 'cls', or 'max'.")

        return embedding.cpu().numpy().tolist()[0]
