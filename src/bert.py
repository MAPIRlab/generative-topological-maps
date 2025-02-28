from transformers import BertTokenizer, BertModel
import torch

class BERTEmbedder:
    def __init__(self, model_name='bert-base-uncased', device=None):
        """
        Initializes the BERT model and tokenizer.
        
        Args:
            model_name (str): Name of the pretrained BERT model.
            device (str, optional): Device to run the model on ('cuda' or 'cpu'). Defaults to None.
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def embed_text(self, text, pooling='mean'):
        """
        Embeds a given text (word or sentence) using BERT.
        
        Args:
            text (str): A string (word or sentence) to be embedded.
            pooling (str): Pooling method ('mean', 'cls', or 'max'). Defaults to 'mean'.
        
        Returns:
            list: A list representing the text embedding.
        """
        tokens = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        input_ids = tokens['input_ids'].to(self.device)
        attention_mask = tokens['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
        
        last_hidden_state = outputs.last_hidden_state  # Shape: (batch_size, sequence_length, hidden_dim)
        
        if pooling == 'mean':
            embedding = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        elif pooling == 'cls':
            embedding = last_hidden_state[:, 0, :]  # CLS token embedding
        elif pooling == 'max':
            embedding = last_hidden_state.masked_fill(attention_mask.unsqueeze(-1) == 0, float('-inf')).max(dim=1)[0]
        else:
            raise ValueError("Invalid pooling method. Choose from 'mean', 'cls', or 'max'.")
        
        return embedding.cpu().numpy().tolist()[0]

# Example usage:
if __name__ == "__main__":
    embedder = BERTEmbedder()
    sentence = "BERT is a powerful transformer model."
    embedding = embedder.embed_text(sentence, pooling='mean')
    print("Embedding length:", len(embedding))
