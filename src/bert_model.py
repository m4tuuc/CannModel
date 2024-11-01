from transformers import BertTokenizer, BertModel
import torch


class BERTEmbedding:
    def __init__(self):
        """modelo y tokenizer de BERT en español."""
        self.tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased')
        self.model = BertModel.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased')

    def get_embedding(self, text):
        """genera un embedding para el texto dado."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)
