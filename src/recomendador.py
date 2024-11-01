import torch
from sklearn.metrics.pairwise import cosine_similarity


class RecomendadorCannabis:
    def __init__(self, df, modelo_bert):
        """Inicializa el recomendador con el dataset y modelo de embeddings."""
        self.df = df
        self.modelo_bert = modelo_bert
        self.embeddings = self._generar_embeddings(df)

    def _generar_embeddings(self, df):
        """Genera embeddings para todas las descripciones del dataset."""
        embeddings = []
        for descripcion in df['Description']:
            embedding = self.modelo_bert.get_embedding(descripcion)
            embeddings.append(embedding)
        return torch.stack(embeddings).squeeze()

    def recomendar(self, entrada_usuario, top_n=5):
        """Recomienda las cepas mas similares al input del usuario."""
        embedding_usuario = self.modelo_bert.get_embedding(entrada_usuario)
        similitudes = cosine_similarity(embedding_usuario, self.embeddings)
        mejores_indices = similitudes.argsort()[0][-top_n:][::-1]
        return self.df.iloc[mejores_indices]
