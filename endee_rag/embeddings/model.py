from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class EmbeddingModel:
    """ Wrapper for generating embeddings using sentence-transformers. """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def generate(self, texts: List[str]) -> List[List[float]]:
        """ Generates vector embeddings for a list of texts. """
        embeddings = self.model.encode(texts)
        # Convert numpy array to list of lists
        return embeddings.tolist()

    def generate_single(self, text: str) -> List[float]:
        """ Generates a single vector embedding for a text string. """
        return self.generate([text])[0]
