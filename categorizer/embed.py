from typing import Iterable, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
_NORMALIZED_OUTPUT = True

def l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
	denom = np.linalg.norm(vectors, axis=1, keepdims=True)
	denom = np.maximum(denom, eps)
	return vectors / denom


class EmbeddingModel:
	def __init__(
		self,
		model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
		device: Optional[str] = None,
		batch_size: int = 64,
	):
		self.model = SentenceTransformer(model_name, device=device)
		self.batch_size = batch_size

	def encode(self, texts: Iterable[str]) -> np.ndarray:
		embeddings = self.model.encode(
			list(texts),
			batch_size=self.batch_size,
			convert_to_numpy=True,
			normalize_embeddings=_NORMALIZED_OUTPUT,
		)
		return embeddings


def compute_cosine_similarities(query_vectors: np.ndarray, key_vectors: np.ndarray) -> np.ndarray:
	"""Return matrix of shape (n_queries, n_keys) with cosine similarities.

	Assumes inputs are L2-normalized. If not, will normalize defensively.
	"""
	#if not np.allclose(np.linalg.norm(query_vectors, axis=1), 1.0, atol=1e-3):
	if not _NORMALIZED_OUTPUT:
		query_vectors = l2_normalize(query_vectors)
		key_vectors = l2_normalize(key_vectors)
	return np.matmul(query_vectors, key_vectors.T)


