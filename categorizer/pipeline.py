from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import json
import numpy as np
import pandas as pd

from .config import CATEGORY_ANCHORS, CATEGORY_LABELS
from .data import TextBuildOptions, prepare_corpus_texts
from .embed import EmbeddingModel, compute_cosine_similarities


def _flatten_category_anchors() -> Tuple[List[str], List[str]]:
	labels: List[str] = []
	texts: List[str] = []
	for category, phrases in CATEGORY_ANCHORS.items():
		for phrase in phrases:
			labels.append(category)
			texts.append(category + " - " + phrase)
	return labels, texts


@dataclass
class CategorizationResult:
	label: str
	score: float
	top_categories: List[Tuple[str, float]]


class Categorizer:
	def __init__(
		self,
		model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
		device: str | None = None,
		top_k: int = 1,
		batch_size: int = 64,
	):
		self.model = EmbeddingModel(model_name=model_name, device=device, batch_size=batch_size)
		self.top_k = max(1, top_k)

		# Build anchor embedding matrix
		self.anchor_labels, self.anchor_texts = _flatten_category_anchors()
		self.anchor_vectors = self.model.encode(self.anchor_texts)

	def categorize_texts(self, texts: List[str]) -> List[CategorizationResult]:
		if len(texts) == 0:
			return []
		query_vectors = self.model.encode(texts)
		sim_matrix = compute_cosine_similarities(query_vectors, self.anchor_vectors)

		results: List[CategorizationResult] = []
		for row in sim_matrix:
			# Aggregate scores by category (max over anchor phrases per category)
			category_to_score: Dict[str, float] = {label: -1.0 for label in CATEGORY_LABELS}
			for sim, anchor_label in zip(row.tolist(), self.anchor_labels):
				category_to_score[anchor_label] = max(category_to_score[anchor_label], sim)

			# Sort and choose top-k
			sorted_pairs = sorted(category_to_score.items(), key=lambda kv: kv[1], reverse=True)
			top_pairs = sorted_pairs[: self.top_k]
			label, score = top_pairs[0]
			results.append(CategorizationResult(label=label, score=float(score), top_categories=[(l, float(s)) for l, s in top_pairs]))
		return results

	def categorize_dataframe(self, df: pd.DataFrame, text_options: TextBuildOptions | None = None) -> pd.DataFrame:
		texts = prepare_corpus_texts(df, text_options)
		results = self.categorize_texts(texts)
		# Assemble output
		out_df = df.copy()
		out_df["combined_text"] = texts
		out_df["category_pred"] = [r.label for r in results]
		out_df["category_score"] = [r.score for r in results]
		out_df["top_categories"] = [json.dumps(r.top_categories) for r in results]
		return out_df


