from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import json
import numpy as np
import pandas as pd

from .config import CATEGORY_ANCHORS, CATEGORY_LABELS
from .data import TextBuildOptions, FieldWeights, safe_str
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

	def _truncate_description(self, value: str) -> str:
		if not value:
			return ""
		desc = str(value).split('.')
		return '.'.join(desc[: min(len(desc), 3)])

	def _combine_field_similarities(
		self,
		prde_texts: List[str],
		sort_texts: List[str],
		description_texts: List[str],
		weights: FieldWeights,
	) -> np.ndarray:
		# Collect per-field similarity matrices based on non-zero weights
		weighted_sums: np.ndarray | None = None
		total_weight: float = 0.0

		if weights.prde > 0:
			qv = self.model.encode(prde_texts)
			sim = compute_cosine_similarities(qv, self.anchor_vectors)
			weighted_sums = sim * float(weights.prde) if weighted_sums is None else (weighted_sums + sim * float(weights.prde))
			total_weight += float(weights.prde)

		if weights.sort > 0:
			qv = self.model.encode(sort_texts)
			sim = compute_cosine_similarities(qv, self.anchor_vectors)
			weighted_sums = sim * float(weights.sort) if weighted_sums is None else (weighted_sums + sim * float(weights.sort))
			total_weight += float(weights.sort)

		if weights.description > 0:
			qv = self.model.encode(description_texts)
			sim = compute_cosine_similarities(qv, self.anchor_vectors)
			weighted_sums = sim * float(weights.description) if weighted_sums is None else (weighted_sums + sim * float(weights.description))
			total_weight += float(weights.description)

		# Fallback to prde with unit weight if all are zero
		if weighted_sums is None or total_weight == 0:
			qv = self.model.encode(prde_texts)
			weighted_sums = compute_cosine_similarities(qv, self.anchor_vectors)
			total_weight = 1.0

		return weighted_sums / total_weight

	def _aggregate_category_results(self, sim_matrix: np.ndarray) -> List[CategorizationResult]:
		results: List[CategorizationResult] = []
		for row in sim_matrix:
			category_to_score: Dict[str, float] = {label: -1.0 for label in CATEGORY_LABELS}
			for sim, anchor_label in zip(row.tolist(), self.anchor_labels):
				category_to_score[anchor_label] = max(category_to_score[anchor_label], sim)
			sorted_pairs = sorted(category_to_score.items(), key=lambda kv: kv[1], reverse=True)
			top_pairs = sorted_pairs[: self.top_k]
			label, score = top_pairs[0]
			results.append(
				CategorizationResult(
					label=label,
					score=float(score),
					top_categories=[(l, float(s)) for l, s in top_pairs if float(s)],
				)
			)
		return results

	def categorize_dataframe(self, df: pd.DataFrame, text_options: TextBuildOptions | None = None) -> pd.DataFrame:
		if text_options is None:
			text_options = TextBuildOptions()

		# Build per-field texts (no repetition); truncate description to first 3 sentences
		prde_texts: List[str] = [safe_str(row.get("prde")) for _, row in df.iterrows()]
		sort_texts: List[str] = [safe_str(row.get("sort")) for _, row in df.iterrows()]
		desc_texts: List[str] = [self._truncate_description(row.get("description")) for _, row in df.iterrows()]

		# Compute weighted similarity matrix across fields
		sim_matrix = self._combine_field_similarities(
			prde_texts=prde_texts,
			sort_texts=sort_texts,
			description_texts=desc_texts,
			weights=text_options.weights,
		)

		# Aggregate scores by category and choose top-k
		results = self._aggregate_category_results(sim_matrix)

		# Human-readable combined text for output (simple concatenation, no repetition)
		sep = text_options.separator
		combined_texts: List[str] = []
		for prde_val, sort_val, desc_val in zip(prde_texts, sort_texts, desc_texts):
			segments = [seg for seg in [prde_val, sort_val, desc_val] if seg]
			combined_texts.append(sep.join(segments).strip())

		# Assemble output
		out_df = df.copy()
		out_df["combined_text"] = combined_texts
		out_df["category_pred"] = [r.label for r in results]
		out_df["category_score"] = [r.score for r in results]
		out_df["top_categories"] = [json.dumps(r.top_categories) for r in results]
		out_df["category_calibrated"] = [
			r.top_categories[1][0]
			if (len(r.top_categories) > 1
				and r.top_categories[0][0] == "Non Infrastructure"
				and r.top_categories[1][1] > 0.3
				and r.top_categories[1][1] > r.top_categories[0][1] * 0.8)
			else r.top_categories[0][0]
			for r in results
		]
		return out_df


