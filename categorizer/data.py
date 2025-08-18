from dataclasses import dataclass
from typing import Iterable, List, Optional

import pandas as pd

from .config import FieldWeights


REQUIRED_COLUMNS = ["prde", "sort", "description"]


@dataclass
class TextBuildOptions:
	weights: FieldWeights = FieldWeights()
	separator: str = " \n "


def validate_input_df(df: pd.DataFrame) -> None:
	missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
	if missing:
		raise ValueError(f"Missing required columns: {missing}")


def safe_str(value) -> str:
	if pd.isna(value):
		return ""
	return str(value)


def build_weighted_text(row: pd.Series, options: TextBuildOptions) -> str:
	parts: List[str] = []
	if options.weights.prde > 0:
		parts.append((safe_str(row.get("prde")), options.weights.prde))
	if options.weights.sort > 0:
		parts.append((safe_str(row.get("sort")), options.weights.sort))
	if options.weights.description > 0:
		description = row.get("description")
		if description:
			description = str(description).split('.')
			description = '.'.join( description[:min(len(description), 3)] )
			parts.append((safe_str(description), options.weights.description))

	# Repeat tokens proportionally to weights by simple scaling
	text_segments: List[str] = []
	for text, weight in parts:
		if not text:
			continue
		repeats = max(1, int(round(weight)))
		text_segments.append('/'.join([text] * repeats))
	return options.separator.join(text_segments).strip()


def prepare_corpus_texts(df: pd.DataFrame, options: Optional[TextBuildOptions] = None) -> List[str]:
	if options is None:
		options = TextBuildOptions()
	validate_input_df(df)
	texts = [build_weighted_text(row, options) for _, row in df.iterrows()]
	return texts


