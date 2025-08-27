import argparse
import os
import sys

import pandas as pd

from categorizer.config import FieldWeights
from categorizer.data import TextBuildOptions
from categorizer.pipeline import Categorizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Categorize infrastructure projects using sentence embeddings")
    parser.add_argument("--input_path", required=True, help="Path to input CSV with columns: prde, sort, description")
    parser.add_argument("--output_path", required=True, help="Path to write output CSV with predictions")
    parser.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2", help="Hugging Face model name")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cpu or cuda")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top categories to include")
    parser.add_argument("--batch_size", type=int, default=64, help="Embedding batch size")
    parser.add_argument("--w_prde", type=int, default=1, help="Weight for prde field")
    parser.add_argument("--w_sort", type=int, default=0, help="Weight for sort field")
    parser.add_argument("--w_description", type=int, default=0, help="Weight for description field")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not os.path.exists(args.input_path):
        print(f"Input file not found: {args.input_path}", file=sys.stderr)
        return 1

    df = pd.read_csv(args.input_path)

    text_options = TextBuildOptions(weights=FieldWeights(prde=args.w_prde, sort=args.w_sort, description=args.w_description))
    categorizer = Categorizer(model_name=args.model_name, device=args.device, top_k=args.top_k, batch_size=args.batch_size)

    out_df = categorizer.categorize_dataframe(df, text_options)
    out_dir = os.path.dirname(os.path.abspath(args.output_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    out_df.to_csv(args.output_path, index=False)
    print(f"Wrote categorized CSV to: {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


