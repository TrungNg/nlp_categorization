## Project categorization via sentence embeddings

This repo provides a simple NLP workflow to categorize infrastructure construction projects into seven categories using sentence embeddings and nearest-neighbor matching in embedding space.

Categories:
- Transportation (excluding ports & airports)
- Ports and Airports
- Water and Wastewater
- Energy and ICT
- Site development (including subdivision, parks, reserves, cemeteries, etc.)
- Environmental (including waste management, geotechnical, erosion control, flood control, etc.)
- Non infrastructure

### Input data

CSV with columns:
- `prde`: high-level short description
- `sort`: project name
- `description`: free-text description

The text used for embedding is a weighted concatenation of these fields.

### Quickstart

1) Create a virtual environment (optional but recommended)

```bash
python -m venv .venv && source .venv/bin/activate
```

2) Install dependencies

```bash
pip install -r requirements.txt
```

3) Run categorization

```bash
python main.py \
  --input_path path/to/projects.csv \
  --output_path path/to/projects_categorized.csv
```

Optional flags:
- `--model_name` (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `--device` (e.g., `cpu`, `cuda`)
- `--top_k` (default: 3) number of top categories to output

### Output

The output CSV includes the original columns and the following:
- `category_pred`: chosen category label
- `category_score`: cosine similarity to chosen category
- `top_categories`: JSON list of `[label, score]` pairs for the top-k categories
- `combined_text`: the text that was embedded for each row (for transparency)

### Notes

- Models are downloaded from Hugging Face on first use and cached locally.
- Cosine similarity is used; the highest similarity is selected.
- Category anchors are defined in `categorizer/config.py` and can be refined for your data.


