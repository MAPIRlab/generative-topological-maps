#!/bin/bash
# PLACES

# Stage 1. Categorization

# Method 1. geometric
echo "PLACE SEGMENTATION; method: geometric"
PYTHONPATH=src python src/generative_place_categorization/main_places.py --stage segmentation -n 10 --method geometric --clustering-algorithm hdbscan --dimensionality_reductor pca

# Method 2. bert
echo "PLACE SEGMENTATION; method: bert"
PYTHONPATH=src python src/generative_place_categorization/main_places.py --stage segmentation -n 10 --method bert --clustering-algorithm hdbscan --semantic-weight 0.55 --semantic-dimension 3 --dimensionality_reductor pca

# Method 3. roberta
echo "PLACE SEGMENTATION; method: roberta"
PYTHONPATH=src python src/generative_place_categorization/main_places.py --stage segmentation -n 10 --method roberta --clustering-algorithm hdbscan  --semantic-weight 0.55 --semantic-dimension 3 --dimensionality_reductor pca

# Method 4. bert+post
echo "PLACE SEGMENTATION; method: bert+post"
PYTHONPATH=src python src/generative_place_categorization/main_places.py --stage segmentation -n 10 --method bert+post --clustering-algorithm hdbscan --semantic-weight 0.55 --semantic-dimension 3 --dimensionality_reductor pca --merge-geometric-threshold 1.0 --merge-semantic-threshold 0.95 --split-semantic-threshold 2.5

# Method 5. llm+sbert
echo "PLACE SEGMENTATION; method: llm+sbert"
PYTHONPATH=src python src/generative_place_categorization/main_places.py --stage segmentation -n 10 --method llm+sbert --clustering-algorithm hdbscan  --semantic-weight 0.55 --semantic-dimension 3 --dimensionality_reductor pca

# Method 6. llm+sbert+post
echo "PLACE SEGMENTATION; method: llm+sbert+post"
PYTHONPATH=src python src/generative_place_categorization/main_places.py --stage segmentation -n 10 --method llm+sbert+post --clustering-algorithm hdbscan  --semantic-weight 0.55 --semantic-dimension 3 --dimensionality_reductor pca --merge-geometric-threshold 1.0 --merge-semantic-threshold 0.95 --split-semantic-threshold 2.5

# Method 7. llm
echo "PLACE SEGMENTATION; method: llm"
PYTHONPATH=src python src/generative_place_categorization/main_places.py --stage segmentation -n 10 --method llm --llm-request

# Stage 2. Categorization
echo "PLACE CATEGORIZATION"
PYTHONPATH=src python src/generative_place_categorization/main_places.py -p --stage categorization

# EVALUATE PLACES
echo "SEGMENTATION EVALUATION"
PYTHONPATH=src python src/generative_place_categorization/evaluate_places.py