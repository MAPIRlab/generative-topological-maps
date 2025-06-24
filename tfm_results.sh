
# PLACES

# Stage 1. Categorization

# Method 1. geometric
echo "PLACE SEGMENTATION; method: geometric"
PYTHONPATH=src python src/generative_place_categorization/main_places.py -p --stage segmentation -n 10 --method geometric --clustering-algorithm dbscan --eps 1.5 --min-samples 2 --dimensionality_reductor pca

# Method 2. bert
echo "PLACE SEGMENTATION; method: bert"
PYTHONPATH=src python src/generative_place_categorization/main_places.py -p --stage segmentation -n 10 --method bert --clustering-algorithm dbscan --eps 1.5 --min-samples 2 --semantic-weight 0.55 --semantic-dimension 3 --dimensionality_reductor pca

# Method 3. roberta
echo "PLACE SEGMENTATION; method: roberta"
PYTHONPATH=src python src/generative_place_categorization/main_places.py -p --stage segmentation -n 10 --method roberta --clustering-algorithm dbscan --eps 1.5 --min-samples 2  --semantic-weight 0.55 --semantic-dimension 3 --dimensionality_reductor pca

# Method 4. deepseek+sbert
echo "PLACE SEGMENTATION; method: deepseek+sbert"
PYTHONPATH=src python src/generative_place_categorization/main_places.py -p --stage segmentation -n 10 --method llm+sbert --clustering-algorithm dbscan --eps 1.5 --min-samples 2  --semantic-weight 0.55 --semantic-dimension 3 --dimensionality_reductor pca

# Method 4. llm+sbert+post
echo "PLACE SEGMENTATION; method: deepseek+sbert+post"
PYTHONPATH=src python src/generative_place_categorization/main_places.py -p --stage segmentation -n 10 --method llm+sbert+post --clustering-algorithm dbscan --eps 1.5 --min-samples 2  --semantic-weight 0.55 --semantic-dimension 3 --dimensionality_reductor pca --merge-geometric-threshold 1.0 --merge-semantic-threshold 0.55 --split-semantic-threshold 2.5

# Method 5. llm
echo "PLACE SEGMENTATION; method: llm"
PYTHONPATH=src python src/generative_place_categorization/main_places.py -p --stage segmentation -n 10 --method llm --llm-request

# Stage 2. Categorization
echo "PLACE CATEGORIZATION"
PYTHONPATH=src python src/generative_place_categorization/main_places.py -p --stage categorization

# EVALUATE PLACES
echo "SEGMENTATION EVALUATION"
PYTHONPATH=src python src/generative_place_categorization/evaluate_places.py

# RELATIONSHIPS

# Method 1. LLM
echo "RELATIONSHIPS INFERENCE; method: llm"
PYTHONPATH=src python src/generative_place_categorization/main_relationships.py -p --method llm --llm-request --geometric-threshold 0.7

# Method 2. LVLM
echo "RELATIONSHIPS INFERENCE; method: lvlm"
PYTHONPATH=src python src/generative_place_categorization/main_relationships.py -p --method lvlm --llm-request --num-images 2 --geometric-threshold 0.7 --number-maps 5

echo "Script completed successfully."
