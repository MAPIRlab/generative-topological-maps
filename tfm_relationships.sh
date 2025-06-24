# RELATIONSHIPS

# Method 1. LLM
echo "RELATIONSHIPS INFERENCE; method: llm"
PYTHONPATH=src python src/generative_place_categorization/main_relationships.py -p --method llm --llm-request --geometric-threshold 0.7

# Method 2. LVLM
echo "RELATIONSHIPS INFERENCE; method: lvlm"
PYTHONPATH=src python src/generative_place_categorization/main_relationships.py -p --method lvlm --llm-request --num-images 2 --geometric-threshold 0.7 --number-maps 5

echo "Script completed successfully."
