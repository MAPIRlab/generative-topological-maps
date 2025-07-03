# RELATIONSHIPS

# Method 1. LLM
echo "RELATIONSHIPS INFERENCE; method: llm"
PYTHONPATH=src python src/generative_place_categorization/main_relationships.py -p --method llm --geometric-threshold 0.5 --llm-request

# Method 2. LVLM
echo "RELATIONSHIPS INFERENCE; method: lvlm"
PYTHONPATH=src python src/generative_place_categorization/main_relationships.py -p --method lvlm --num-images 2 --geometric-threshold 0.5 --number-maps 5 --llm-request

echo "Script completed successfully."
