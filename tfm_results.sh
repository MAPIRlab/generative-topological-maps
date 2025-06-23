

# RELATIONSHIPS

# Method 1. LLM
PYTHONPATH=src python src/generative_place_categorization/main_relationships.py --method llm --llm-request --geometric-threshold 0.7 0.7

# Method 2. LVLM
PYTHONPATH=src python src/generative_place_categorization/main_relationships.py --method lvlm --llm-request --num-images 2 --geometric-threshold 0.7 --number-maps 5