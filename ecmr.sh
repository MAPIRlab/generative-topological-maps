for eps in 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0; do
    echo "PLACE SEGMENTATION; method: geometric; epsilon=$eps"
    PYTHONPATH=src python src/generative_place_categorization/main_places.py \
        -p --stage segmentation -n 10 \
        --method geometric \
        --clustering-algorithm dbscan \
        --dimensionality_reductor pca \
        -e $eps -m 2

    echo "PLACE SEGMENTATION; method: bert; epsilon=$eps"
    PYTHONPATH=src python src/generative_place_categorization/main_places.py \
        -p --stage segmentation -n 10 \
        --method bert \
        --clustering-algorithm dbscan \
        --semantic-weight 0.55 \
        --semantic-dimension 3 \
        --dimensionality_reductor pca \
        -e $eps -m 2

    echo "PLACE SEGMENTATION; method: roberta; epsilon=$eps"
    PYTHONPATH=src python src/generative_place_categorization/main_places.py \
        -p --stage segmentation -n 10 \
        --method roberta \
        --clustering-algorithm dbscan \
        --semantic-weight 0.55 \
        --semantic-dimension 3 \
        --dimensionality_reductor pca \
        -e $eps -m 2

    echo "PLACE SEGMENTATION; method: llm+sbert; epsilon=$eps"
    PYTHONPATH=src python src/generative_place_categorization/main_places.py \
        -p --stage segmentation -n 10 \
        --method llm+sbert \
        --clustering-algorithm dbscan \
        --semantic-weight 0.55 \
        --semantic-dimension 3 \
        --dimensionality_reductor pca \
        -e $eps -m 2
done
