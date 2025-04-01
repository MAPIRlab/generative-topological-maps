#!/bin/bash

# geometric
python src/main.py -p -n 10 --method geometric -c dbscan -e 1.0 -m 2 -w 0 -r umap

# bert
python src/main.py -p -n 10 --method bert -c dbscan -e 1.0 -m 2 -w 0.15 -d 3 -r umap

# roberta
python src/main.py -p -n 10 --method roberta -c dbscan -e 1.0 -m 2 -w 0.15 -d 3 -r umap

# deepseek+bert
python src/main.py -p -n 10 --method deepseek+sbert -c dbscan -e 1.0 -m 2 -w 0.15 -d 3 -r umap

# bert+post
python src/main.py -p -n 10 --method bert+post -c dbscan -e 1.0 -m 2 -w 0.15 -d 3 \
    --merge-geometric-threshold 2.5 --merge-semantic-threshold 0.99 -r umap

# deepseek+sbert+post
python src/main.py -p -n 10 --method deepseek+sbert+post -c dbscan -e 1.0 -m 2 -w 0.15 -d 3 \
    --merge-geometric-threshold 2.5 --merge-semantic-threshold 0.99 -r umap

# deepseek
# python src/main.py -p -n 10 --method deepseek -r umap