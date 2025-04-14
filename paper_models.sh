#!/bin/bash

e=1.2
m=2
w=0.40

# geometric
echo "Running model: geometric"
python src/main.py -p -n 10 --method geometric -c dbscan -e $e -m $m -r pca

# # bert
echo "Running model: bert"
python src/main.py -p -n 10 --method bert -c dbscan -e $e -m $m -w $w -d 3 -r pca

# # roberta
echo "Running model: roberta"
python src/main.py -p -n 10 --method roberta -c dbscan -e $e -m $m  -w $w -d 3 -r pca

# # deepseek+bert
echo "Running model: deepseek+sbert"
python src/main.py -p -n 10 --method deepseek+sbert -c dbscan -e $e -m $m  -w $w -d 3 -r pca

