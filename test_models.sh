#!/bin/bash

# # geometric
# python src/main.py -p -n 10 -s geometric -e 1.0 -m 1 -w 0

# # bert
# python src/main.py -p -n 10 -s bert -e 1.0 -m 1 -w 0.009 -d 3 

# # roberta
# python src/main.py -p -n 10 -s roberta -e 1.0 -m 1 -w 0.009 -d 3

# # deepseek+bert
# python src/main.py -p -n 10 -s deepseek+sbert -e 1.0 -m 1 -w 0.009 -d 3

# python src/main.py -p -n 10 -s deepseek+sbert+post -e 1.0 -m 1 -w 0.009 -d 3 \
#     --merge-geometric-threshold 2.5 --merge-semantic-threshold 0.97

# python src/main.py -p -n 10 -s deepseek+sbert+post -e 1.0 -m 1 -w 0.009 -d 3 \
#     --merge-geometric-threshold 2.0 --merge-semantic-threshold 0.97

# python src/main.py -p -n 10 -s deepseek+sbert+post -e 1.0 -m 1 -w 0.009 -d 3 \
#     --merge-geometric-threshold 1.5 --merge-semantic-threshold 0.97

# python src/main.py -p -n 10 -s deepseek+sbert+post -e 1.0 -m 1 -w 0.009 -d 3 \
#     --merge-geometric-threshold 1.0 --merge-semantic-threshold 0.97

# python src/main.py -p -n 10 -s deepseek+sbert+post -e 1.0 -m 1 -w 0.009 -d 3 \
#     --merge-geometric-threshold 2.5 --merge-semantic-threshold 0.98

# python src/main.py -p -n 10 -s deepseek+sbert+post -e 1.0 -m 1 -w 0.009 -d 3 \
#     --merge-geometric-threshold 2.0 --merge-semantic-threshold 0.98

# python src/main.py -p -n 10 -s deepseek+sbert+post -e 1.0 -m 1 -w 0.009 -d 3 \
#     --merge-geometric-threshold 1.5 --merge-semantic-threshold 0.98

# python src/main.py -p -n 10 -s deepseek+sbert+post -e 1.0 -m 1 -w 0.009 -d 3 \
#     --merge-geometric-threshold 1.0 --merge-semantic-threshold 0.98

# python src/main.py -p -n 10 -s deepseek+sbert+post -e 1.0 -m 1 -w 0.009 -d 3 \
#     --merge-geometric-threshold 2.5 --merge-semantic-threshold 0.99

# python src/main.py -p -n 10 -s deepseek+sbert+post -e 1.0 -m 1 -w 0.009 -d 3 \
#     --merge-geometric-threshold 2.0 --merge-semantic-threshold 0.99

# python src/main.py -p -n 10 -s deepseek+sbert+post -e 1.0 -m 1 -w 0.009 -d 3 \
#     --merge-geometric-threshold 1.5 --merge-semantic-threshold 0.99

# python src/main.py -p -n 10 -s deepseek+sbert+post -e 1.0 -m 1 -w 0.009 -d 3 \
#     --merge-geometric-threshold 1.0 --merge-semantic-threshold 0.99

# python src/main.py -p -n 10 -s bert+post -e 1.0 -m 1 -w 0.009 -d 3 \
#     --merge-geometric-threshold 2.5 --merge-semantic-threshold 0.97

# python src/main.py -p -n 10 -s bert+post -e 1.0 -m 1 -w 0.009 -d 3 \
#     --merge-geometric-threshold 2.0 --merge-semantic-threshold 0.97

# python src/main.py -p -n 10 -s bert+post -e 1.0 -m 1 -w 0.009 -d 3 \
#     --merge-geometric-threshold 1.5 --merge-semantic-threshold 0.97

# python src/main.py -p -n 10 -s bert+post -e 1.0 -m 1 -w 0.009 -d 3 \
#     --merge-geometric-threshold 1.0 --merge-semantic-threshold 0.97

# python src/main.py -p -n 10 -s bert+post -e 1.0 -m 1 -w 0.009 -d 3 \
#     --merge-geometric-threshold 2.5 --merge-semantic-threshold 0.98

# python src/main.py -p -n 10 -s bert+post -e 1.0 -m 1 -w 0.009 -d 3 \
#     --merge-geometric-threshold 2.0 --merge-semantic-threshold 0.98

# python src/main.py -p -n 10 -s bert+post -e 1.0 -m 1 -w 0.009 -d 3 \
#     --merge-geometric-threshold 1.5 --merge-semantic-threshold 0.98

# python src/main.py -p -n 10 -s bert+post -e 1.0 -m 1 -w 0.009 -d 3 \
#     --merge-geometric-threshold 1.0 --merge-semantic-threshold 0.98

# python src/main.py -p -n 10 -s bert+post -e 1.0 -m 1 -w 0.009 -d 3 \
#     --merge-geometric-threshold 2.5 --merge-semantic-threshold 0.99

# python src/main.py -p -n 10 -s bert+post -e 1.0 -m 1 -w 0.009 -d 3 \
#     --merge-geometric-threshold 2.0 --merge-semantic-threshold 0.99

# python src/main.py -p -n 10 -s bert+post -e 1.0 -m 1 -w 0.009 -d 3 \
#     --merge-geometric-threshold 1.5 --merge-semantic-threshold 0.99

# python src/main.py -p -n 10 -s bert+post -e 1.0 -m 1 -w 0.009 -d 3 \
#     --merge-geometric-threshold 1.0 --merge-semantic-threshold 0.99

python src/main.py -p -n 10 -s deepseek+sbert+post -e 1.0 -m 1 -w 0.009 -d 3 \
    --merge-geometric-threshold 2.5 --merge-semantic-threshold 0.97