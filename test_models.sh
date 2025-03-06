#!/bin/bash

# none
python src/main.py -p -n 10 -s none -e 1 -m 1
python src/main.py -p -n 10 -s none -e 1.1 -m 1
python src/main.py -p -n 10 -s none -e 1.2 -m 1
python src/main.py -p -n 10 -s none -e 1.3 -m 1

# # bert
python src/main.py -p -n 10 -s bert -w 0.005 -d 3 -e 1 -m 1
python src/main.py -p -n 10 -s bert -w 0.005 -d 5 -e 1 -m 1
python src/main.py -p -n 10 -s bert -w 0.005 -d 8 -e 1 -m 1
python src/main.py -p -n 10 -s bert -w 0.005 -e 1 -m 1
python src/main.py -p -n 10 -s bert -w 0.005 -d 3 -e 1.1 -m 1
python src/main.py -p -n 10 -s bert -w 0.005 -d 5 -e 1.1 -m 1
python src/main.py -p -n 10 -s bert -w 0.005 -d 8 -e 1.1 -m 1
python src/main.py -p -n 10 -s bert -w 0.005 -e 1.1 -m 1
python src/main.py -p -n 10 -s bert -w 0.005 -d 3 -e 1.2 -m 1
python src/main.py -p -n 10 -s bert -w 0.005 -d 5 -e 1.2 -m 1
python src/main.py -p -n 10 -s bert -w 0.005 -d 8 -e 1.2 -m 1
python src/main.py -p -n 10 -s bert -w 0.005 -e 1.2 -m 1

# # roberta
python src/main.py -p -n 10 -s roberta -w 0.005 -d 3 -e 1 -m 1
python src/main.py -p -n 10 -s roberta -w 0.005 -d 5 -e 1 -m 1
python src/main.py -p -n 10 -s roberta -w 0.005 -d 8 -e 1 -m 1
python src/main.py -p -n 10 -s roberta -w 0.005 -e 1 -m 1
python src/main.py -p -n 10 -s roberta -w 0.005 -d 3 -e 1.1 -m 1
python src/main.py -p -n 10 -s roberta -w 0.005 -d 5 -e 1.1 -m 1
python src/main.py -p -n 10 -s roberta -w 0.005 -d 8 -e 1.1 -m 1
python src/main.py -p -n 10 -s roberta -w 0.005 -e 1.1 -m 1
python src/main.py -p -n 10 -s roberta -w 0.005 -d 3 -e 1.2 -m 1
python src/main.py -p -n 10 -s roberta -w 0.005 -d 5 -e 1.2 -m 1
python src/main.py -p -n 10 -s roberta -w 0.005 -d 8 -e 1.2 -m 1
python src/main.py -p -n 10 -s roberta -w 0.005 -e 1.2 -m 1

# # deepseek+bert
python src/main.py -p -n 10 -s deepseek+sbert -w 0.005 -d 3 -e 1 -m 1
python src/main.py -p -n 10 -s deepseek+sbert -w 0.005 -d 5 -e 1 -m 1
python src/main.py -p -n 10 -s deepseek+sbert -w 0.005 -d 8 -e 1 -m 1
python src/main.py -p -n 10 -s deepseek+sbert -w 0.005 -e 1 -m 1
python src/main.py -p -n 10 -s deepseek+sbert -w 0.005 -d 3 -e 1.1 -m 1
python src/main.py -p -n 10 -s deepseek+sbert -w 0.005 -d 5 -e 1.1 -m 1
python src/main.py -p -n 10 -s deepseek+sbert -w 0.005 -d 8 -e 1.1 -m 1
python src/main.py -p -n 10 -s deepseek+sbert -w 0.005 -e 1.1 -m 1
python src/main.py -p -n 10 -s deepseek+sbert -w 0.005 -d 3 -e 1.2 -m 1
python src/main.py -p -n 10 -s deepseek+sbert -w 0.005 -d 5 -e 1.2 -m 1
python src/main.py -p -n 10 -s deepseek+sbert -w 0.005 -d 8 -e 1.2 -m 1
python src/main.py -p -n 10 -s deepseek+sbert -w 0.005 -e 1.2 -m 1
python src/main.py -p -n 10 -s deepseek+sbert -w 0.005 -d 3 -e 1.3 -m 1
python src/main.py -p -n 10 -s deepseek+sbert -w 0.005 -d 5 -e 1.3 -m 1
python src/main.py -p -n 10 -s deepseek+sbert -w 0.005 -d 8 -e 1.3 -m 1
python src/main.py -p -n 10 -s deepseek+sbert -w 0.005 -e 1.3 -m 1