#!/bin/bash

e_list=(1.2 1.3 1.4 1.5)
m_list=(2)
w_list=(0.45 0.55 0.65)
# mgt_list=(0 2 2.5 3 3.5)
# mst_list=(0.2 0.3 0.4 0.5)
# sst_list=(2.0 2.5 3.0 3.5 100)

# geometric
# (DBSCAN)
for e in "${e_list[@]}"; do
    for m in "${m_list[@]}"; do
        echo "Running model: geometric with c=dbscan, e=$e, m=$m"
        python src/main.py -p -n 10 --method geometric -c dbscan -e $e -m $m -r pca
    done
done
# # (HDBSCAN)
# echo "Running model: geometric with c=hdbscan"
# python src/main.py -p -n 10 --method geometric -c hdbscan -r pca

# bert, roberta, deepseek+sbert
for w in "${w_list[@]}"; do
    # (DBSCAN)
    for e in "${e_list[@]}"; do
        for m in "${m_list[@]}"; do
            # bert
            echo "Running model: bert with c=dbscan, e=$e, m=$m, w=$w"
            python src/main.py -p -n 10 --method bert -c dbscan -e $e -m $m -w $w -d 3 -r pca

            # roberta
            echo "Running model: roberta with c=dbscan, e=$e, m=$m, w=$w"
            python src/main.py -p -n 10 --method roberta -c dbscan -e $e -m $m -w $w -d 3 -r pca

            # deepseek+sbert
            echo "Running model: deepseek+sbert with c=dbscan, e=$e, m=$m, w=$w"
            python src/main.py -p -n 10 --method deepseek+sbert -c dbscan -e $e -m $m -w $w -d 3 -r pca
        done
    done

    # # (HDBSCAN)
    # echo "Running model: bert with c=hdbscan, w=$w"
    # python src/main.py -p -n 10 --method bert -c hdbscan -w $w -d 3 -r pca

    # # roberta
    # echo "Running model: roberta with c=hdbscan, w=$w"
    # python src/main.py -p -n 10 --method roberta -c hdbscan -w $w -d 3 -r pca

    # # deepseek+bert
    # echo "Running model: deepseek+sbert with c=hdbscan, w=$w"
    # python src/main.py -p -n 10 --method deepseek+sbert -c hdbscan -w $w -d 3 -r pca
done

# # bert+post, deepseek+sbert+post
# for mgt in "${mgt_list[@]}"; do
#     for mst in "${mst_list[@]}"; do
#         for sst in "${sst_list[@]}"; do
#             for w in "${w_list[@]}"; do
#                 # (DBSCAN)
#                 for e in "${e_list[@]}"; do
#                     for m in "${m_list[@]}"; do
#                         # bert+post
#                         echo "Running model: bert+post with w=$w, c=dbscan, e=$e, m=$m"
#                         python src/main.py -p -n 10 --method bert+post -c dbscan -e $e -m $m -w $w -d 3 \
#                         --merge-geometric-threshold $mgt --merge-semantic-threshold $mst -r pca

#                         # deepseek+sbert+post
#                         echo "Running model: deepseek+sbert+post with w=$w, c=dbscan, e=$e, m=$m, w=$w"
#                         python src/main.py -p -n 10 --method deepseek+sbert+post -c dbscan -e $e -m $m -w $w -d 3 \
#                         --merge-geometric-threshold $mgt --merge-semantic-threshold $mst -r pca
#                     done
#                 done

#                 # (HDBSCAN)
#                 # bert+post
#                 echo "Running model: bert+post with w=$w, c=hdbscan, w=$w"
#                 python src/main.py -p -n 10 --method bert+post -c hdbscan -w $w -d 3 \
#                 --merge-geometric-threshold $mgt --merge-semantic-threshold $mst --split-semantic-threshold $sst -r pca

#                 # deepseek+sbert+post
#                 echo "Running model: deepseek+sbert+post with w=$w, c=hdbscan, w=$w"
#                 python src/main.py -p -n 10 --method deepseek+sbert+post -c hdbscan -w $w -d 3 \
#                 --merge-geometric-threshold $mgt --merge-semantic-threshold $mst --split-semantic-threshold $sst -r pca
#             done
#         done
#     done
# done