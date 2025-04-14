
import argparse
import os
from typing import Dict, List

import numpy as np
import tqdm

import constants
from show.metrics_table import MetricsTable
from utils import file_utils
from voxeland.clustering import Clustering


def main(args):

    # Load ground-truth clustering results
    ground_truth_clustering_results: Dict[str, List[Clustering]] = dict()
    for clustering_file_name in sorted(os.listdir(constants.CLUSTERINGS_FOLDER_PATH)):
        # Get basename and split semantic map and possibility
        clustering_basename = file_utils.get_file_basename(
            clustering_file_name)
        semantic_map_basename, option_id = clustering_basename.rsplit(
            "_opt", 1)
        option_id = int(option_id)

        # Save clustering into dictionary
        clustering_file_path = os.path.join(
            constants.CLUSTERINGS_FOLDER_PATH, clustering_file_name)
        if semantic_map_basename not in ground_truth_clustering_results:
            ground_truth_clustering_results[semantic_map_basename] = [Clustering.load_from_json(
                clustering_file_path)]
        else:
            ground_truth_clustering_results[semantic_map_basename].append(
                Clustering.load_from_json(clustering_file_path))

    # Load method clustering results
    methods_clustering_results = dict()
    for method in file_utils.list_subdirectories(os.path.join(constants.RESULTS_FOLDER_PATH,
                                                              "method_results")):
        methods_clustering_results[method] = dict()
        for semantic_map in file_utils.list_subdirectories(os.path.join(constants.RESULTS_FOLDER_PATH,
                                                                        "method_results",
                                                                        method)):
            methods_clustering_results[method][semantic_map] = Clustering.load_from_json(
                os.path.join(constants.RESULTS_FOLDER_PATH,
                             "method_results",
                             method,
                             semantic_map,
                             "clustering.json"))

    # Evaluate clustering results
    methods_metrics = dict()
    for method in tqdm.tqdm(sorted(methods_clustering_results),
                            desc="Evaluating clustering results", unit="method"):
        methods_metrics[method] = dict()

        for semantic_map in sorted(methods_clustering_results[method]):
            methods_metrics[method][semantic_map] = dict()

            best_evaluation_metrics = {
                "ARI": -1,
                "NMI": -1,
                "V-Measure": -1,
                "FMI": -1,
            }

            # Get the method clustering result
            method_cr: Clustering = methods_clustering_results[method][semantic_map]

            # Iterate over the ground truth clustering results
            for option_id, ground_truth_cr in enumerate(ground_truth_clustering_results[semantic_map]):

                # Evaluate the clustering result against the ground truth
                evaluation_metrics = method_cr.evaluate_against_ground_truth(
                    ground_truth_cr)

                if np.mean(list(evaluation_metrics.values())) > np.mean(list(best_evaluation_metrics.values())):
                    # print(f"La opción {option_id} es mejor!")
                    best_evaluation_metrics = evaluation_metrics

            # Print results
            # print(f"Method: {method}, Semantic Map: {semantic_map}")
            # print(
            #     f"\tAdjusted Rand Index (ARI): {best_evaluation_metrics['ARI']:.4f}")
            # print(
            #     f"\tNormalized Mutual Information (NMI): {best_evaluation_metrics['NMI']:.4f}")
            # print(f"\tV-Measure: {best_evaluation_metrics['V-Measure']:.4f}")
            # print(
            #     f"\tFowlkes-Mallows Index (FMI): {best_evaluation_metrics['FMI']:.4f}")
            # print("-" * 60)

            # Save results
            methods_metrics[method][semantic_map] = {
                "ARI": best_evaluation_metrics['ARI'],
                "NMI": best_evaluation_metrics['NMI'],
                "V-Measure": best_evaluation_metrics['V-Measure'],
                "FMI": best_evaluation_metrics['FMI'],
            }

    # RESULT 1: Metrics Table
    metrics_table = MetricsTable(methods_metrics)
    # metrics_table.display_best(10, "V-Measure")
    metrics_table.display_best(10, "ARI", ["Method"])
    metrics_table.display_best(10, "NMI", ["Method"])
    metrics_table.display_best(10, "V-Measure", ["Method"])
    metrics_table.display_best(10, "FMI", ["Method"])
    # metrics_table.display_best(10, "ARI", ["Method", "Dataset"])
    # metrics_table.display_best(10, "NMI", ["Method", "Dataset"])
    # metrics_table.display_best(10, "V-Measure", ["Method", "Dataset"])
    # metrics_table.display_best(10, "FMI", ["Method", "Dataset"])
    # metrics_table.filter_methods("none_", ["Method"])
    # metrics_table.filter_methods("none_")
    metrics_table.filter_dataset("scannet", group_by=["Method"])
    metrics_table.filter_dataset("scenenn", group_by=["Method"])
    metrics_table.semantic_map_vs_method_pivot_table("ARI")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Evaluates different clusterings")

    args = parser.parse_args()

    main(args)
