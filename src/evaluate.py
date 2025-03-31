
import argparse
import os

import constants
from show.metrics_table import MetricsTable
from utils import file_utils
from voxeland.clustering import Clustering


def main(args):

    # Load ground-truth clustering results
    ground_truth_clustering_results = dict()
    for semantic_map_clustering_file_name in sorted(os.listdir(constants.CLUSTERINGS_FOLDER_PATH)):
        semantic_map_basename = file_utils.get_file_basename(
            semantic_map_clustering_file_name)

        ground_truth_clustering_results[semantic_map_basename] = Clustering.load_from_json(
            os.path.join(constants.CLUSTERINGS_FOLDER_PATH,
                         semantic_map_clustering_file_name))

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
    for method in sorted(methods_clustering_results):
        methods_metrics[method] = dict()

        for semantic_map in sorted(methods_clustering_results[method]):
            methods_metrics[method][semantic_map] = dict()

            ground_truth_cr: Clustering = ground_truth_clustering_results[semantic_map]
            method_cr: Clustering = methods_clustering_results[method][semantic_map]

            # Evaluate the clustering result against the ground truth
            evaluation_metrics = method_cr.evaluate_against_ground_truth(
                ground_truth_cr)

            # Print results
            print(f"Method: {method}, Semantic Map: {semantic_map}")
            print(
                f"\tAdjusted Rand Index (ARI): {evaluation_metrics['ARI']:.4f}")
            print(
                f"\tNormalized Mutual Information (NMI): {evaluation_metrics['NMI']:.4f}")
            print(f"\tV-Measure: {evaluation_metrics['V-Measure']:.4f}")
            print(
                f"\tFowlkes-Mallows Index (FMI): {evaluation_metrics['FMI']:.4f}")
            print("-" * 60)

            # Save results
            methods_metrics[method][semantic_map] = {
                "ARI": evaluation_metrics['ARI'],
                "NMI": evaluation_metrics['NMI'],
                "V-Measure": evaluation_metrics['V-Measure'],
                "FMI": evaluation_metrics['FMI'],
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
