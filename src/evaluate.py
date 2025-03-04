
import argparse
import os

import constants
from utils import file_utils
from voxeland.clustering import ClusteringResult


def main(args):

    # Load ground-truth clustering results
    ground_truth_clustering_results = dict()
    for semantic_map_clustering_file_name in sorted(os.listdir(constants.CLUSTERINGS_FOLDER_PATH)):
        semantic_map_basename = file_utils.get_file_basename(
            semantic_map_clustering_file_name)

        ground_truth_clustering_results[semantic_map_basename] = ClusteringResult.load_from_json(
            os.path.join(constants.CLUSTERINGS_FOLDER_PATH,
                         semantic_map_clustering_file_name))

    # Load method clustering results
    methods_clustering_results = dict()
    for method in file_utils.list_subdirectories(constants.RESULTS_FOLDER_PATH):
        methods_clustering_results[method] = dict()
        for semantic_map in file_utils.list_subdirectories(os.path.join(constants.RESULTS_FOLDER_PATH,
                                                                        method)):
            methods_clustering_results[method][semantic_map] = ClusteringResult.load_from_json(
                os.path.join(constants.RESULTS_FOLDER_PATH,
                             method,
                             semantic_map,
                             "clustering.json"))

    # Evaluate clustering results
    for method in sorted(methods_clustering_results):
        for semantic_map in sorted(methods_clustering_results[method]):
            print("evaluating", method, semantic_map)

            ground_truth_cr: ClusteringResult = ground_truth_clustering_results[semantic_map]
            method_cr: ClusteringResult = methods_clustering_results[method][semantic_map]

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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Evaluates different clusterings")

    args = parser.parse_args()

    main(args)
