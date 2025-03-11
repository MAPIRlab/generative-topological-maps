
import argparse
import os
from typing import List


from utils import file_utils
import constants

from voxeland.clustering import Clustering
from voxeland.semantic_map import SemanticMap
from voxeland.semantic_map_object import SemanticMapObject


def main(args):

    # Load and pre-process semantic map
    semantic_maps: List[SemanticMap] = list()
    for semantic_map_file_name in sorted(os.listdir(constants.SEMANTIC_MAPS_FOLDER_PATH)):

        semantic_map_basename = file_utils.get_file_basename(
            semantic_map_file_name)
        # Load semantic map
        semantic_map_obj = file_utils.load_json(os.path.join(constants.SEMANTIC_MAPS_FOLDER_PATH,
                                                             semantic_map_file_name))
        # Create SemanticMap object
        semantic_maps.append(SemanticMap(semantic_map_basename,
                                         [SemanticMapObject(obj_id, obj_data) for obj_id, obj_data in semantic_map_obj["instances"].items()]))

    # For each semantic map
    for semantic_map in semantic_maps[:args.number_maps]:

        print("#"*40)
        print(f"Processing {semantic_map.get_semantic_map_id()}...")
        print("#"*40)

        # Load ground truth clustering result
        ground_truth_cr_file_path = os.path.join(constants.CLUSTERINGS_FOLDER_PATH,
                                                 f"{semantic_map.get_semantic_map_id()}.json")
        ground_truth_cr = Clustering.load_from_json(
            ground_truth_cr_file_path)

        # TODO: check that all objects are present
        ground_truth_cr_plot_file_path = os.path.join(constants.RESULTS_FOLDER_PATH,
                                                      "ground_truth",
                                                      f"{semantic_map.get_semantic_map_id()}.png")
        file_utils.create_directories_for_file(ground_truth_cr_plot_file_path)
        ground_truth_cr.visualize(
            f"Ground truth for {semantic_map.get_semantic_map_id()}",
            ground_truth_cr_plot_file_path,
            semantic_map)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Performs some checks on the ground truth, and represents the places")

    parser.add_argument("-n", "--number-maps",
                        help="Number of semantic map to which place categorization will be applied.",
                        type=int,
                        default=10)

    # SEMANTIC DESCRIPTOR parameters
    parser.add_argument("-s", "--semantic-descriptor",
                        help="How to compute the semantic descriptor.",
                        choices=[constants.METHOD_BERT, constants.METHOD_OPENAI,
                                 constants.METHOD_DEEPSEEK_SBERT, constants.METHOD_DEEPSEEK_OPENAI],
                        default=constants.METHOD_BERT)

    parser.add_argument("-w", "--semantic-weight",
                        help="Semantic weight in DBSCAN distance.",
                        type=float,
                        default=0.005)

    parser.add_argument("-d", "--semantic-dimension",
                        help="Dimensions to which reduce the semantic descriptor using PCA",
                        type=int,
                        default=None)

    # DBSCAN parameters
    parser.add_argument("-e", "--eps",
                        help="eps parameter in the DBSCAN algorithm",
                        type=int,
                        default=1)

    parser.add_argument("-m", "--min-samples",
                        help="min_samples parameter in the DBSCAN algorithm",
                        type=int,
                        default=2)

    args = parser.parse_args()

    main(args)
