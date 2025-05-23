
import argparse
import os
from typing import Dict, List

from generative_place_categorization import constants
from generative_place_categorization.utils import file_utils
from generative_place_categorization.voxeland.clustering import Clustering
from generative_place_categorization.voxeland.semantic_map import SemanticMap
from generative_place_categorization.voxeland.semantic_map_object import (
    SemanticMapObject,
)


def main(args):

    # Load and pre-process semantic map
    semantic_maps: Dict[str, SemanticMap] = dict()
    for semantic_map_file_name in sorted(os.listdir(constants.SEMANTIC_MAPS_FOLDER_PATH)):

        # Load semantic map
        semantic_map_basename = file_utils.get_file_basename(
            semantic_map_file_name)
        semantic_map_objects_dict = file_utils.load_json(os.path.join(constants.SEMANTIC_MAPS_FOLDER_PATH,
                                                                      semantic_map_file_name))
        # Create SemanticMap object
        semantic_map_objects: List[SemanticMapObject] = []
        for obj_id, obj_data in semantic_map_objects_dict["instances"].items():
            # Create SemanticMapObject object
            semantic_map_object = SemanticMapObject(obj_id, obj_data)

            semantic_map_objects.append(semantic_map_object)
        semantic_maps[semantic_map_basename] = SemanticMap(semantic_map_basename,
                                                           semantic_map_objects)

    # Load semantic map clusterings
    clusterings: Dict[str, List[Clustering]] = dict()
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
        if semantic_map_basename not in clusterings:
            clusterings[semantic_map_basename] = [Clustering.load_from_json(
                clustering_file_path, semantic_maps[semantic_map_basename])]
        else:
            clusterings[semantic_map_basename].append(
                Clustering.load_from_json(clustering_file_path, semantic_maps[semantic_map_basename]))

    # For each semantic map
    for semantic_map in list(semantic_maps.values())[:args.number_maps]:

        print("#"*40)
        print(f"Processing {semantic_map.semantic_map_id}...")
        print("#"*40)

        # For each clustreing possibility
        for option_id, ground_truth_cr in enumerate(clusterings[semantic_map.semantic_map_id]):

            print("#"*20)
            print(f"Option {option_id}...")
            print("#"*20)

            # Check that all objects in the map are present in the place segmentation/clustering result
            for object1 in semantic_map.get_all_objects():
                occurrences = 0
                for cluster in ground_truth_cr.clusters:
                    for object2 in cluster.objects:
                        if object1.object_id == object2.object_id:
                            occurrences += 1
                if occurrences != 1:
                    print(
                        f"Warning: Object {object1.object_id} in semantic map {semantic_map.semantic_map_id} appears {occurrences} times in the clustering ground truth.")

            # Visualize place segmentation
            ground_truth_cr_plot_file_path = os.path.join(constants.RESULTS_FOLDER_PATH,
                                                          "ground_truth",
                                                          f"{semantic_map.semantic_map_id}_opt{option_id}.png")
            file_utils.create_directories_for_file(
                ground_truth_cr_plot_file_path)
            ground_truth_cr.visualize_2D(
                f"Ground truth for {semantic_map.semantic_map_id} (option {option_id})",
                semantic_map,
                file_path=ground_truth_cr_plot_file_path)


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
