import argparse
import os

from generative_topological_maps import constants
from generative_topological_maps.utils import file_utils
from generative_topological_maps.voxeland.clustering import Clustering
from generative_topological_maps.voxeland.semantic_map import SemanticMap
from generative_topological_maps.voxeland.semantic_map_object import (
    SemanticMapObject,
)


def main(args):
    # Load semantic map
    semantic_map_dict = file_utils.load_json(
        os.path.join(constants.SEMANTIC_MAPS_FOLDER_PATH,
                     f"{args.semantic_map}.json")
    )
    semantic_map = SemanticMap(
        args.semantic_map,
        [
            SemanticMapObject(obj_id, obj_data)
            for obj_id, obj_data in semantic_map_dict["instances"].items()
        ],
    )

    # Load clustering
    clustering = Clustering.load_from_json(
        args.clustering_file, semantic_map=semantic_map)

    # Plot based on visualization method
    if args.visualization_method == constants.VISUALIZATION_METHOD_2D:
        clustering.visualize_2D(
            title="",
            file_path=args.output_file,
            semantic_map=semantic_map
        )
    elif args.visualization_method == constants.VISUALIZATION_METHOD_3D:
        clustering.visualize_3D(
            title="",
            file_path=args.output_file,
            semantic_map=semantic_map
        )
    elif args.visualization_method == constants.VISUALIZATION_METHOD_POINT_CLOUD:
        clustering.visualize_in_point_cloud(
            ply_path=args.ply_path,
            semantic_map=semantic_map,
            show_axes=True
        )
    else:
        raise ValueError("Only 2D and 3D visualizations are supported.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize clusters in 2D or 3D from a clustering file."
    )
    parser.add_argument(
        "-c",
        "--clustering-file",
        help="Path to the clustering.json file to visualize.",
        required=True,
    )
    parser.add_argument(
        "-s",
        "--semantic-map",
        help="Name of the semantic map file (without extension) to load.",
        required=True,
    )
    parser.add_argument(
        "-v",
        "--visualization-method",
        choices=[constants.VISUALIZATION_METHOD_2D, constants.VISUALIZATION_METHOD_3D,
                 constants.VISUALIZATION_METHOD_POINT_CLOUD],
        required=True,
        help="Method of visualization: 2D, 3D, or point Cloud.",
    )
    parser.add_argument(
        "-p",
        "--ply-path",
        help="Path to the PLY file for point cloud visualization. Required if visualization method is 'point_cloud'.",
        required=False
    )
    parser.add_argument(
        "-o",
        "--output-file",
        help="Path to save the visualization image. If not specified, the visualization will be displayed on the screen.",
        required=False,
    )
    args = parser.parse_args()

    if args.visualization_method == constants.VISUALIZATION_METHOD_POINT_CLOUD and not args.ply_path:
        parser.error(
            "The --ply-path argument is required when using point cloud visualization.")

    main(args)
