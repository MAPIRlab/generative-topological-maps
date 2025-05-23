import argparse
import os

import matplotlib.cm as cm
import matplotlib.colors as mcolors

from generative_place_categorization import constants
from generative_place_categorization.utils import file_utils
from generative_place_categorization.voxeland.clustering import Clustering
from generative_place_categorization.voxeland.semantic_map import SemanticMap
from generative_place_categorization.voxeland.semantic_map_object import (
    SemanticMapObject,
)


def main(args):
    # Load clustering
    clustering = Clustering.load_from_json(args.clustering_file)

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

    # Generate unique colors for each cluster
    norm = mcolors.Normalize(vmin=0, vmax=len(clustering.clusters) - 1)
    colormap = cm.get_cmap("tab10", len(clustering.clusters))
    colors = [colormap(norm(i)) for i in range(len(clustering.clusters))]

    # Plot based on dimensions
    if args.dimensions == 2:
        clustering.visualize_2D(
            title="2D Cluster Visualization",
            file_path=args.output_file,
            semantic_map=semantic_map
        )
    elif args.dimensions == 3:
        clustering.visualize_3D(
            title="3D Cluster Visualization",
            file_path=args.output_file,
            semantic_map=semantic_map
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
        "-d",
        "--dimensions",
        help="Number of dimensions for visualization (2 or 3).",
        type=int,
        choices=[2, 3],
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output-file",
        help="Path to save the visualization image. If not specified, the visualization will be displayed on the screen.",
        required=False,
    )
    args = parser.parse_args()
    main(args)
