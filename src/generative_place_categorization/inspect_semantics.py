import argparse
import os
from typing import List

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial.distance import cdist

from generative_place_categorization import constants
from generative_place_categorization.embedding.bert_embedder import BERTEmbedder
from generative_place_categorization.embedding.openai_embedder import OpenAIEmbedder
from generative_place_categorization.embedding.roberta_embedder import RoBERTaEmbedder
from generative_place_categorization.embedding.sentence_embedder import (
    SentenceBERTEmbedder,
)
from generative_place_categorization.llm.large_language_model import LargeLanguageModel
from generative_place_categorization.semantic.dimensionality_reduction_engine import (
    DimensionalityReductionEngine,
)
from generative_place_categorization.semantic.semantic_descriptor_engine import (
    SemanticDescriptorEngine,
)
from generative_place_categorization.utils import file_utils
from generative_place_categorization.voxeland.semantic_map import SemanticMap
from generative_place_categorization.voxeland.semantic_map_object import (
    SemanticMapObject,
)

load_dotenv()


# Models
bert_embedder = None
roberta_embedder = None
openai_embedder = None
sbert_embedder = None
deepseek_llm = None


def plot_2d(object_set, embeddings_2d, closest_neighbors, colors, semantic_descriptor, dim_reductor):
    fig, ax = plt.subplots()

    for i in range(len(object_set)):
        ax.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1],
                   color=colors[i], label=object_set[i])

    for i, neighbor_idx in enumerate(closest_neighbors):
        distance = np.linalg.norm(
            embeddings_2d[i] - embeddings_2d[neighbor_idx])
        num_segments = min(5 + i, 20)
        if distance > 1.0:
            num_segments += 5

        x_values = np.linspace(
            embeddings_2d[i, 0], embeddings_2d[neighbor_idx, 0], num=num_segments)
        y_values = np.linspace(
            embeddings_2d[i, 1], embeddings_2d[neighbor_idx, 1], num=num_segments)

        for j in range(0, len(x_values) - 1, 2):
            ax.plot([x_values[j], x_values[j + 1]],
                    [y_values[j], y_values[j + 1]],
                    linestyle='dashed', linewidth=1.5, color=colors[i], alpha=0.7)

    for i, obj in enumerate(object_set):
        x, y = embeddings_2d[i]
        ax.text(x, y, obj)

    plt.title(
        f"semantic_descriptor={semantic_descriptor}, dim_reductor={dim_reductor}")
    plt.show()


def plot_3d(object_set, embeddings_3d, closest_neighbors, colors, semantic_descriptor, dim_reductor):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(len(object_set)):
        ax.scatter(embeddings_3d[i, 0], embeddings_3d[i, 1], embeddings_3d[i, 2],
                   color=colors[i], label=object_set[i])

    segments = []
    segment_colors = []
    for i, neighbor_idx in enumerate(closest_neighbors):
        distance = np.linalg.norm(
            embeddings_3d[i] - embeddings_3d[neighbor_idx])
        num_segments = min(5 + i, 20)
        if distance > 1.0:
            num_segments += 5

        x_values = np.linspace(
            embeddings_3d[i, 0], embeddings_3d[neighbor_idx, 0], num=num_segments)
        y_values = np.linspace(
            embeddings_3d[i, 1], embeddings_3d[neighbor_idx, 1], num=num_segments)
        z_values = np.linspace(
            embeddings_3d[i, 2], embeddings_3d[neighbor_idx, 2], num=num_segments)

        for j in range(0, len(x_values) - 1, 2):
            segments.append([(x_values[j], y_values[j], z_values[j]),
                             (x_values[j + 1], y_values[j + 1], z_values[j + 1])])
            segment_colors.append(colors[i])

    line_collection = Line3DCollection(
        segments, colors=segment_colors, linewidths=1.5, alpha=0.7)
    ax.add_collection3d(line_collection)

    for i, obj in enumerate(object_set):
        x, y, z = embeddings_3d[i]
        ax.text(x, y, z, obj)

    plt.title(
        f"semantic_descriptor={semantic_descriptor}, dim_reductor={dim_reductor}")
    plt.show()


def main(args, object_set: List[str]):
    semantic_descriptor_engine = SemanticDescriptorEngine(
        bert_embedder, roberta_embedder, openai_embedder, sbert_embedder, deepseek_llm)

    # Instantiate dimensionality reduction engine
    dim_reduction_engine = DimensionalityReductionEngine()

    semantic_descriptors = []
    if args.semantic_descriptor == constants.SEMANTIC_DESCRIPTOR_ALL:
        semantic_descriptors = [
            constants.SEMANTIC_DESCRIPTOR_BERT,
            constants.SEMANTIC_DESCRIPTOR_ROBERTA,
            constants.SEMANTIC_DESCRIPTOR_OPENAI,
            constants.SEMANTIC_DESCRIPTOR_DEEPSEEK_SBERT,
            constants.SEMANTIC_DESCRIPTOR_DEEPSEEK_OPENAI]
    else:
        semantic_descriptors = [args.semantic_descriptor]

    for semantic_descriptor in semantic_descriptors:
        print(
            f"Showing word embeddings for semantic descriptor {semantic_descriptor}")

        # Get embeddings for each object
        embeddings = []
        for obj in object_set:
            emb = semantic_descriptor_engine.get_semantic_descriptor(
                semantic_descriptor, obj)
            embeddings.append(emb)

        # Convert to NumPy array (assuming each emb is 1D or can be flattened to 1D)
        embeddings = np.array(embeddings)

        # Apply dimensionality reduction
        embeddings_2_or_3d = dim_reduction_engine.reduce(
            embeddings, target_dimension=args.semantic_dimension, method=args.dimensionality_reductor)

        # Compute pairwise distances and find the closest neighbor for each point
        # Compute all pairwise distances
        distances = cdist(embeddings_2_or_3d, embeddings_2_or_3d)
        np.fill_diagonal(distances, np.inf)  # Avoid self-matching
        # Get the index of the closest neighbor
        closest_neighbors = np.argmin(distances, axis=1)

        # Generate unique colors for each point using a colormap
        norm = mcolors.Normalize(vmin=0, vmax=len(object_set) - 1)
        # Using 'tab10' for distinct colors
        colormap = cm.get_cmap('tab10', len(object_set))
        colors = [colormap(norm(i)) for i in range(len(object_set))]

        # PLOT
        if args.semantic_dimension == 2:
            plot_2d(object_set, embeddings_2_or_3d, closest_neighbors, colors,
                    semantic_descriptor, args.dimensionality_reductor)
        else:
            plot_3d(object_set, embeddings_2_or_3d, closest_neighbors, colors,
                    semantic_descriptor, args.dimensionality_reductor)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Inspects the semantic descriptors of a set of objects."
    )

    # SEMANTIC DESCRIPTOR parameters
    parser.add_argument(
        "-s",
        "--semantic-descriptor",
        help="How to compute the semantic descriptor.",
        choices=[
            constants.SEMANTIC_DESCRIPTOR_ALL,
            constants.SEMANTIC_DESCRIPTOR_BERT,
            constants.SEMANTIC_DESCRIPTOR_ROBERTA,
            constants.SEMANTIC_DESCRIPTOR_OPENAI,
            constants.SEMANTIC_DESCRIPTOR_DEEPSEEK_SBERT,
            constants.SEMANTIC_DESCRIPTOR_DEEPSEEK_OPENAI,
        ],
        required=True,
    )

    parser.add_argument("-d", "--semantic-dimension",
                        help="Dimensions to which reduce the semantic descriptor using PCA",
                        type=int,
                        choices=[2, 3],
                        default=3)

    parser.add_argument("-r", "--dimensionality-reductor",
                        help="Dimensionality reduction method to apply to the semantic descriptor.",
                        choices=[constants.DIM_REDUCTOR_PCA,
                                 constants.DIM_REDUCTOR_UMAP],
                        default=constants.DIM_REDUCTOR_PCA)

    parser.add_argument("-o",
                        "--object-set",
                        help="A set of objects for processing, where each object contains one word or two words enclosed in double quotes.",
                        nargs="+",
                        required=False)

    parser.add_argument("--semantic-map",
                        help="Path to the semantic map file to process objects",
                        required=False)

    args = parser.parse_args()

    # Ensure either object_set or semantic_map is provided, but not both
    if (args.object_set and args.semantic_map) or (not args.object_set and not args.semantic_map):
        raise ValueError(
            "You must provide either --object-set or --semantic-map, but not both.")

    object_set = []
    # If OBJECT_SET
    if args.object_set:
        if args.object_set:
            for obj in args.object_set:
                if obj.startswith('"') and obj.endswith('"'):
                    object_set.append(obj.strip('"'))
                else:
                    object_set.append(obj)

    # If SEMANTIC_MAP
    # Load and pre-process semantic map
    elif args.semantic_map:
        semantic_map_dict = file_utils.load_json(os.path.join(constants.SEMANTIC_MAPS_FOLDER_PATH,
                                                              f"{args.semantic_map}.json"))
        semantic_map = SemanticMap(args.semantic_map,
                                   [SemanticMapObject(obj_id, obj_data) for obj_id, obj_data in semantic_map_dict["instances"].items()])
        object_set = list(map(
            lambda object: object.get_most_probable_class(), semantic_map.get_all_objects()))

    print(f"Object set: {args.object_set}")

    # Initialize fast models
    bert_embedder = BERTEmbedder()
    roberta_embedder = RoBERTaEmbedder()
    openai_embedder = OpenAIEmbedder()
    sbert_embedder = SentenceBERTEmbedder(
        model_id="sentence-transformers/all-mpnet-base-v2")
    deepseek_llm = LargeLanguageModel(model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                                      cache_path=constants.LLM_CACHE_FILE_PATH)

    main(args, object_set)
