import argparse
import os
import sys
from typing import List

import constants
import numpy as np
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from generative_topological_maps.embedding.all_mpnet_base_v2_embedder import (
    AllMpnetBaseV2Embedder,
)
from generative_topological_maps.embedding.bert_embedder import BERTEmbedder
from generative_topological_maps.embedding.openai_embedder import OpenAIEmbedder
from generative_topological_maps.embedding.roberta_embedder import RoBERTaEmbedder
from generative_topological_maps.llm.gemini_provider import GeminiProvider
from generative_topological_maps.llm.huggingface_large_language_model import (
    HuggingfaceLargeLanguageModel,
)
from generative_topological_maps.prompt.conversation_history import (
    ConversationHistory,
)
from generative_topological_maps.prompt.place_categorizer_prompt import (
    PlaceCategorizerPrompt,
)
from generative_topological_maps.prompt.place_segmenter_prompt import (
    PlaceSegmenterPrompt,
)
from generative_topological_maps.semantic.clustering_engine import ClusteringEngine
from generative_topological_maps.semantic.dimensionality_reduction_engine import (
    DimensionalityReductionEngine,
)
from generative_topological_maps.semantic.semantic_descriptor_engine import (
    SemanticDescriptorEngine,
)
from generative_topological_maps.utils import file_utils
from generative_topological_maps.voxeland.cluster import Cluster
from generative_topological_maps.voxeland.clustering import Clustering
from generative_topological_maps.voxeland.semantic_map import SemanticMap


def get_results_path_for_method(args):
    base_path = os.path.join(constants.PLACES_RESULTS_FOLDER_PATH)
    method_specific_path = f"{args.method}"

    def clustering_suffix():
        suffix = f"_ca{args.clustering_algorithm}"
        if args.clustering_algorithm != "hdbscan":
            suffix += f"_e{args.eps}_m{args.min_samples}"
        return suffix

    if args.method in (constants.METHOD_BERT_POST, constants.METHOD_LLM_SBERT_POST):
        method_specific_path += (
            clustering_suffix() +
            f"_w{args.semantic_weight}"
            f"_r{args.dimensionality_reductor}"
            f"_d{args.semantic_dimension}"
            f"_mgt_{args.merge_geometric_threshold}"
            f"_mst_{args.merge_semantic_threshold}"
            f"_sst_{args.split_semantic_threshold}"
        )
    elif args.method == constants.METHOD_GEOMETRIC:
        method_specific_path += clustering_suffix()
    elif args.method != constants.METHOD_LLM:
        method_specific_path += (
            clustering_suffix() +
            f"_w{args.semantic_weight}"
            f"_r{args.dimensionality_reductor}"
            f"_d{args.semantic_dimension}"
        )

    return os.path.join(base_path, method_specific_path)


def get_chart_title_for_method(args, semantic_map_id: str) -> str:
    """
    Build a chart title string based on the clustering method and its parameters.
    """
    # Start with the map identifier and method
    title = f"{semantic_map_id}\nmethod={args.method}"

    # Post-processed methods: include clustering algorithm first, then DBSCAN and thresholds
    if args.method in (constants.METHOD_BERT_POST, constants.METHOD_LLM_SBERT_POST):
        title += (
            f", clustering_algorithm={args.clustering_algorithm}"
            f", eps={args.eps}, min_samples={args.min_samples}"
            f", semantic_weight={args.semantic_weight}"
            f", dimensionality_reductor={args.dimensionality_reductor}"
            f", semantic_dimension={args.semantic_dimension}"
            f", merge_geometric_threshold={args.merge_geometric_threshold}"
            f", merge_semantic_threshold={args.merge_semantic_threshold}"
            f", split_semantic_threshold={args.split_semantic_threshold}"
        )
    # Pure geometric clustering: algorithm first, then eps/min_samples
    elif args.method == constants.METHOD_GEOMETRIC:
        title += (
            f", clustering_algorithm={args.clustering_algorithm}"
            f", eps={args.eps}, min_samples={args.min_samples}"
        )
    # Other non-LLM methods: algorithm first, then eps/min_samples and semantic params
    elif args.method != constants.METHOD_LLM:
        title += (
            f", clustering_algorithm={args.clustering_algorithm}"
            f", eps={args.eps}, min_samples={args.min_samples}"
            f", semantic_weight={args.semantic_weight}"
            f", dimensionality_reductor={args.dimensionality_reductor}"
            f", semantic_dimension={args.semantic_dimension}"
        )

    # For LLM-only method, no extra params
    return title


def main_segmentation(args):
    # Load environment variables
    load_dotenv()

    # Instantiate models
    bert_embedder = BERTEmbedder()
    roberta_embedder = RoBERTaEmbedder()
    openai_embedder = OpenAIEmbedder()
    sbert_embedder = AllMpnetBaseV2Embedder()
    # llm = GeminiProvider(
    #     credentials_file=constants.GOOGLE_GEMINI_CREDENTIALS_FILENAME,
    #     project_id=constants.GOOGLE_GEMINI_PROJECT_ID,
    #     project_location=constants.GOOGLE_GEMINI_PROJECT_LOCATION,
    #     model_name=GeminiProvider.GEMINI_2_0_FLASH,
    #     cache_path=constants.LLM_CACHE_FILE_PATH
    # )
    llm = HuggingfaceLargeLanguageModel(
        model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        cache_path=constants.LLM_CACHE_FILE_PATH)

    # Engines
    semantic_descriptor_engine = SemanticDescriptorEngine(
        bert_embedder, roberta_embedder, openai_embedder, sbert_embedder, llm
    )
    dim_reduction_engine = DimensionalityReductionEngine()
    clustering_engine = ClusteringEngine()

    # Load and pre-process semantic maps
    semantic_maps: List[SemanticMap] = []
    for json_file in constants.SEMANTIC_MAPS_PATHS:
        # Create a SemanticMap directly from its JSON file
        sm = SemanticMap.from_json_path(str(json_file))
        semantic_maps.append(sm)

    # For each semantic map
    for semantic_map in semantic_maps[:args.number_maps]:

        # Files to save clusterings
        clustering_json_file_path = os.path.join(get_results_path_for_method(args),
                                                 semantic_map.semantic_map_id,
                                                 "clustering.json")
        clustering_plot_file_path = os.path.join(get_results_path_for_method(args),
                                                 semantic_map.semantic_map_id,
                                                 "plot.png")

        print("#"*40)
        print(f"Processing {semantic_map.semantic_map_id}...")
        print("#"*40)

        # Set geometric descriptors = bounding box center
        for semantic_map_object in tqdm(semantic_map.get_all_objects(),
                                        desc=f"Setting geometric descriptors {semantic_map.semantic_map_id}..."):
            semantic_map_object.geometric_descriptor = semantic_map_object.bbox_center

        if args.method == constants.METHOD_LLM:

            # Build prompt and conversation history
            place_segmenter_prompt = PlaceSegmenterPrompt(
                semantic_map=semantic_map.get_prompt_json_representation())
            conversation_history = ConversationHistory.create_from_user_message(
                place_segmenter_prompt.get_prompt_text())

            if args.llm_request:
                # Perform LLM request
                # response = llm.generate_json(conversation_history, retries=10)
                response = file_utils.load_json(
                    os.path.join(get_results_path_for_method(args),
                                 semantic_map.semantic_map_id,
                                 "clustering.json"))

                # Assemble clustering from response
                mixed_clustering = Clustering([])
                for i, place_entry in enumerate(response["places"]):
                    tag = place_entry.get("name", "")
                    description = place_entry.get("description", "")
                    objects_list = place_entry["objects"]

                    cluster = Cluster(
                        cluster_id=i,
                        objects=[],
                        tag=tag,
                        description=description,
                    )

                    # Fill semantic map objects
                    for object_id in objects_list:
                        semantic_map_object = semantic_map.find_object(
                            object_id)
                        if semantic_map_object is not None:
                            cluster.append_object(semantic_map_object)

                    mixed_clustering.append_cluster(cluster)

            # Save prompt
            prompt_text_path = os.path.join(
                get_results_path_for_method(args),
                semantic_map.semantic_map_id,
                "prompt.txt"
            )
            file_utils.create_directories_for_file(prompt_text_path)
            file_utils.save_text_to_file(
                place_segmenter_prompt.get_prompt_text(), prompt_text_path)

        else:
            # Assign semantic descriptor -> use semantic descriptor engine
            for semantic_map_object in tqdm(semantic_map.get_all_objects(),
                                            desc=f"Generating features for {semantic_map.semantic_map_id}..."):
                semantic_map_object.semantic_descriptor = semantic_descriptor_engine.get_semantic_descriptor_from_method(
                    args.method, semantic_map_object.get_most_probable_class()
                )

            # Convert features into numpy arrays
            # Shape: (num_objects, geometric_dim)
            geometric_descriptor_matrix = np.array(
                list(map(lambda obj: obj.geometric_descriptor, semantic_map.get_all_objects())))
            print(
                f"[main] Geometric descriptor matrix shape: {geometric_descriptor_matrix.shape}")

            # Shape: (num_objects, semantic_dim)
            semantic_descriptor_matrix = np.array(
                list(map(lambda obj: obj.semantic_descriptor, semantic_map.get_all_objects())))
            print(
                f"[main] Semantic descriptor matrix shape: {semantic_descriptor_matrix.shape}")

            # Perform dimensionality reduction on semantic descriptor
            if args.method != constants.METHOD_GEOMETRIC and args.semantic_dimension is not None:
                reduced_semantic_descriptor_matrix = dim_reduction_engine.reduce(
                    semantic_descriptor_matrix,
                    args.semantic_dimension,
                    args.dimensionality_reductor
                )
            else:
                reduced_semantic_descriptor_matrix = semantic_descriptor_matrix

            # Normalize both descriptors separately
            normalized_geometric_descriptor_matrix = StandardScaler().fit_transform(
                geometric_descriptor_matrix)
            print(
                f"[main] Normalized geometric descriptor matrix shape: {normalized_geometric_descriptor_matrix.shape}")
            if args.method != constants.METHOD_GEOMETRIC:
                normalized_semantic_descriptor_matrix = StandardScaler().fit_transform(
                    reduced_semantic_descriptor_matrix)
                print(
                    f"[main] Normalized semantic descriptor matrix shape: {normalized_semantic_descriptor_matrix.shape}")
            else:
                normalized_semantic_descriptor_matrix = reduced_semantic_descriptor_matrix

            # Create mixed descriptor
            if args.method != constants.METHOD_GEOMETRIC:
                mixed_descriptor_matrix = np.hstack(
                    (normalized_geometric_descriptor_matrix, normalized_semantic_descriptor_matrix))
            else:
                mixed_descriptor_matrix = normalized_geometric_descriptor_matrix
            print(
                f"[main] Normalized mixed descriptor matrix shape: {mixed_descriptor_matrix.shape}")

            # Update object descriptors with normalized and reduced descriptors
            for i, object in enumerate(semantic_map.get_all_objects()):
                object.geometric_descriptor = list(
                    normalized_geometric_descriptor_matrix[i])
                object.semantic_descriptor = list(
                    normalized_semantic_descriptor_matrix[i])
                object.global_descriptor = list(mixed_descriptor_matrix[i])

            # Perform clustering
            mixed_clustering = clustering_engine.clusterize(
                semantic_map, args.clustering_algorithm, eps=args.eps, min_samples=args.min_samples, semantic_weight=args.semantic_weight, noise_objects_new_clusters=True)

            # Post process clustering
            if args.method in (constants.METHOD_BERT_POST, constants.METHOD_LLM_SBERT_POST):
                mixed_clustering = clustering_engine.post_process_clustering(
                    semantic_map, mixed_clustering, args.merge_geometric_threshold, args.merge_semantic_threshold, args.split_semantic_threshold, clustering_json_file_path, clustering_plot_file_path)

        if args.method != constants.METHOD_LLM or args.llm_request:
            # Save clustering
            file_utils.create_directories_for_file(clustering_json_file_path)
            file_utils.create_directories_for_file(clustering_plot_file_path)
            mixed_clustering.save_to_json(clustering_json_file_path)
            mixed_clustering.visualize_2D(
                title=get_chart_title_for_method(
                    args, semantic_map.semantic_map_id),
                semantic_map=semantic_map,
                geometric_threshold=args.merge_geometric_threshold,
                file_path=clustering_plot_file_path)

    print("[main] The main script finished successfully!")


def main_categorization(args):
    # Load environment variables
    load_dotenv()

    # Instantiate models
    llm = GeminiProvider(
        credentials_file=constants.GOOGLE_GEMINI_CREDENTIALS_FILENAME,
        project_id=constants.GOOGLE_GEMINI_PROJECT_ID,
        project_location=constants.GOOGLE_GEMINI_PROJECT_LOCATION,
        model_name=GeminiProvider.GEMINI_1_5_PRO,
        cache_path=constants.LLM_CACHE_FILE_PATH
    )

    # Load semantic maps
    semantic_maps: List[SemanticMap] = []
    for semantic_map_path, colors_path in zip(constants.SEMANTIC_MAPS_PATHS, constants.SEMANTIC_MAPS_COLORS_PATHS):
        sm = SemanticMap.from_json_path(
            semantic_map_path, colors_path=colors_path)
        semantic_maps.append(sm)

    # Iterate over segmented methods
    for method in file_utils.list_subdirectories(os.path.join(constants.PLACES_RESULTS_FOLDER_PATH)):

        # Skip LLM only method, as it already categories places
        if method == constants.METHOD_LLM:
            print(f"Skipping method {method} for categorization.")
            continue

        for semantic_map in semantic_maps[:args.number_maps]:

            # Load clustering
            clustering_file_path = os.path.join(constants.PLACES_RESULTS_FOLDER_PATH,
                                                method,
                                                semantic_map.semantic_map_id,
                                                "clustering.json")
            clustering = Clustering.load_from_json(clustering_file_path)

            # Categorize each cluster
            for cluster in clustering.clusters:

                # Build prompt and conversation history
                place_categorizer_prompt = PlaceCategorizerPrompt(
                    [semantic_map.find_object(cluster_object.object_id) for cluster_object in cluster.objects])
                conv_his = ConversationHistory.create_from_user_message(
                    place_categorizer_prompt.get_prompt_text())

                # Perform LLM request and set cluster tag and description
                if args.llm_request:
                    response = llm.generate_json(conv_his,
                                                 retries=3)
                    clustering.find_cluster_by_id(
                        cluster.cluster_id).tag = response["tag"]
                    clustering.find_cluster_by_id(
                        cluster.cluster_id).description = response["description"]

                # Save prompt
                prompt_text_path = os.path.join(constants.PLACES_RESULTS_FOLDER_PATH,
                                                method,
                                                semantic_map.semantic_map_id,
                                                f"place_{cluster.cluster_id}_categorization_prompt.txt")
                file_utils.create_directories_for_file(prompt_text_path)
                file_utils.save_text_to_file(
                    place_categorizer_prompt.get_prompt_text(), prompt_text_path)

            # Save categorized clustering
            if args.llm_request:
                print(
                    f"Saving clustering for {semantic_map.semantic_map_id}...")
                clustering.save_to_json(clustering_file_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Perform place segmentation and categorization on a set of semantic maps"
    )

    parser.add_argument(
        "-p", "--persist-log",
        action="store_true",
        help="Redirect output to a log file instead of printing to the terminal."
    )

    parser.add_argument(
        "-n", "--number-maps",
        type=int,
        default=10,
        help="Number of semantic map to which place categorization will be applied."
    )

    parser.add_argument(
        "--stage",
        choices=[constants.STAGE_SEGMENTATION, constants.STAGE_CATEGORIZATION],
        default=constants.STAGE_SEGMENTATION,
        help="Stage of the pipeline to run."
    )

    # SEGMENTATION STAGE parameters
    # SEMANTIC DESCRIPTOR parameters
    parser.add_argument(
        "--method",
        choices=[
            constants.METHOD_GEOMETRIC,
            constants.METHOD_BERT,
            constants.METHOD_OPENAI,
            constants.METHOD_ROBERTA,
            constants.METHOD_LLM_SBERT,
            constants.METHOD_LLM_OPENAI,
            constants.METHOD_BERT_POST,
            constants.METHOD_LLM_SBERT_POST,
            constants.METHOD_LLM
        ],
        default=constants.METHOD_BERT,
        help="How to compute the semantic descriptor."
    )

    parser.add_argument(
        "-w", "--semantic-weight",
        type=float,
        default=0.005,
        help="Semantic weight in DBSCAN distance."
    )

    parser.add_argument(
        "-d", "--semantic-dimension",
        type=int,
        default=None,
        help="Dimensions to which reduce the semantic descriptor using PCA."
    )

    parser.add_argument(
        "-r", "--dimensionality_reductor",
        choices=[constants.DIM_REDUCTOR_PCA, constants.DIM_REDUCTOR_UMAP],
        default=constants.DIM_REDUCTOR_PCA,
        help="Dimensionality reduction method to apply to the semantic descriptor."
    )

    # DBSCAN parameters
    parser.add_argument(
        "-e", "--eps",
        type=float,
        default=1.0,
        help="eps parameter in the DBSCAN algorithm."
    )

    parser.add_argument(
        "-m", "--min-samples",
        type=int,
        default=2,
        help="min_samples parameter in the DBSCAN algorithm."
    )

    # POST-PROCESSING parameters
    parser.add_argument(
        "--merge-geometric-threshold",
        type=float,
        default=1.5,
        help="Maximum distance between two clusters that could be merged."
    )

    parser.add_argument(
        "--merge-semantic-threshold",
        type=float,
        default=0.99,
        help="Minimum semantic distance between two clusters that should be merged."
    )

    parser.add_argument(
        "--split-semantic-threshold",
        type=float,
        default=0.5,
        help="Minimum semantic variance to split clusters."
    )

    parser.add_argument(
        "-c", "--clustering-algorithm",
        choices=[constants.CLUSTERING_ALGORITHM_DBSCAN,
                 constants.CLUSTERING_ALGORITHM_HDBSCAN],
        default=constants.CLUSTERING_ALGORITHM_DBSCAN,
        help="Clustering algorithm to use."
    )

    # CATEGORIZATION STAGE parameters
    parser.add_argument(
        "--llm-request",
        action="store_true",
        help="Whether to perform the LLM request to characterize a place, or just generate the prompt."
    )

    args = parser.parse_args()

    # Redirect output to a log file (args.persist_log)
    if args.persist_log:
        log_file_path = os.path.join(
            get_results_path_for_method(args), "log.txt"
        )
        file_utils.create_directories_for_file(log_file_path)
        sys.stdout = open(log_file_path, "w")
        sys.stderr = sys.stdout

    if args.stage == constants.STAGE_SEGMENTATION:
        main_segmentation(args)
    elif args.stage == constants.STAGE_CATEGORIZATION:
        main_categorization(args)
