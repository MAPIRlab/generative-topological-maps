
from functools import partial
import sys
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import argparse
import os
from typing import List


from embedding.bert_embedder import BERTEmbedder
from embedding.openai_embedder import OpenAIEmbedder
from embedding.roberta_embedder import RoBERTaEmbedder
from embedding.sentence_embedder import SentenceEmbedder
from llm.large_language_model import LargeLanguageModel
from prompt.sentence_generator_prompt import SentenceGeneratorPrompt
from utils import file_utils
import constants
import numpy as np
from scipy.spatial.distance import pdist, squareform

from voxeland.clustering import ClusteringResult, ObjectCluster
from voxeland.semantic_map import SemanticMap
from voxeland.semantic_map_object import SemanticMapObject

from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()


def geometric_semantic_distance(A, B, semantic_weight: float = 0.005):
    """
    Custom distance:
    - Euclidean distance for the first three dimensions (geometric)
    - Plus a weighted, dimension-normalized Euclidean distance for the rest (semantic).
    """
    print("[geometric_semantic_distance] Computing distance between A and B")
    A, B = np.array(A), np.array(B)
    print(f"[geometric_semantic_distance] A.shape: {A.shape}")
    print(f"[geometric_semantic_distance] B.shape: {B.shape}")

    # Geometric distance
    geometric_dist = np.sqrt(np.sum((A[:3] - B[:3]) ** 2))
    print(
        f"[geometric_semantic_distance] Geometric distance: {geometric_dist}")

    # Semantic distance
    semantic_dim = max(0, len(A) - 3)
    if semantic_dim > 0:
        print(
            f"[geometric_semantic_distance] Normalizing semantic distance for {semantic_dim} dimensions")
        raw_semantic_dist = np.sqrt(np.sum((A[3:] - B[3:]) ** 2))
        norm_semantic_dist = raw_semantic_dist / np.sqrt(semantic_dim)
        semantic_dist = norm_semantic_dist * semantic_weight
    else:
        semantic_dist = 0.0
    print(f"[geometric_semantic_distance] Semantic distance: {semantic_dist}")

    # Global distance
    global_dist = geometric_dist + semantic_dist
    print(f"[geometric_semantic_distance] Global distance: {global_dist}")

    return global_dist


def apply_clustering(object_label_descriptor_dict, eps, min_samples, semantic_weight) -> ClusteringResult:
    """Performs DBSCAN clustering using a custom distance metric and returns a Clustering instance."""

    object_labels = list(object_label_descriptor_dict.keys())
    object_descriptors = np.array(list(object_label_descriptor_dict.values()))

    # Compute the custom distance matrix
    distance_matrix = squareform(
        pdist(object_descriptors, metric=partial(geometric_semantic_distance, semantic_weight=semantic_weight)))

    # Apply DBSCAN with precomputed distance matrix
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    labels = clustering.fit_predict(distance_matrix)

    # Group objects into clusters
    clusters_dict = {}
    for object_label, object_cluster in zip(object_labels, labels):
        clusters_dict.setdefault(object_cluster, []).append(object_label)

    # Convert clusters into ObjectCluster instances
    object_clusters = [ObjectCluster(cluster_id, obj_labels)
                       for cluster_id, obj_labels in clusters_dict.items()]

    return ClusteringResult(object_clusters)


def get_results_path_for_method(args):
    return os.path.join(constants.RESULTS_FOLDER_PATH,
                        "method_results",
                        f"{args.semantic_descriptor}_e{args.eps}_m{args.min_samples}_w{args.semantic_weight}_d{args.semantic_dimension}")


# Models
bert_embedder = None
roberta_embedder = None
openai_embedder = None
sbert_embedder = None
deepseek_llm = None


def main(args):
    # Instantiate models
    global bert_embedder, roberta_embedder, openai_embedder, sbert_embedder
    bert_embedder = BERTEmbedder()
    roberta_embedder = RoBERTaEmbedder()
    openai_embedder = OpenAIEmbedder()
    sbert_embedder = SentenceEmbedder()

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

        # Compute object semantic and geometric features
        semantic_descriptors = dict()
        geometric_descriptors = dict()

        # For each object
        for semantic_map_object in tqdm(semantic_map.get_all_objects(),
                                        desc=f"Generating features for {semantic_map.get_semantic_map_id()}..."):

            # Geometric feature = bounding box
            geometric_descriptors[semantic_map_object.get_object_id()] = \
                get_geometric_descriptor(semantic_map_object)

            # Semantic feature = to be decided
            semantic_descriptors[semantic_map_object.get_object_id()] = \
                get_semantic_descriptor(
                    args.semantic_descriptor, semantic_map, semantic_map_object, verbose=True)

        # Convert features into numpy arrays
        # Shape: (num_objects, geometric_dim)
        geometric_descriptor_matrix = np.array(
            list(geometric_descriptors.values()))

        # Shape: (num_objects, semantic_dim)
        semantic_descriptor_matrix = np.array(
            list(semantic_descriptors.values()))
        print("semantic_descriptor_matrix", semantic_descriptor_matrix.shape)

        # Perform dimensionality reduction for semantic_features
        if args.semantic_descriptor != constants.SEMANTIC_DESCRIPTOR_NONE and args.semantic_dimension is not None:
            if semantic_descriptor_matrix.shape[1] > args.semantic_dimension:
                pca = PCA(n_components=args.semantic_dimension)
                reduced_semantic_descriptor_matrix = pca.fit_transform(
                    semantic_descriptor_matrix)
            else:
                raise ValueError(
                    f"Semantic descriptor size is lower than target dim {args.semantic_dimension}")
        else:
            reduced_semantic_descriptor_matrix = semantic_descriptor_matrix

        # Normalize both descriptors separately
        normalized_geometric_descriptor_matrix = StandardScaler().fit_transform(
            geometric_descriptor_matrix)
        print("normalized_geometric_shape",
              normalized_geometric_descriptor_matrix.shape)
        if args.semantic_descriptor != constants.SEMANTIC_DESCRIPTOR_NONE:
            normalized_semantic_descriptor_matrix = StandardScaler().fit_transform(
                reduced_semantic_descriptor_matrix)
            print("normalized_semantic_shape",
                  normalized_semantic_descriptor_matrix.shape)
        else:
            normalized_semantic_descriptor_matrix = reduced_semantic_descriptor_matrix

        # Create mixed descriptor
        if args.semantic_descriptor != constants.SEMANTIC_DESCRIPTOR_NONE:
            mixed_descriptor_matrix = np.hstack(
                (normalized_geometric_descriptor_matrix, normalized_semantic_descriptor_matrix))
        else:
            mixed_descriptor_matrix = normalized_geometric_descriptor_matrix
        print("mixed_descriptor_matrix", mixed_descriptor_matrix.shape)

        # Convert back to dictionary format
        mixed_descriptors = {obj_id: mixed_descriptor_matrix[i] for i, obj_id in enumerate(
            geometric_descriptors.keys())}

        # Perform clustering
        mixed_clustering_result: ClusteringResult = apply_clustering(
            mixed_descriptors, eps=args.eps, min_samples=args.min_samples, semantic_weight=args.semantic_weight)
        # Visualize using visualize_clusters
        print("Saving clustering result...")

        json_file_path = os.path.join(get_results_path_for_method(args),
                                      semantic_map.get_semantic_map_id(),
                                      "clustering.json")
        plot_file_path = os.path.join(get_results_path_for_method(args),
                                      semantic_map.get_semantic_map_id(),
                                      "plot.png")
        file_utils.create_directories_for_file(json_file_path)
        file_utils.create_directories_for_file(plot_file_path)
        mixed_clustering_result.save_to_json(json_file_path)
        mixed_clustering_result.visualize(semantic_map.get_all_objects(),
                                          f"s_d={args.semantic_descriptor} eps={args.eps}, m_s={args.min_samples}, s_w={args.semantic_weight}, s_d={args.semantic_dimension}",
                                          plot_file_path)


def get_geometric_descriptor(semantic_map_object: SemanticMapObject):
    return semantic_map_object.get_bbox_center()


def get_semantic_descriptor(semantic_descriptor: str, semantic_map: SemanticMap, semantic_map_object: SemanticMapObject, verbose: bool = False):

    if semantic_descriptor == constants.SEMANTIC_DESCRIPTOR_NONE:
        return []
    elif semantic_descriptor == constants.SEMANTIC_DESCRIPTOR_BERT:
        return bert_embedder.embed_text(semantic_map_object.get_most_probable_class())
    elif semantic_descriptor == constants.SEMANTIC_DESCRIPTOR_ROBERTA:
        return roberta_embedder.embed_text(semantic_map_object.get_most_probable_class())
    elif semantic_descriptor == constants.SEMANTIC_DESCRIPTOR_OPENAI:
        return openai_embedder.embed_text(semantic_map_object.get_most_probable_class())
    elif semantic_descriptor in (constants.SEMANTIC_DESCRIPTOR_DEEPSEEK_SBERT, constants.SEMANTIC_DESCRIPTOR_DEEPSEEK_OPENAI):

        # If LLM not instantiated, instantiate! SLOW PROCESS
        global deepseek_llm
        if deepseek_llm:
            _, deepseek_llm = LargeLanguageModel.create_from_huggingface(
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")

        # Generate sentence
        llm_response_file_path = os.path.join(constants.RESULTS_FOLDER_PATH,
                                              "llm_responses",
                                              semantic_map.get_semantic_map_id(),
                                              f"{semantic_map_object.get_object_id()}.json")
        if os.path.exists(llm_response_file_path):
            response = file_utils.load_json(llm_response_file_path)
        else:
            sentence_generator_prompt = SentenceGeneratorPrompt(
                word=semantic_map_object.get_most_probable_class())
            response = deepseek_llm.generate_json_retrying(
                sentence_generator_prompt.get_prompt_text(), params={"max_length": 1000}, retries=10)
            file_utils.create_directories_for_file(llm_response_file_path)
            file_utils.save_dict_to_json_file(response, llm_response_file_path)

        sentence = response["description"]
        if verbose:
            print(
                f"Sentence for {semantic_map_object.get_object_id()} ({semantic_map_object.get_most_probable_class()}) -> {sentence}")

        # Generate embedding
        if semantic_descriptor == constants.SEMANTIC_DESCRIPTOR_DEEPSEEK_SBERT:
            return sbert_embedder.embed_text(sentence)
        elif semantic_descriptor == constants.SEMANTIC_DESCRIPTOR_DEEPSEEK_OPENAI:
            return openai_embedder.embed_text(response)
    else:
        raise NotImplementedError(
            f"Not implemented semantic descriptor {semantic_descriptor}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Performs place categorization on a set of semantic map")

    parser.add_argument("-p", "--persist-log",
                        help="Redirect output to a log file instead of printing to the terminal.",
                        action="store_true")

    parser.add_argument("-n", "--number-maps",
                        help="Number of semantic map to which place categorization will be applied.",
                        type=int,
                        default=10)

    # SEMANTIC DESCRIPTOR parameters
    parser.add_argument("-s", "--semantic-descriptor",
                        help="How to compute the semantic descriptor.",
                        choices=[constants.SEMANTIC_DESCRIPTOR_NONE,
                                 constants.SEMANTIC_DESCRIPTOR_BERT, constants.SEMANTIC_DESCRIPTOR_OPENAI, constants.SEMANTIC_DESCRIPTOR_ROBERTA,
                                 constants.SEMANTIC_DESCRIPTOR_DEEPSEEK_SBERT, constants.SEMANTIC_DESCRIPTOR_DEEPSEEK_OPENAI],
                        default=constants.SEMANTIC_DESCRIPTOR_BERT)

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
                        type=float,
                        default=1.0)

    parser.add_argument("-m", "--min-samples",
                        help="min_samples parameter in the DBSCAN algorithm",
                        type=int,
                        default=1)

    args = parser.parse_args()

    # Redirect output to a log file (args.persist_log)
    if args.persist_log:
        log_file_path = os.path.join(
            get_results_path_for_method(args), "log.txt")
        file_utils.create_directories_for_file(log_file_path)

        sys.stdout = open(log_file_path, "w")
        sys.stderr = sys.stdout

    main(args)
