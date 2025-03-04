
from functools import partial
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import argparse
import os
from typing import List


from embedding.bert_embedder import BERTEmbedder
from embedding.openai_embedder import OpenAIEmbedder
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
    - Euclidean distance for the first three dimensions
    - + 0.001 * Euclidean distance for all remaining dimensions
    """
    # print(A.shape)
    # print(B.shape)
    A, B = np.array(A), np.array(B)

    # Euclidean distance for the first three dimensions
    dist_first_3 = np.sqrt(np.sum((A[:3] - B[:3]) ** 2))
    print("first3", dist_first_3)

    # Euclidean distance for remaining dimensions (if any)
    dist_rest = (np.sqrt(np.sum((A[3:] - B[3:]) ** 2))
                 if len(A) > 3 else 0.0) * semantic_weight
    print("rest", dist_rest)

    return dist_first_3 + dist_rest


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


# Models
bert_embedder = None
openai_embedder = None
sbert_embedder = None
deepseek_llm = None


def main(args):
    # Instantiate models
    global bert_embedder, openai_embedder, sbert_embedder, deepseek_llm
    bert_embedder = BERTEmbedder()
    openai_embedder = OpenAIEmbedder()
    sbert_embedder = SentenceEmbedder()
    if args.semantic_descriptor in (constants.SEMANTIC_DESCRIPTOR_DEEPSEEK_SBERT, constants.SEMANTIC_DESCRIPTOR_DEEPSEEK_OPENAI):
        _, deepseek_llm = LargeLanguageModel.create_from_huggingface(
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")

    # Load and pre-process semantic map
    semantic_maps: List[SemanticMap] = list()
    for semantic_map_file in sorted(os.listdir(constants.SEMANTIC_MAPS_FOLDER_PATH)):

        semantic_map_basename = file_utils.get_file_basename(semantic_map_file)
        # Load semantic map
        semantic_map_obj = file_utils.load_json(os.path.join(constants.SEMANTIC_MAPS_FOLDER_PATH,
                                                             semantic_map_file))
        # Create SemanticMap object
        semantic_maps.append(SemanticMap(semantic_map_basename,
                                         [SemanticMapObject(obj_id, obj_data) for obj_id, obj_data in semantic_map_obj["instances"].items()]))

    # For each semantic map
    for semantic_map in semantic_maps[:args.number_maps]:

        print("#"*40)
        print(f"Processing {semantic_map.get_semantic_map_id()}...")
        print("#"*40)

        # Compute object semantic and geometric features
        semantic_features = dict()
        geometric_features = dict()

        # For each object
        for semantic_map_object in tqdm(semantic_map.get_objects(),
                                        desc=f"Generating features for {semantic_map.get_semantic_map_id()}..."):

            # Geometric feature = bounding box
            geometric_features[semantic_map_object.get_object_id()] = \
                get_geometric_descriptor(semantic_map_object)

            # Semantic feature = to be decided
            semantic_features[semantic_map_object.get_object_id()] = \
                get_semantic_descriptor(
                    args.semantic_descriptor, semantic_map_object)

        # Convert features into numpy arrays
        # Shape: (num_objects, geometric_dim)
        geometric_descriptor_matrix = np.array(
            list(geometric_features.values()))

        # Shape: (num_objects, semantic_dim)
        semantic_descriptor_matrix = np.array(list(semantic_features.values()))

        # Perform dimensionality reduction for semantic_features
        # target_dim = 3  # Define target dimensionality
        # if semantic_descriptor_matrix.shape[1] > target_dim:
        #     pca = PCA(n_components=target_dim)
        #     reduced_semantic_descriptor = pca.fit_transform(
        #         semantic_descriptor_matrix)
        # else:
        #     raise ValueError(
        #         f"Semantic descriptor size is lower than target dim {target_dim}")
        reduced_semantic_descriptor = semantic_descriptor_matrix

        # Normalize both descriptors separately
        normalized_geometric = StandardScaler().fit_transform(
            geometric_descriptor_matrix)
        print("normalized_geometric_shape", normalized_geometric.shape)
        normalized_semantic = StandardScaler().fit_transform(
            reduced_semantic_descriptor)
        print("normalized_semantic_shape", normalized_semantic.shape)

        # Create mixed descriptor
        mixed_descriptor_matrix = np.hstack(
            (normalized_geometric, normalized_semantic))
        print("mixed_descriptor_matrix", mixed_descriptor_matrix.shape)

        # Convert back to dictionary format
        mixed_descriptors = {obj_id: mixed_descriptor_matrix[i] for i, obj_id in enumerate(
            geometric_features.keys())}

        # Perform clustering
        eps = 1
        for min_samples in range(1, 3):
            mixed_clustering_result: ClusteringResult = apply_clustering(
                mixed_descriptors, eps=1, min_samples=min_samples, semantic_weight=args.semantic_weight)
            # Visualize using visualize_clusters
            print("Saving clustering result...")

            json_file_path = os.path.join(constants.RESULTS_FOLDER_PATH,
                                          args.semantic_descriptor,
                                          f"{semantic_map.get_semantic_map_id()}_mixed_e{eps}_m{min_samples}_w{args.semantic_weight}",
                                          "clustering.json")
            plot_file_path = os.path.join(constants.RESULTS_FOLDER_PATH,
                                          args.semantic_descriptor,
                                          f"{semantic_map.get_semantic_map_id()}_mixed_e{eps}_m{min_samples}_w{args.semantic_weight}",
                                          "plot.png")
            file_utils.create_directories_for_file(json_file_path)
            file_utils.create_directories_for_file(plot_file_path)
            mixed_clustering_result.save_to_json(json_file_path)
            mixed_clustering_result.visualize(semantic_map.get_objects(),
                                              f"Mixed Clustering eps={eps}, min_samples={min_samples}, semantic_weight={args.semantic_weight}", plot_file_path)


def get_geometric_descriptor(semantic_map_object: SemanticMapObject):
    return semantic_map_object.get_bbox_center()


def get_semantic_descriptor(semantic_descriptor: str, semantic_map_object: SemanticMapObject, verbose: bool = False):

    if semantic_descriptor == constants.SEMANTIC_DESCRIPTOR_BERT:
        # Generate embedding
        return bert_embedder.embed_text(semantic_map_object.get_most_probable_class())
    elif semantic_descriptor == constants.SEMANTIC_DESCRIPTOR_OPENAI:
        # Generate embedding
        return openai_embedder.embed_text(semantic_map_object.get_most_probable_class())
    elif semantic_descriptor == constants.SEMANTIC_DESCRIPTOR_DEEPSEEK_SBERT:
        # Generate sentence
        sentence_generator_prompt = SentenceGeneratorPrompt(
            word=semantic_map_object.get_most_probable_class())
        sentence = deepseek_llm.generate_json_retrying(
            sentence_generator_prompt.get_prompt_text(), params={"max_length": 1000}, retries=10)["description"]
        if verbose:
            print(
                f"Sentence for {semantic_map_object.get_most_probable_class()} -> {sentence}")
        # Generate embedding
        return sbert_embedder.embed_text(sentence)
    elif semantic_descriptor == constants.SEMANTIC_DESCRIPTOR_DEEPSEEK_OPENAI:
        # Generate sentence
        sentence_generator_prompt = SentenceGeneratorPrompt(
            word=semantic_map_object.get_most_probable_class())
        sentence = deepseek_llm.generate_json_retrying(
            sentence_generator_prompt.get_prompt_text(), params={"max_length": 1000}, retries=10)["description"]
        if verbose:
            print(
                f"Sentence for {semantic_map_object.get_most_probable_class()} -> {sentence}")
        # Generate embedding
        return openai_embedder.embed_text(sentence)
    else:
        raise NotImplementedError(
            f"Not implemented semantic descriptor {semantic_descriptor}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Performs place categorization on a set of semantic map")

    parser.add_argument("-n", "--number-maps",
                        help="Number of semantic map to which place categorization will be applied.",
                        type=int,
                        default=10)

    parser.add_argument("-s", "--semantic-descriptor",
                        help="How to compute the semantic descriptor.",
                        choices=[constants.SEMANTIC_DESCRIPTOR_BERT, constants.SEMANTIC_DESCRIPTOR_OPENAI,
                                 constants.SEMANTIC_DESCRIPTOR_DEEPSEEK_SBERT, constants.SEMANTIC_DESCRIPTOR_DEEPSEEK_OPENAI],
                        default=constants.SEMANTIC_DESCRIPTOR_BERT)

    parser.add_argument("-w", "--semantic-weight",
                        help="Semantic weight in DBSCAN distance.",
                        type=float,
                        default=0.005)

    args = parser.parse_args()

    main(args)
