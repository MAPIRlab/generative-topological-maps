
import argparse
import os
from typing import List

import nltk

from embedding.bert_embedder import BERTEmbedder
from embedding.openai_embedder import OpenAIEmbedder
from utils import file_utils
import constants

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

from voxeland.semantic_map import SemanticMap
from voxeland.semantic_map_object import SemanticMapObject

from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()


def visualize_clusters(semantic_map_objects: List[SemanticMapObject], clusters, title, filename):
    """Visualizes the clustered objects from a top-down view and saves the plot."""
    plt.figure(figsize=(8, 8))
    # Get distinct colors for clusters
    colors = plt.cm.get_cmap("tab10", len(clusters))
    legend_labels = []
    legend_colors = []

    for cluster_id, object_ids in clusters.items():
        color = colors(cluster_id)
        legend_labels.append(f"Cluster {cluster_id}")
        legend_colors.append(color)

        for obj_id in object_ids:
            obj = next(
                (o for o in semantic_map_objects if o.get_object_id() == obj_id), None)
            if obj:
                print(obj.get_most_probable_class())
                bbox_center = obj.get_bbox_center()
                print(bbox_center)
                bbox_size = obj.get_bbox_size()
                print(bbox_size)
                x, y = bbox_center[:2]  # Top-down view (x, y plane)
                w, h = bbox_size[:2]

                # Draw bounding box
                rect = plt.Rectangle(
                    (x - w/2, y - h/2), w, h, edgecolor=color, facecolor=color, alpha=0.5)
                plt.gca().add_patch(rect)

                # Add label
                plt.text(x, y, obj.get_most_probable_class(),
                         fontsize=8, ha='center', va='center', color='black')

    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')

    for cluster_id, object_ids in clusters.items():
        color = colors(cluster_id)
        for obj_id in object_ids:
            obj = next(
                (o for o in semantic_map_objects if o.get_object_id() == obj_id), None)
            if obj:
                bbox_center = obj.get_bbox_center()
                bbox_size = obj.get_bbox_size()
                x, y = bbox_center[:2]  # Top-down view (x, y plane)
                w, h = bbox_size[:2]

                x_min, x_max = min(x_min, x - w/2), max(x_max, x + w/2)
                y_min, y_max = min(y_min, y - h/2), max(y_max, y + h/2)

    plt.xlim(x_min - 1, x_max + 1)
    plt.ylim(y_min - 1, y_max + 1)

    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.title(title)
    plt.grid(True)

    # Add legend
    legend_patches = [plt.Rectangle((0, 0), 1, 1, fc=color)
                      for color in legend_colors]
    plt.legend(legend_patches, legend_labels, loc="upper right")

    os.makedirs(constants.PLOTS_FOLDER_PATH, exist_ok=True)
    filepath = os.path.join(constants.PLOTS_FOLDER_PATH, filename)
    plt.savefig(filepath)
    print("saving...", filepath)
    plt.close()


def find_optimal_clusters(data, max_k=10):
    """Uses the Elbow Method and Silhouette Score to determine the optimal k."""
    inertia = []
    silhouette_scores = []
    k_values = range(2, max_k + 1)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, kmeans.labels_))

    # Plot Elbow Method
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(k_values, inertia, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')

    # Plot Silhouette Score
    plt.subplot(1, 2, 2)
    plt.plot(k_values, silhouette_scores, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score')

    plt.show()

    # Choosing the best k based on Silhouette Score
    best_k = k_values[np.argmax(silhouette_scores)]
    return best_k


def apply_clustering(features, feature_type, k=None, max_k=10):
    """Performs clustering and returns labels."""

    data = np.array(list(features.values()))
    if k is None:
        best_k = find_optimal_clusters(data, max_k)
    else:
        best_k = k

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data)

    # print(f"Optimal k for {feature_type}: {best_k}")
    clusters = {i: [] for i in range(best_k)}

    for obj_id, label in zip(features.keys(), labels):
        clusters[label].append(obj_id)

    # print(f"\n{feature_type} Clusters:")
    # for cluster_id, objects in clusters.items():
        # print(f"Cluster {cluster_id}: {objects}")

    return best_k, clusters


def main(args):
    # Prepare models
    nltk.download("wordnet")
    bert_embedder = BERTEmbedder()
    openai_embedder = OpenAIEmbedder()

    # Load and pre-process semantic map
    semantic_maps = list()
    for semantic_map_file in sorted(os.listdir(constants.SEMANTIC_MAPS_FOLDER_PATH)):

        semantic_map_basename = file_utils.get_file_basename(semantic_map_file)

        # Load semantic map
        semantic_map_obj = file_utils.load_json(os.path.join(constants.SEMANTIC_MAPS_FOLDER_PATH,
                                                             semantic_map_file))

        semantic_maps.append((semantic_map_basename, semantic_map_obj))

    # For each semantic map
    for (semantic_map_id, semantic_map_json_data) in semantic_maps[:args.number_maps]:

        print(semantic_map_id)

        # Create semantic map object
        semantic_map_objects: List[SemanticMapObject] = [SemanticMapObject(
            obj_id, obj_data) for obj_id, obj_data in semantic_map_json_data["instances"].items()]
        semantic_map: SemanticMap = SemanticMap(semantic_map_objects)

        # Compute object semantic and geometric features
        semantic_features = dict()
        geometric_features = dict()

        # For each object
        for semantic_map_object in tqdm(semantic_map.get_objects(),
                                        desc="Generating geometric and semantic features..."):

            # Geometric feature = bounding box
            geometric_features[semantic_map_object.get_object_id()] = \
                get_geometric_feature(semantic_map_object)

            # Semantic feature = to be decided
            semantic_features[semantic_map_object.get_object_id()] = \
                get_semantic_feature(semantic_map_object)

            # print(object_id, len(geometric_features[object_id]), len(semantic_features[object_id]))

        geometric_best_k, geometric_clusters = apply_clustering(
            geometric_features, 'Geometric Features')
        print("Geometric clusters")
        print(f"k = {geometric_best_k}")
        for cluster_id, cluster_list in geometric_clusters.items():
            print(f"Cluster {cluster_id}")
            for object_id in cluster_list:
                semantic_map_object = semantic_map.find_object(object_id)
                print(
                    f"\t{object_id} -> {semantic_map_object.get_most_probable_class()}")

        semantic_best_k, semantic_clusters = apply_clustering(
            semantic_features, 'Semantic Features', k=geometric_best_k)
        print("Semantic clusters")
        print(f"k = {geometric_best_k}")
        for cluster_id, cluster_list in semantic_clusters.items():
            print(f"Cluster {cluster_id}")
            for object_id in cluster_list:
                semantic_map_object = semantic_map.find_object(object_id)
                print(
                    f"\t{object_id} -> {semantic_map_object.get_most_probable_class()}")

        print(f"Saving Geometric Clusters plot {semantic_map_id}")
        visualize_clusters(semantic_map_objects, geometric_clusters,
                           "Geometric Clustering", f"{semantic_map_id}_geometric.png")

        print("Saving Semantic Clusters plot")
        visualize_clusters(semantic_map_objects, semantic_clusters,
                           "Semantic Clustering", f"{semantic_map_id}_semantic.png")


def get_geometric_feature(semantic_map_object: SemanticMapObject):
    return semantic_map_object.get_bbox_center()


def get_semantic_feature(semantic_map_object: SemanticMapObject):

    # Get sentence from object
    

    return openai_embedder.embed_text(
        semantic_map_object.get_most_probable_class())


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Performs place categorization on a set of semantic map")

    parser.add_argument("-n", "--number-maps",
                        help="Number of semantic map to which place categorization will be applied.",
                        type=int,
                        default=10)

    args = parser.parse_args()

    main(args)
