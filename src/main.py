
import argparse
import os

from bert import BERTEmbedder
from utils import file_utils
from voxeland import preprocess
import constants

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

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

def apply_clustering(features, feature_type, max_k=10):
    """Performs clustering and returns labels."""
    data = np.array(list(features.values()))
    best_k = find_optimal_clusters(data, max_k)
    
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data)
    
    print(f"Optimal k for {feature_type}: {best_k}")
    clusters = {i: [] for i in range(best_k)}
    
    for obj_id, label in zip(features.keys(), labels):
        clusters[label].append(obj_id)
    
    print(f"\n{feature_type} Clusters:")
    for cluster_id, objects in clusters.items():
        print(f"Cluster {cluster_id}: {objects}")
    
    return clusters

def main(args):
    
    bert = BERTEmbedder()

    # Load and pre-process semantic map
    semantic_maps = list()
    for semantic_map_file in os.listdir(constants.SEMANTIC_MAPS_FOLDER_PATH):

        semantic_map_basename = file_utils.get_file_basename(semantic_map_file)

        # Load semantic map
        semantic_map_obj = file_utils.load_json(os.path.join(constants.SEMANTIC_MAPS_FOLDER_PATH,
                                                             semantic_map_file))

        semantic_maps.append((semantic_map_basename, semantic_map_obj))

    # For each semantic map
    for (s_m_id, s_m_object) in semantic_maps[:args.number_maps]:
        
        print(s_m_id)
        # Pre-process semantic map
        s_m_object = preprocess.preprocess_semantic_map(s_m_object,
                                                   class_uncertainty=False)
        
        # Compute object semantic and geometric features
        semantic_features = dict()
        geometric_features = dict()

        # For each object
        for object_id, object_value in s_m_object["instances"].items():
            
            # Geometric feature = bounding box
            geometric_features[object_id] = object_value["bbox"]["center"]

            # Semantic feature = BERT embedding
            tag_text = next(iter(object_value["results"].keys()))
            # print(tag_text)
            semantic_features[object_id] = bert.embed_text(tag_text)

            # print(object_id, len(geometric_features[object_id]), len(semantic_features[object_id]))

        geometric_clusters = apply_clustering(geometric_features, 'Geometric Features')
        semantic_clusters = apply_clustering(semantic_features, 'Semantic Features')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Performs place categorization on a set of semantic map")

    parser.add_argument("-n", "--number-maps",
                        help="Number of semantic map to which place categorization will be applied.",
                        type=int,
                        default=10)
    
    args = parser.parse_args()

    main(args)