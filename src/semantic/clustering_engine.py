from sklearn.cluster import DBSCAN
from hdbscan import HDBSCAN
from scipy.spatial.distance import pdist, squareform
from functools import partial
import numpy as np
import constants
from voxeland.clustering import Clustering
from voxeland.cluster import Cluster


def geometric_semantic_distance(A, B, semantic_weight: float = 0.005):
    """
    Custom distance:
    - Euclidean distance for the first three dimensions (geometric)
    - Plus a weighted, dimension-normalized Euclidean distance for the rest (semantic).
    """
    A, B = np.array(A), np.array(B)

    # Geometric distance
    geometric_dist = np.sqrt(np.sum((A[:3] - B[:3]) ** 2))

    # Semantic distance
    semantic_dim = max(0, len(A) - 3)
    if semantic_dim > 0:
        raw_semantic_dist = np.sqrt(np.sum((A[3:] - B[3:]) ** 2))
        norm_semantic_dist = raw_semantic_dist / np.sqrt(semantic_dim)
        semantic_dist = norm_semantic_dist * semantic_weight
    else:
        semantic_dist = 0.0

    # Global distance
    global_dist = geometric_dist + semantic_dist

    return global_dist


class ClusteringEngine:
    """
    Engine for applying clustering algorithms (DBSCAN or HDBSCAN).
    """

    def apply_clustering(self, semantic_map, algorithm, eps=1.0, min_samples=1, semantic_weight=0.005):
        """
        Apply the specified clustering algorithm to the semantic map.

        Args:
            semantic_map (SemanticMap): The semantic map to cluster.
            algorithm (str): The clustering algorithm ('dbscan' or 'hdbscan').
            eps (float): Epsilon parameter for DBSCAN.
            min_samples (int): Minimum samples parameter for DBSCAN/HDBSCAN.
            semantic_weight (float): Weight for semantic distance.

        Returns:
            Clustering: The resulting clustering object.
        """
        object_labels = list(
            map(lambda obj: obj.object_id, semantic_map.get_all_objects()))
        object_descriptors = list(
            map(lambda obj: obj.global_descriptor, semantic_map.get_all_objects()))

        # Compute custom distance matrix
        distance_matrix = squareform(
            pdist(object_descriptors, metric=partial(
                geometric_semantic_distance, semantic_weight=semantic_weight))
        )

        # Apply clustering algorithm
        if algorithm == constants.CLUSTERING_ALGORITHM_DBSCAN:
            clustering_model = DBSCAN(
                eps=eps, min_samples=min_samples, metric="precomputed")
        elif algorithm == constants.CLUSTERING_ALGORITHM_HDBSCAN:
            clustering_model = HDBSCAN(
                min_cluster_size=min_samples, metric="precomputed")
        else:
            raise ValueError(f"Unsupported clustering algorithm: {algorithm}")

        labels = clustering_model.fit_predict(distance_matrix)

        # Control clusterings with all points noisy
        if np.all(labels == -1):
            labels[:] = 0

        # Create a Clustering object
        clustering = Clustering([])
        for object_label, object_cluster in zip(object_labels, labels):

            # Skip noise points
            if object_cluster == -1:  # noise
                continue
            else:
                cluster = clustering.find_cluster_by_id(object_cluster)
                if cluster is None:  # new cluster
                    clustering.append_cluster(Cluster(object_cluster, [], ""))
                    cluster = clustering.find_cluster_by_id(object_cluster)
                cluster.append_object(semantic_map.find_object(object_label))

        # Assign noise points to the closest cluster
        for object_label, object_cluster in zip(object_labels, labels):
            if object_cluster == -1:
                obj = semantic_map.find_object(object_label)
                # Find the geometrically closest cluster
                closest_cluster = None
                closest_distance = float('inf')
                for cluster in clustering.clusters:
                    cluster_center = np.mean(
                        [obj.bbox_center for obj in cluster.objects], axis=0)
                    distance = np.linalg.norm(
                        obj.bbox_center - cluster_center)
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_cluster = cluster

                # Assign the object to the closest cluster
                if closest_cluster is not None:
                    print(
                        f"Assigning noise object {obj.object_id} to cluster {closest_cluster.cluster_id}")
                    closest_cluster.append_object(obj)

        return clustering
