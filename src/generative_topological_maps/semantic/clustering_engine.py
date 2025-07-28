from functools import partial
from itertools import combinations
from typing import Tuple

import numpy as np
from hdbscan import HDBSCAN
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN, KMeans

from generative_topological_maps import constants
from generative_topological_maps.utils import file_utils
from generative_topological_maps.voxeland.cluster import Cluster
from generative_topological_maps.voxeland.clustering import Clustering
from generative_topological_maps.voxeland.semantic_map import SemanticMap


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

    def clusterize(self, semantic_map: SemanticMap, algorithm: str, eps=1.0, min_samples=1, semantic_weight=0.005, n_clusters=2, noise_objects_new_clusters: bool = False):
        """
        Apply the specified clustering algorithm to the semantic map.

        Args:
            semantic_map (SemanticMap): The semantic map to cluster.
            algorithm (str): The clustering algorithm ('dbscan', 'hdbscan', 'kmeans').
            eps (float): Epsilon parameter for DBSCAN.
            min_samples (int): Minimum samples parameter for DBSCAN/HDBSCAN.
            semantic_weight (float): Weight for semantic distance.
            n_clusters (int, optional): Number of clusters for KMeans.

        Returns:
            Clustering: The resulting clustering object.
        """
        print("[ClusteringEngine.clusterize] Applying clustering...")
        object_ids = [
            obj.object_id for obj in semantic_map.get_all_objects()]
        object_descriptors = [
            obj.global_descriptor for obj in semantic_map.get_all_objects()]

        if algorithm in [constants.CLUSTERING_ALGORITHM_DBSCAN, constants.CLUSTERING_ALGORITHM_HDBSCAN]:
            # Compute custom distance matrix
            distance_matrix = squareform(
                pdist(object_descriptors, metric=partial(
                    geometric_semantic_distance, semantic_weight=semantic_weight))
            )

            if algorithm == constants.CLUSTERING_ALGORITHM_DBSCAN:
                clustering_model = DBSCAN(
                    eps=eps, min_samples=min_samples, metric="precomputed")
            elif algorithm == constants.CLUSTERING_ALGORITHM_HDBSCAN:
                clustering_model = HDBSCAN(
                    min_cluster_size=min_samples, metric="precomputed")

            cluster_labels = clustering_model.fit_predict(distance_matrix)

        elif algorithm == constants.CLUSTERING_ALGORITHM_KMEANS:
            clustering_model = KMeans(n_clusters=n_clusters, random_state=0)
            cluster_labels = clustering_model.fit_predict(object_descriptors)
        else:
            raise ValueError(f"Unsupported clustering algorithm: {algorithm}")

        # Count the number of unique clusters (excluding noise if present)
        num_clusters = len(set(cluster_labels)) - \
            (1 if -1 in cluster_labels else 0)
        print(
            f"[ClusteringEngine.clusterize] Number of clusters: {num_clusters}")

        clustering = Clustering([])

        # Create clusters based on labels
        for object_label, object_cluster in zip(object_ids, cluster_labels):
            # Skip noise points for DBSCAN/HDBSCAN
            if object_cluster == -1 and algorithm != constants.CLUSTERING_ALGORITHM_KMEANS:
                continue
            else:
                cluster = clustering.find_cluster_by_id(object_cluster)
                if cluster is None:
                    clustering.append_cluster(Cluster(object_cluster, [], ""))
                    cluster = clustering.find_cluster_by_id(object_cluster)
                cluster.append_object(semantic_map.find_object(object_label))

        # Handle noise reassignment (only for DBSCAN/HDBSCAN)
        if algorithm in [constants.CLUSTERING_ALGORITHM_DBSCAN, constants.CLUSTERING_ALGORITHM_HDBSCAN]:
            print(
                f"[ClusteringEngine.clusterize] WARNING! There were {list(cluster_labels).count(-1)}/{len(cluster_labels)} noise points.")

            if noise_objects_new_clusters:
                for object_label, object_cluster in zip(object_ids, cluster_labels):
                    if object_cluster == -1:
                        obj = semantic_map.find_object(object_label)
                        new_cluster = Cluster(cluster_id=clustering.get_next_available_cluster_id(),
                                              objects=[obj])
                        clustering.append_cluster(new_cluster)
            else:
                if list(cluster_labels).count(-1) == len(cluster_labels):  # ALL noisy points
                    only_cluster = Cluster(
                        -1, list(map(lambda object_id: semantic_map.find_object(object_id), object_ids)))
                    clustering.append_cluster(only_cluster)

                else:  # NOT ALL noisy points
                    for object_label, object_cluster in zip(object_ids, cluster_labels):
                        if object_cluster == -1:
                            obj = semantic_map.find_object(object_label)
                            closest_cluster = None
                            closest_distance = float('inf')
                            for cluster in clustering.clusters:
                                cluster_center = np.mean(
                                    [o.bbox_center for o in cluster.objects], axis=0)
                                distance = np.linalg.norm(
                                    obj.bbox_center - cluster_center)
                                if distance < closest_distance:
                                    closest_distance = distance
                                    closest_cluster = cluster
                            if closest_cluster is not None:
                                print(
                                    f"[ClusteringEngine.clusterize] Noisy object {obj.object_id} -> cluster {closest_cluster.cluster_id}")
                                closest_cluster.append_object(obj)

        return clustering

    def post_process_clustering(self, semantic_map: SemanticMap, clustering: Clustering, merge_geometric_threshold: float, merge_semantic_threshold: float, split_semantic_threshold: float, clustering_json_file_path: str, clustering_plot_file_path: str) -> Clustering:

        num_phases = 0
        max_num_phases = 5

        # Show clusters
        iteration_json_file_path = clustering_json_file_path.replace(
            ".json", f"{num_phases}.json")
        iteration_plot_file_path = clustering_plot_file_path.replace(
            ".png", f"{num_phases}.png")
        file_utils.create_directories_for_file(iteration_json_file_path)
        file_utils.create_directories_for_file(iteration_plot_file_path)
        clustering.save_to_json(iteration_json_file_path)
        clustering.visualize_2D(
            f"Clustering before post-processing {num_phases}",
            semantic_map,
            geometric_threshold=merge_geometric_threshold,
            file_path=iteration_plot_file_path)

        while num_phases < max_num_phases:

            # Decide clusters to split
            cluster_to_be_split = self.decide_cluster_to_be_split(
                clustering, split_semantic_threshold)
            print(
                f"[ClusteringEngine.post_process_clustering] Cluster to be split: {cluster_to_be_split}")

            if cluster_to_be_split is not None:
                # Split clusters
                clustering, _ = self.split_clusters(
                    clustering, [cluster_to_be_split])

                # Show clusters
                iteration_json_file_path = clustering_json_file_path.replace(
                    ".json", f"{num_phases}_after_split.json")
                iteration_plot_file_path = clustering_plot_file_path.replace(
                    ".png", f"{num_phases}_after_split.png")
                file_utils.create_directories_for_file(
                    iteration_json_file_path)
                file_utils.create_directories_for_file(
                    iteration_plot_file_path)
                clustering.save_to_json(iteration_json_file_path)
                clustering.visualize_2D(
                    f"[{num_phases}] Clustering after splitting {cluster_to_be_split}",
                    semantic_map,
                    geometric_threshold=merge_geometric_threshold,
                    file_path=iteration_plot_file_path)

            clusters_to_be_merged = self.decide_cluster_to_be_merged(
                clustering, merge_geometric_threshold, merge_semantic_threshold)

            merge_iteration = 0
            while clusters_to_be_merged is not None:

                # Merge clusters
                clustering = self.merge_clusters(
                    clustering, [clusters_to_be_merged])

                # Show clusters
                iteration_json_file_path = clustering_json_file_path.replace(
                    ".json", f"{num_phases}_after_merge_{merge_iteration}.json")
                iteration_plot_file_path = clustering_plot_file_path.replace(
                    ".png", f"{num_phases}_after_merge_{merge_iteration}.png")
                file_utils.create_directories_for_file(
                    iteration_json_file_path)
                file_utils.create_directories_for_file(
                    iteration_plot_file_path)
                clustering.save_to_json(iteration_json_file_path)
                clustering.visualize_2D(
                    f"[{num_phases}] Clustering after merging {clusters_to_be_merged} it {merge_iteration}",
                    semantic_map,
                    geometric_threshold=merge_geometric_threshold,
                    file_path=iteration_plot_file_path)

                merge_iteration += 1

                clusters_to_be_merged = self.decide_cluster_to_be_merged(
                    clustering, merge_geometric_threshold, merge_semantic_threshold)

            # Decide clusters to split
            cluster_to_be_split = self.decide_cluster_to_be_split(
                clustering, split_semantic_threshold)

            num_phases += 1

        return clustering

    def merge_clusters(self, clustering: Clustering, clusters_to_be_merged: Tuple[str, str]):
        """Merge the specified clusters in the clustering object."""
        for cluster_1_id, cluster_2_id in clusters_to_be_merged:
            if clustering.find_cluster_by_id(cluster_1_id) is not None and clustering.find_cluster_by_id(cluster_2_id) is not None:
                clustering.merge_clusters(cluster_1_id, cluster_2_id)
        print(f"[ClusteringEngine.merge_clusters] Merged clusters {clusters_to_be_merged}",
              )
        return clustering

    def decide_cluster_to_be_merged(self, clustering: Clustering, merge_geometric_threshold: float, merge_semantic_threshold: float):
        """Decide which clusters should be merged based on geometric and semantic thresholds."""
        clusters_to_be_merged = None

        max_similarity = -1.0

        for cluster_1, cluster_2 in combinations(clustering.clusters, 2):

            cluster_1: Cluster = cluster_1
            cluster_2: Cluster = cluster_2

            if cluster_1.compute_geometric_euclidean_distance_to(cluster_2) <= merge_geometric_threshold:

                # Compute semantic distance
                similarity = cluster_1.compute_semantic_similarity_to(
                    cluster_2)

                # Check if semantic distance is below the threshold
                if similarity >= merge_semantic_threshold and similarity > max_similarity:
                    clusters_to_be_merged = (cluster_1.cluster_id,
                                             cluster_2.cluster_id)

        print(
            f"[Clustering.decide_cluster_to_be_merged] Cluster to be merged: {clusters_to_be_merged}")
        return clusters_to_be_merged

    def decide_cluster_to_be_split(self, clustering: Clustering, split_semantic_threshold: float):
        """Decide which clusters should be split based on TODO."""
        cluster_to_be_split = None
        max_splitting_score = 0.0

        for cluster in clustering.clusters:

            splitting_score = cluster.compute_splitting_score()
            print(
                f"[ClusteringEngine.decide_cluster_to_be_split] Splitting score for cluster {cluster.cluster_id}: {splitting_score}")
            if splitting_score >= split_semantic_threshold and splitting_score >= max_splitting_score:
                cluster_to_be_split = cluster.cluster_id
                max_splitting_score = splitting_score

        print(
            f"[ClusteringEngine.decide_cluster_to_be_split] Cluster to be split: {cluster_to_be_split}")
        return cluster_to_be_split

    def split_clusters(self, clustering: Clustering, clusters_to_be_split: list[str]):
        """Split the specified clusters in the clustering object."""
        for cluster_to_be_split_id in clusters_to_be_split:
            print(
                f"[ClusteringEngine.split_clusters] Cluster {cluster_to_be_split_id} will be split!")
            # Find cluster
            cluster = clustering.find_cluster_by_id(cluster_to_be_split_id)

            # Build semantic map with objects
            semantic_map = SemanticMap(
                "splitting semantic map", cluster.objects)

            # Apply clustering on semantic map
            new_clustering = self.clusterize(
                semantic_map, constants.CLUSTERING_ALGORITHM_DBSCAN, eps=1, min_samples=2, semantic_weight=0.5, noise_objects_new_clusters=True)

            if len(new_clustering.clusters) > 1:
                print(
                    "[ClusteringEngine.split_clusters] Splitting cluster into multiple clusters!")

                # Remove original cluster
                clustering.remove_cluster_by_id(cluster_to_be_split_id)

                # Extend original clustering with new clustering
                clustering.extend(new_clustering)
            else:
                print(
                    "[ClusteringEngine.split_clusters] WARNING! Splitting resulted in only one cluster! No real splitting done.")

        return clustering, new_clustering.get_cluster_count()
