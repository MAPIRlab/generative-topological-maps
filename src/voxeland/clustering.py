from typing import Dict, List

from matplotlib import pyplot as plt
from sklearn.metrics import adjusted_rand_score, fowlkes_mallows_score, normalized_mutual_info_score, v_measure_score

from utils import file_utils
from voxeland.semantic_map_object import SemanticMapObject


class ObjectCluster:
    """Represents a cluster containing objects."""

    def __init__(self, cluster_id, object_labels):
        # Cluster identifier (DBSCAN assigns -1 to noise)
        self.cluster_id = cluster_id
        self.object_labels = object_labels  # List of objects in the cluster

    def __repr__(self):
        return f"ObjectCluster(cluster_id={self.cluster_id}, objects={self.object_labels})"


class ClusteringResult:
    """Represents a set of object clusters."""

    def __init__(self, clusters: List[ObjectCluster]):
        self.clusters = clusters  # List of ObjectCluster instances

    def get_cluster_by_id(self, cluster_id: int) -> ObjectCluster:
        """Returns the cluster with the specified cluster ID, or None if not found."""
        return next((cluster for cluster in self.clusters if cluster.cluster_id == cluster_id), None)

    def get_all_clusters(self) -> List[ObjectCluster]:
        """Returns the list of all clusters."""
        return self.clusters

    def get_noise_cluster(self) -> ObjectCluster:
        """Returns the cluster representing noise (DBSCAN assigns cluster ID -1)."""
        return self.get_cluster_by_id(-1)

    def get_cluster_count(self) -> int:
        """Returns the number of clusters (excluding noise)."""
        return sum(1 for cluster in self.clusters if cluster.cluster_id != -1)

    def get_total_object_count(self) -> int:
        """Returns the total number of objects across all clusters."""
        return sum(len(cluster.object_labels) for cluster in self.clusters)

    def visualize(self, semantic_map_objects: List[SemanticMapObject], title: str, file_path: str):
        """
        Visualizes the clustered objects from a top-down view and saves the plot.
        """
        objects_by_id = {
            obj.get_object_id(): obj for obj in semantic_map_objects}
        plt.figure(figsize=(8, 8))

        num_clusters = len(self.clusters)
        colors = plt.cm.get_cmap("tab10", num_clusters)

        legend_labels = []
        legend_patches = []
        x_min, x_max = float('inf'), float('-inf')
        y_min, y_max = float('inf'), float('-inf')

        for idx, cluster in enumerate(self.clusters):
            cluster_id = cluster.cluster_id
            object_ids = cluster.object_labels
            # Ensure color cycling if more than 10 clusters
            color = colors(idx % num_clusters)

            legend_labels.append(
                f"Cluster {cluster_id}" if cluster_id != -1 else "Noise")
            legend_patches.append(plt.Rectangle((0, 0), 1, 1, fc=color))

            for obj_id in object_ids:
                obj = objects_by_id.get(obj_id)
                if not obj:
                    continue

                x_center, y_center = obj.get_bbox_center()[:2]
                w, h = obj.get_bbox_size()[:2]
                half_w, half_h = w / 2, h / 2

                x_min = min(x_min, x_center - half_w)
                x_max = max(x_max, x_center + half_w)
                y_min = min(y_min, y_center - half_h)
                y_max = max(y_max, y_center + half_h)

                rect = plt.Rectangle((x_center - half_w, y_center - half_h), w, h,
                                     edgecolor=color, facecolor=color, alpha=0.5)
                plt.gca().add_patch(rect)

                label_text = f"{obj.get_object_id()}-{obj.get_most_probable_class()}"
                if cluster_id == -1:
                    label_text = f"({label_text})"
                plt.text(x_center, y_center, label_text,
                         fontsize=8, ha='center', va='center', color='black')

        # Adjust plot limits
        margin = 1
        plt.xlim(x_min - margin, x_max + margin)
        plt.ylim(y_min - margin, y_max + margin)

        plt.xlabel("X coordinate")
        plt.ylabel("Y coordinate")
        plt.title(title)
        plt.grid(True)
        plt.legend(legend_patches, legend_labels, loc="upper right")

        plt.savefig(file_path)
        plt.close()

    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, v_measure_score, fowlkes_mallows_score

    def evaluate_against_ground_truth(self, ground_truth_cr: "ClusteringResult") -> Dict[str, float]:
        """
        Evaluates this clustering result against a ground truth ClusteringResult.

        :param ground_truth_cr: The ground truth ClusteringResult to compare against.
        :return: Dictionary with evaluation metrics (ARI, NMI, V-Measure, FMI).
        """
        predicted_labels = []
        ground_truth_labels = []

        # Create mapping of object labels to their cluster IDs in the predicted result
        predicted_cluster_map = {obj: cluster.cluster_id for cluster in self.get_all_clusters(
        ) for obj in cluster.object_labels}
        # print(predicted_cluster_map)

        # Create mapping of object labels to their cluster IDs in the ground truth result
        ground_truth_cluster_map = {obj: cluster.cluster_id for cluster in ground_truth_cr.get_all_clusters(
        ) for obj in cluster.object_labels}
        # print(ground_truth_cluster_map)

        # Align ground truth and predicted labels
        for obj in ground_truth_cluster_map:
            if obj in predicted_cluster_map:
                predicted_labels.append(predicted_cluster_map[obj])
                ground_truth_labels.append(ground_truth_cluster_map[obj])

        if not predicted_labels:
            return {
                "ARI": 0,
                "NMI": 0,
                "V-Measure": 0,
                "FMI": 0,
            }

        return {
            "ARI": adjusted_rand_score(ground_truth_labels, predicted_labels),
            "NMI": normalized_mutual_info_score(ground_truth_labels, predicted_labels),
            "V-Measure": v_measure_score(ground_truth_labels, predicted_labels),
            "FMI": fowlkes_mallows_score(ground_truth_labels, predicted_labels),
        }

    def save_to_json(self, file_path: str):
        """
        Saves the clustering result in a JSON file with the structure:
        {
            "clusters": {
                "0": ["obj1", "obj2", "obj3"],
                "1": ["obj4", "obj5"]
            }
        }
        """
        clustering_data = {"clusters": {
            str(cluster.cluster_id): cluster.object_labels for cluster in self.clusters}}

        file_utils.save_dict_to_json_file(clustering_data, file_path)

    @staticmethod
    def load_from_json(file_path: str) -> "ClusteringResult":
        """
        Loads a ClusteringResult object from a JSON file with the structure:
        {
            "clusters": {
                "0": ["obj1", "obj2", "obj3"],
                "1": ["obj4", "obj5"]
            }
        }
        """
        clustering_data = file_utils.load_json(file_path)

        clusters = [
            ObjectCluster(int(cluster_id), object_labels)
            for cluster_id, object_labels in clustering_data["clusters"].items()
        ]

        return ClusteringResult(clusters)

    def __repr__(self):
        return f"Clustering(num_clusters={self.get_cluster_count()}, total_objects={self.get_total_object_count()})"
