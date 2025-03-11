from typing import Dict, List, Optional

from matplotlib import pyplot as plt
from sklearn.metrics import adjusted_rand_score, fowlkes_mallows_score, normalized_mutual_info_score, v_measure_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, v_measure_score, fowlkes_mallows_score
from utils import file_utils
from voxeland.cluster import Cluster
from voxeland.semantic_map import SemanticMap
from voxeland.semantic_map_object import SemanticMapObject


from typing import List, Optional


class Clustering:
    """Represents a set of object clusters."""

    def __init__(self, clusters: List[Cluster]):
        self.clusters = clusters  # List of ObjectCluster instances

    def append_cluster(self, cluster: Cluster):
        # TODO
        self.clusters.append(cluster)

    def find_cluster_by_id(self, cluster_id: int) -> Cluster:
        """Returns the cluster with the specified cluster ID, or None if not found."""
        return next((cluster for cluster in self.clusters if cluster.cluster_id == cluster_id), None)

    def get_cluster_count(self) -> int:
        """Returns the number of clusters (excluding noise)."""
        return sum(1 for cluster in self.clusters if cluster.cluster_id != -1)

    def get_total_object_count(self) -> int:
        """Returns the total number of objects across all clusters."""
        return sum(len(cluster.objects) for cluster in self.clusters)

    def visualize(self, title: str, file_path: str, semantic_map: Optional[SemanticMap] = None):
        """
        Visualizes the clustered objects from a top-down view and saves the plot.
        """
        plt.figure(figsize=(8, 8))

        num_clusters = len(self.clusters)
        colors = plt.cm.get_cmap("tab10", num_clusters)

        legend_labels = []
        legend_patches = []
        x_min, x_max = float('inf'), float('-inf')
        y_min, y_max = float('inf'), float('-inf')

        for idx, cluster in enumerate(self.clusters):
            # Ensure color cycling if more than 10 clusters
            color = colors(idx % num_clusters)

            legend_labels.append(
                f"Cluster {cluster.cluster_id}" if cluster.cluster_id != -1 else "Noise")
            legend_patches.append(plt.Rectangle((0, 0), 1, 1, fc=color))

            for obj in cluster.objects:

                if obj.bbox_center is None:
                    complete_obj = semantic_map.find_object(
                        obj.get_object_id())
                else:
                    complete_obj = obj

                x_center, y_center = complete_obj.get_bbox_center()[:2]
                w, h = complete_obj.get_bbox_size()[:2]
                half_w, half_h = w / 2, h / 2

                x_min = min(x_min, x_center - half_w)
                x_max = max(x_max, x_center + half_w)
                y_min = min(y_min, y_center - half_h)
                y_max = max(y_max, y_center + half_h)

                rect = plt.Rectangle((x_center - half_w, y_center - half_h), w, h,
                                     edgecolor=color, facecolor=color, alpha=0.5)
                plt.gca().add_patch(rect)

                object_label_text = f"{complete_obj.get_object_id()}-{complete_obj.get_most_probable_class()}"
                if cluster.cluster_id == -1:
                    object_label_text = f"({object_label_text})"
                plt.text(x_center, y_center, object_label_text,
                         fontsize=8, ha='center', va='center', color='black')

            cluster_label_text = f"cluster{cluster.cluster_id}\n{cluster.compute_semantic_descriptor_variance():.5f}"
            plt.text(cluster.compute_geometric_descriptor()[0], cluster.compute_geometric_descriptor()[1], cluster_label_text,
                     fontsize=16, ha='center', va='center', color='black')

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

    def evaluate_against_ground_truth(self, ground_truth_cr: "Clustering") -> Dict[str, float]:
        """
        Evaluates this clustering result against a ground truth ClusteringResult.

        :param ground_truth_cr: The ground truth ClusteringResult to compare against.
        :return: Dictionary with evaluation metrics (ARI, NMI, V-Measure, FMI).
        """
        predicted_labels = []
        ground_truth_labels = []

        # Create mapping of object labels to their cluster IDs in the predicted result
        predicted_cluster_map = {
            obj.object_id: cluster.cluster_id for cluster in self.clusters for obj in cluster.objects}
        print(
            f"[Clustering.evaluate_against_ground_truth] Predicted cluster map: {predicted_cluster_map}")

        # Create mapping of object labels to their cluster IDs in the ground truth result
        ground_truth_cluster_map = {
            obj.object_id: cluster.cluster_id for cluster in ground_truth_cr.clusters for obj in cluster.objects}
        print(
            f"[Clustering.evaluate_against_ground_truth] Ground truth cluster map: {ground_truth_cluster_map}")

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

    def merge_clusters(self, cluster_1_id: str, cluster_2_id: str):
        cluster_1 = self.find_cluster_by_id(cluster_1_id)
        cluster_2 = self.find_cluster_by_id(cluster_2_id)

        cluster_1.objects.extend(cluster_2.objects)
        self.clusters.remove(cluster_2)

    def save_to_json(self, file_path: str):
        """
        Saves the clustering result in a JSON file with the updated structure:
        {
            "clusters": {
                "0": {
                    "description": "sleeping and working area, with a bed and desk",
                    "objects": ["obj1", "obj10", ...]
                },
                "1": {
                    "tag": "leisure area with sofas",
                    "objects": ["obj101", "obj67", ...]
                }
            }
        }
        """
        clustering_data = {
            "clusters": {
                str(cluster.cluster_id): {
                    "description": cluster.description,
                    "objects": list(map(lambda obj: obj.object_id, cluster.objects))
                }
                for cluster in self.clusters
            }
        }
        print(clustering_data)
        file_utils.save_dict_to_json_file(clustering_data, file_path)

    @staticmethod
    def load_from_json(file_path: str) -> "Clustering":
        """
        Loads a ClusteringResult object from a JSON file with the updated structure.
        """
        clustering_data = file_utils.load_json(file_path)

        clusters = list()
        for cluster_id, data in clustering_data["clusters"].items():
            objects = list()
            for object_id in data["objects"]:
                objects.append(SemanticMapObject(object_id, None))
            clusters.append(Cluster(int(cluster_id),
                                    objects,
                                    data.get("description")))

        return Clustering(clusters)

    def __repr__(self):
        return f"Clustering(num_clusters={self.get_cluster_count()}, total_objects={self.get_total_object_count()})"
