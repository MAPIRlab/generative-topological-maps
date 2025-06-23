from typing import Dict, List, Optional

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.metrics import (
    adjusted_rand_score,
    fowlkes_mallows_score,
    normalized_mutual_info_score,
    v_measure_score,
)

from generative_place_categorization.utils import file_utils
from generative_place_categorization.voxeland.cluster import Cluster
from generative_place_categorization.voxeland.semantic_map import SemanticMap
from generative_place_categorization.voxeland.semantic_map_object import (
    SemanticMapObject,
)


class Clustering:
    """Represents a set of object clusters."""

    def __init__(self, clusters: List[Cluster] = []):
        self.clusters = clusters  # List of ObjectCluster instances

    def append_cluster(self, cluster: Cluster):
        """Appends a cluster to the list of clusters."""
        self.clusters.append(cluster)

    def remove_cluster_by_id(self, cluster_id: int) -> List[Cluster]:
        """Removes the cluster with the specified cluster ID."""
        self.clusters = [
            cluster for cluster in self.clusters if cluster.cluster_id != cluster_id]

    def extend(self, clustering: "Clustering"):
        """Extends the current clustering with another clustering."""
        existing_ids = {cluster.cluster_id for cluster in self.clusters}
        for cluster in clustering.clusters:
            while cluster.cluster_id in existing_ids:
                cluster.cluster_id += 1
            existing_ids.add(cluster.cluster_id)

        self.clusters.extend(clustering.clusters)

    def find_cluster_by_id(self, cluster_id: int) -> Cluster:
        """Returns the cluster with the specified cluster ID, or None if not found."""
        return next((cluster for cluster in self.clusters if cluster.cluster_id == cluster_id), None)

    def get_cluster_count(self) -> int:
        """Returns the number of clusters (excluding noise)."""
        return sum(1 for cluster in self.clusters if cluster.cluster_id != -1)

    def get_total_object_count(self) -> int:
        """Returns the total number of objects across all clusters."""
        return sum(len(cluster.objects) for cluster in self.clusters)

    def get_next_available_cluster_id(self) -> int:
        """Returns the next available cluster ID."""
        if not self.clusters:
            return 0
        return max(cluster.cluster_id for cluster in self.clusters) + 1

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
        # print(
        #     f"[Clustering.evaluate_against_ground_truth] Predicted cluster map: {predicted_cluster_map}")

        # Create mapping of object labels to their cluster IDs in the ground truth result
        ground_truth_cluster_map = {
            obj.object_id: cluster.cluster_id for cluster in ground_truth_cr.clusters for obj in cluster.objects}
        # print(
        #     f"[Clustering.evaluate_against_ground_truth] Ground truth cluster map: {ground_truth_cluster_map}")

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
        print("merging")
        print(cluster_1.objects)
        print(cluster_2.objects)
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
                    "tag": cluster.tag,
                    "objects": list(map(lambda obj: obj.object_id, cluster.objects))
                }
                for cluster in self.clusters
            }
        }
        file_utils.save_dict_to_json_file(clustering_data, file_path)

    @staticmethod
    def load_from_json(file_path: str, semantic_map: Optional[SemanticMap] = None) -> "Clustering":
        """
        Loads a ClusteringResult object from a JSON file with the updated structure. If a semantic map is provided,
        it will be used to find the objects in the clusters.
        """
        clustering_data = file_utils.load_json(file_path)

        clusters = list()
        for cluster_id, data in clustering_data["clusters"].items():
            objects = list()
            for object_id in data["objects"]:
                if semantic_map is not None:  # if semantic map -> search object
                    object = semantic_map.find_object(object_id)
                    if object is None:
                        raise ValueError(
                            f"Object {object_id} not found in semantic map {semantic_map.semantic_map_id}.")
                    objects.append(object)  # else -> add object
                else:
                    objects.append(SemanticMapObject(object_id, None))

            clusters.append(Cluster(int(cluster_id),
                                    objects,
                                    data.get("description")))

        return Clustering(clusters)

    def visualize_2D(self, title: str, semantic_map: SemanticMap, geometric_threshold: Optional[float] = None, file_path: Optional[str] = None):
        """
        Visualizes the clustered objects from a top-down view and saves the plot if file_path is specified.
        Otherwise, displays the plot on the screen.
        """
        print("self", self)
        print("title", title)
        print("semantic_map", semantic_map)
        plt.figure(figsize=(8, 8))

        num_clusters = len(self.clusters)
        colors = plt.cm.get_cmap("tab10", num_clusters)

        legend_labels = []
        legend_patches = []
        x_min, x_max = float('inf'), float('-inf')
        y_min, y_max = float('inf'), float('-inf')

        cluster_centers = {}

        for idx, cluster in enumerate(self.clusters):
            color = colors(idx % num_clusters)
            legend_labels.append(
                f"Cluster {cluster.cluster_id}" if cluster.cluster_id != -1 else "Noise")
            legend_patches.append(plt.Rectangle((0, 0), 1, 1, fc=color))

            for obj in cluster.objects:
                complete_obj = semantic_map.find_object(obj.object_id)
                if obj.bbox_center is None:
                    complete_obj = semantic_map.find_object(obj.object_id)
                else:
                    complete_obj = obj

                x_center, y_center = complete_obj.bbox_center[:2]
                w, h = complete_obj.bbox_size[:2]
                half_w, half_h = w / 2, h / 2

                x_min = min(x_min, x_center - half_w)
                x_max = max(x_max, x_center + half_w)
                y_min = min(y_min, y_center - half_h)
                y_max = max(y_max, y_center + half_h)

                rect = plt.Rectangle((x_center - half_w, y_center - half_h), w, h,
                                     edgecolor=color, facecolor=color, alpha=0.5)
                plt.gca().add_patch(rect)

                object_label_text = f"{complete_obj.object_id}-{complete_obj.get_most_probable_class()}"
                if cluster.cluster_id == -1:
                    object_label_text = f"({object_label_text})"
                plt.text(x_center, y_center, object_label_text,
                         fontsize=8, ha='center', va='center', color='black')

            cluster_center = cluster.compute_center()[:2]
            cluster_centers[cluster.cluster_id] = cluster_center

            # Draw a circle around the cluster center with the specified geometric threshold
            # if geometric_threshold is not None:
            #     circle = plt.Circle(cluster_center, geometric_threshold,
            #                         color=color, fill=False, linestyle='dashed', alpha=0.7)
            #     plt.gca().add_patch(circle)

            # Add a point in the center of the cluster
            plt.plot(cluster_center[0], cluster_center[1],
                     'o', color='black', markersize=8)

            cluster_label_text = f"cluster{cluster.cluster_id}"
            try:
                cluster_label_text += f"\n{cluster.compute_splitting_score():.5f}"
            except ValueError:
                pass

            plt.text(cluster_center[0], cluster_center[1], cluster_label_text,
                     fontsize=16, ha='center', va='center', color='black')

            if cluster.description:
                plt.text(cluster_center[0], cluster_center[1] - 0.25, cluster.description,
                         fontsize=11, ha='center', va='center', color='black')

        # # Draw lines between cluster centers with semantic similarity annotations
        # for i, cluster_a in enumerate(self.clusters):
        #     for j, cluster_b in enumerate(self.clusters):
        #         if j <= i:
        #             continue  # avoid duplicate pairs
        #         try:
        #             similarity = cluster_a.compute_semantic_similarity_to(
        #                 cluster_b)
        #             center_a = cluster_centers[cluster_a.cluster_id]
        #             center_b = cluster_centers[cluster_b.cluster_id]
        #             mid_x, mid_y = (
        #                 center_a[0] + center_b[0]) / 2, (center_a[1] + center_b[1]) / 2
        #             plt.plot([center_a[0], center_b[0]], [center_a[1], center_b[1]],
        #                      linestyle='dotted', color='gray', alpha=0.5)
        #             plt.text(mid_x, mid_y, f"{similarity:.2f}", fontsize=9, color='gray',
        #                      ha='center', va='center', backgroundcolor='white')
        #         except ValueError:
        #             pass  # skip if semantic descriptor computation fails

        # Adjust plot limits
        margin = 1
        plt.xlim(x_min - margin, x_max + margin)
        plt.ylim(y_min - margin, y_max + margin)

        plt.xlabel("X coordinate")
        plt.ylabel("Y coordinate")
        plt.title(title)
        plt.grid(True)

        # Place the legend outside the graph
        plt.legend(legend_patches, legend_labels,
                   loc="center left", bbox_to_anchor=(1, 0.5))

        if file_path:
            plt.savefig(file_path, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def visualize_3D(self, title: str, semantic_map: SemanticMap, file_path: Optional[str] = None):
        """
        Visualizes the clusters in 3D with bounding boxes and colors. Saves the plot if file_path is specified.
        Otherwise, displays the plot on the screen.
        """

        # Generate unique colors for each cluster
        norm = mcolors.Normalize(vmin=0, vmax=len(self.clusters) - 1)
        colormap = cm.get_cmap("tab10", len(self.clusters))
        colors = [colormap(norm(i)) for i in range(len(self.clusters))]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Keep track of min/max values to preserve the natural scale
        all_x, all_y, all_z = [], [], []

        for cluster, color in zip(self.clusters, colors):
            for obj in cluster.objects:
                # Use the semantic map to find the complete object if available
                complete_obj = semantic_map.find_object(
                    obj.object_id) if semantic_map else obj
                center = complete_obj.bbox_center
                size = complete_obj.bbox_size

                x = [center[0] - size[0] / 2, center[0] + size[0] / 2]
                y = [center[1] - size[1] / 2, center[1] + size[1] / 2]
                z = [center[2] - size[2] / 2, center[2] + size[2] / 2]

                # Accumulate all bounding box corners for min/max tracking
                all_x.extend(x)
                all_y.extend(y)
                all_z.extend(z)

                vertices = [
                    [x[0], y[0], z[0]],
                    [x[1], y[0], z[0]],
                    [x[1], y[1], z[0]],
                    [x[0], y[1], z[0]],
                    [x[0], y[0], z[1]],
                    [x[1], y[0], z[1]],
                    [x[1], y[1], z[1]],
                    [x[0], y[1], z[1]],
                ]

                faces = [
                    [vertices[j] for j in [0, 1, 5, 4]],
                    [vertices[j] for j in [1, 2, 6, 5]],
                    [vertices[j] for j in [2, 3, 7, 6]],
                    [vertices[j] for j in [3, 0, 4, 7]],
                    [vertices[j] for j in [0, 1, 2, 3]],
                    [vertices[j] for j in [4, 5, 6, 7]],
                ]
                poly3d = Poly3DCollection(
                    faces, alpha=0.5, facecolors=color, edgecolors="black"
                )
                ax.add_collection3d(poly3d)

                ax.text(center[0], center[1], center[2],
                        complete_obj.object_id, fontsize=8, color="black")

        # Set axis labels
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.title(title)

        # -------------------------------------------------------------
        # 1) Determine the data range in each dimension
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        min_z, max_z = min(all_z), max(all_z)

        # 2) Set the axis limits to exactly match the data extents
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_zlim(min_z, max_z)

        # 3) Preserve the native dimension ratio of the data
        #    Requires Matplotlib 3.3 or higher
        range_x = max_x - min_x
        range_y = max_y - min_y
        range_z = max_z - min_z
        ax.set_box_aspect((range_x, range_y, range_z))
        # -------------------------------------------------------------

        if file_path:
            plt.savefig(file_path, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def __repr__(self):
        return f"Clustering(num_clusters={self.get_cluster_count()}, total_objects={self.get_total_object_count()})"
