from typing import List, Optional

import numpy as np

from generative_place_categorization.voxeland.semantic_map_object import (
    SemanticMapObject,
)


class Cluster:
    """Represents a cluster containing objects with an optional description or tag."""

    def __init__(self, cluster_id: int, objects: List[SemanticMapObject], description: Optional[str] = None):
        """
        Initializes a cluster with a unique identifier, a list of objects, and an optional description.

        Args:
            cluster_id (int): Unique identifier for the cluster (DBSCAN assigns -1 to noise).
            objects (List[SemanticMapObject]): List of semantic map objects that belong to the cluster.
            description (Optional[str]): Optional description or tag for the cluster.
        """
        self.cluster_id = cluster_id
        self.objects = objects
        self.description = description

    def append_object(self, semantic_map_object: SemanticMapObject):
        """Adds a new object to the cluster."""
        self.objects.append(semantic_map_object)

    def find_object(self, object_id: str) -> Optional[SemanticMapObject]:
        """Finds and returns an object by its ID."""
        for obj in self.objects:
            if obj.object_id == object_id:
                return obj
        return None

    def compute_semantic_descriptor(self) -> np.ndarray:
        """
        Computes the semantic descriptor of the cluster by averaging the semantic descriptors
        of the most probable class of each object.
        """
        descriptors = [obj.semantic_descriptor for obj in self.objects]
        # print("lista de descriptores", len(descriptors))
        # print("cada descriptor", len(descriptors[0]))
        if None in descriptors:
            raise ValueError(
                "Cannot compute semantic descriptor of cluster containing objects with None semantic descriptor")
        # print("su media", np.mean(descriptors, axis=0) if descriptors else np.zeros_like(
        #     self.objects[0].semantic_descriptor))
        return np.mean(descriptors, axis=0) if descriptors else np.zeros_like(self.objects[0].semantic_descriptor)

    def compute_center(self) -> np.ndarray:
        """Computes the cluster center from objects bounding box centers."""
        bbox_centers = [object.bbox_center for object in self.objects]
        if None in bbox_centers:
            raise ValueError(
                "Cannot compute center of a cluster containing objects with None bounding box center")
        return np.mean(bbox_centers, axis=0) if bbox_centers else np.zeros_like(self.objects[0].bbox_center)

    def compute_semantic_descriptor_variance(self) -> float:
        """
        Computes a normalized semantic variance, independent of the number of objects.
        It is defined as the average squared distance to the centroid,
        divided by the maximum squared distance observed.
        """
        if len(self.objects) == 1:
            return 0.0

        cluster_descriptor = self.compute_semantic_descriptor()
        squared_distances = [
            (obj.semantic_descriptor - cluster_descriptor) ** 2
            for obj in self.objects
        ]
        return np.sum(squared_distances) / (len(self.objects)-1)

    def compute_semantic_similarity_to(self, other_cluster: "Cluster") -> float:
        """
        Computes the cosine similarity between two cluster embeddings.
        """
        self_descriptor = self.compute_semantic_descriptor()
        other_descriptor = other_cluster.compute_semantic_descriptor()

        self_norm = np.linalg.norm(self_descriptor)
        other_norm = np.linalg.norm(other_descriptor)

        return np.dot(self_descriptor, other_descriptor) / (self_norm * other_norm) if self_norm and other_norm else 0.0

    def _boxes_overlap(self, center1, size1, center2, size2) -> bool:
        """Helper function to check if two bounding boxes overlap in 3D from above, i.e., in the X and Y axis."""
        return all(
            abs(center1[i] - center2[i]) <= (size1[i] / 2 + size2[i] / 2)
            for i in range(2)
        )

    def compute_overlapping_to(self, other_cluster: "Cluster") -> bool:
        """
        Checks if two clusters have overlapping bounding boxes by iterating through each object's bounding box in both clusters.
        """
        return any(
            self._boxes_overlap(obj1.bbox_center, obj1.bbox_size,
                                obj2.bbox_center, obj2.bbox_size)
            for obj1 in self.objects for obj2 in other_cluster.objects
        )

    def compute_geometric_euclidean_distance_to(self, other_cluster: "Cluster") -> float:
        """
        Computes the Euclidean distance between the geometric descriptors of two clusters.
        """
        self_geometric_descriptor = self.compute_center()
        other_geometric_descriptor = other_cluster.compute_center()
        return np.linalg.norm(self_geometric_descriptor - other_geometric_descriptor)

    def compute_splitting_score(self) -> float:
        """
        Computes a splitting score for the cluster. The score is influenced by the semantic descriptor variance
        and the average pairwise geometric distance between objects in the cluster.
        A higher score indicates that the cluster is more likely to be split.
        """
        if len(self.objects) == 1:
            return 0.0

        semantic_variance = self.compute_semantic_descriptor_variance()

        pairwise_distances = [
            np.linalg.norm(np.array(obj1.bbox_center) -
                           np.array(obj2.bbox_center))
            for i, obj1 in enumerate(self.objects)
            for j, obj2 in enumerate(self.objects)
            if i < j
        ]
        avg_pairwise_distance = np.mean(pairwise_distances)

        # Combine semantic variance and average pairwise distance into the splitting score
        splitting_score = 0.6 * semantic_variance + 0.4 * avg_pairwise_distance

        return splitting_score

    def __repr__(self):
        return (f"Cluster(cluster_id={self.cluster_id}, description={self.description}, "
                f"objects={[obj.object_id for obj in self.objects]}, "
                f"semantic_variance={self.compute_semantic_descriptor_variance():.4f})")


if __name__ == "__main__":
    # Example usage with dummy objects
    dummy_objects1 = [
        SemanticMapObject("obj1", {"bbox": {"center": np.array([1.0, 2.0, 3.0]), "size": np.array([0.5, 0.5, 0.5])},
                                   "n_observations": 10, "results": {"chair": 12.32, "table": 456.21}}),
        SemanticMapObject("obj2", {"bbox": {"center": np.array([1.5, 2.5, 3.5]), "size": np.array([0.6, 0.6, 0.6])},
                                   "n_observations": 8, "results": {"tv": 200.27, "laptop": 123.98}})
    ]

    dummy_objects2 = [
        SemanticMapObject("obj3", {"bbox": {"center": np.array([2.0, 3.0, 4.0]), "size": np.array([0.7, 0.7, 0.7])},
                                   "n_observations": 5, "results": {"apple": 13.27, "banana": 789.98}})
    ]

    # Manually setting descriptors
    dummy_objects1[0].semantic_descriptor = np.array([0.79, 0.2, 0.48])
    dummy_objects1[0].geometric_descriptor = np.array([1.0, 2.0, 3.0])
    dummy_objects1[1].semantic_descriptor = np.array([0.456, 0.3, 0.12])
    dummy_objects1[1].geometric_descriptor = np.array([1.5, 2.5, 3.5])

    dummy_objects2[0].semantic_descriptor = np.array([0.3, 0.4, 0.5])
    dummy_objects2[0].geometric_descriptor = np.array([2.0, 3.0, 4.0])

    cluster1 = Cluster(1, dummy_objects1, "Cluster 1")
    cluster2 = Cluster(2, dummy_objects2, "Cluster 2")

    print(cluster1)
    print(cluster2)
    print("Geometric Descriptor Cluster 1:",
          cluster1.compute_center())
    print("Geometric Descriptor Cluster 2:",
          cluster2.compute_center())
    print("Semantic Descriptor Variance Cluster 1:",
          cluster1.compute_semantic_descriptor_variance())
    print("Semantic Descriptor Variance Cluster 2:",
          cluster2.compute_semantic_descriptor_variance())
    print("Semantic similarity between clusters:",
          cluster1.compute_semantic_similarity_to(cluster2))
    print("Geometric distance between clusters:",
          cluster1.compute_geometric_euclidean_distance_to(cluster2))
    print("Clusters overlapping:", cluster1.compute_overlapping_to(cluster2))
