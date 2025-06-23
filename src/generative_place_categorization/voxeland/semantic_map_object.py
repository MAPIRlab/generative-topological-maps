import math
from typing import Dict, List, Tuple


class SemanticMapObject:
    def __init__(self, object_id: str, data: Dict):
        self.object_id = object_id
        if data is not None:
            self.bbox_center = data.get("bbox", {}).get("center")
            self.bbox_size = data.get("bbox", {}).get("size")
            self.n_observations = data.get("n_observations")
            self.results = data.get("results", {})
            # frames: mapping frame_id -> point count
            self.frames: Dict[int, int] = data.get("frames", {}) or {}
            # Descriptors
            self.geometric_descriptor = self.bbox_center
            self.semantic_descriptor = None
            self.global_descriptor = None
        else:
            self.bbox_center = None
            self.bbox_size = None
            self.n_observations = None
            self.results = {}
            self.frames = {}
            # Descriptors
            self.geometric_descriptor = None
            self.semantic_descriptor = None
            self.global_descriptor = None

    def get_classes(self) -> List[str]:
        """Returns the ordered list of classes by relevance."""
        return sorted(self.results.keys(), key=lambda k: self.results[k], reverse=True)

    def get_most_probable_class(self) -> str:
        """Returns the class with the highest probability."""
        if not self.results:
            return "unknown"
        return max(self.results, key=self.results.get)

    def distance_to(self, other: 'SemanticMapObject') -> float:
        """
        Compute the Euclidean distance between the bbox centers of this object and another.

        Raises:
            ValueError: if either bbox_center is not defined or has mismatched dimensions.
        """
        if self.bbox_center is None or other.bbox_center is None:
            raise ValueError("bbox_center not defined for one of the objects")
        if len(self.bbox_center) != len(other.bbox_center):
            raise ValueError("bbox_center dimensions do not match")
        return math.sqrt(
            sum((c1 - c2) ** 2 for c1, c2 in zip(self.bbox_center, other.bbox_center))
        )

    def __repr__(self) -> str:
        """String representation showing up to first 5 frame entries."""
        # Take first 5 frames sorted by frame id
        frame_items: List[Tuple[int, int]] = sorted(self.frames.items())
        display_items = frame_items[:5]
        more = len(frame_items) > 5
        frames_str = ", ".join(
            f"{fid}:{count}" for fid, count in display_items)
        if more:
            frames_str += ", ..."
        return (
            f"SemanticMapObject(object_id={self.object_id!r}, "
            f"bbox_center={self.bbox_center}, "
            f"bbox_size={self.bbox_size}, "
            f"n_observations={self.n_observations}, "
            f"results={self.results}, "
            f"frames={{ {frames_str} }}, "
            f"geometric_descriptor={self.geometric_descriptor}, "
            f"semantic_descriptor={self.semantic_descriptor}, "
            f"global_descriptor={self.global_descriptor})"
        )
