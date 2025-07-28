import json
import os
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from generative_topological_maps.utils import file_utils
from generative_topological_maps.voxeland.semantic_map_object import (
    SemanticMapObject,
)


class SemanticMap:
    """
    Represents a semantic map containing multiple SemanticMapObjects.

    Can be constructed directly with ID, object list, and an optional path:
    - colors_path: directory of color images.

    Or loaded from a JSON file via `from_json_path`.
    """

    EXCLUDED_CLASSES = ["unknown", "wall", "floor", "ceiling", "door"]

    def __init__(
        self,
        semantic_map_id: str,
        objects: List[SemanticMapObject],
        colors_path: Optional[str] = None
    ):
        self.semantic_map_id = semantic_map_id
        self._objects = objects
        self.colors_path = colors_path

    @staticmethod
    def from_json_path(
        json_path: str,
        colors_path: Optional[str] = None,
    ) -> "SemanticMap":
        """Load a SemanticMap from a JSON file."""
        map_id = os.path.splitext(os.path.basename(json_path))[0]
        data = file_utils.load_json(json_path)
        objects: List[SemanticMapObject] = []
        for obj_id, obj_data in data.get("instances", {}).items():
            smo = SemanticMapObject(obj_id, obj_data)
            objects.append(smo)
        return SemanticMap(map_id, objects, colors_path)

    def find_object(self, object_id: str) -> Optional[SemanticMapObject]:
        for obj in self._objects:
            if obj.object_id == object_id:
                return obj
        return None

    def get_all_objects(
        self,
        include_all_classes: bool = False
    ) -> List[SemanticMapObject]:
        if include_all_classes:
            return list(self._objects)
        return [obj for obj in self._objects
                if obj.get_most_probable_class() not in self.EXCLUDED_CLASSES]

    def get_close_object_pairs(
        self,
        threshold_distance: float,
        include_all_classes: bool = False
    ) -> List[Tuple[SemanticMapObject, SemanticMapObject]]:
        """
        Returns a list of object pairs whose bounding boxes are closer
        than or equal to the given threshold_distance.

        The distance is computed as the minimal separation between
        their axis-aligned bounding boxes (AABB).
        """
        objects = self.get_all_objects(include_all_classes)
        pairs: List[Tuple[SemanticMapObject, SemanticMapObject]] = []
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                obj1 = objects[i]
                obj2 = objects[j]

                # Get centers and sizes
                c1 = np.asarray(obj1.bbox_center)  # [x, y, z]
                s1 = np.asarray(obj1.bbox_size)    # [sx, sy, sz]
                c2 = np.asarray(obj2.bbox_center)  # [x, y, z]
                s2 = np.asarray(obj2.bbox_size)    # [sx, sy, sz]

                # Distance along each axis:
                d = np.abs(c1 - c2) - (s1 / 2.0) - (s2 / 2.0)

                # Distance for each axis is clipped to be at least 0
                d = np.maximum(d, 0.0)

                # Final distance is the Euclidean distance
                dist = np.linalg.norm(d)

                if dist <= threshold_distance:
                    pairs.append((obj1, obj2))
        return pairs

    def get_common_frames_by_pixel_count(
        self,
        object_id1: str,
        object_id2: str
    ) -> List[int]:
        obj1 = self.find_object(object_id1)
        obj2 = self.find_object(object_id2)
        if not obj1 or not obj2:
            return []
        frames1 = obj1.frames
        frames2 = obj2.frames
        common = set(frames1) & set(frames2)
        return sorted(common,
                      key=lambda f: frames1[f] + frames2[f],
                      reverse=True)

    def get_frame_image(self, frame_id: int) -> Image.Image:
        if not self.colors_path:
            raise RuntimeError(
                "colors_path is not defined for this SemanticMap")
        return file_utils.open_image(self.colors_path, str(frame_id))

    def get_prompt_json_representation(self) -> str:
        repr_dict = {"instances": {}}
        for obj in self.get_all_objects():
            obj_entry = {
                "bbox": {"center": [round(c, 2) for c in obj.bbox_center],
                         "size": [round(s, 2) for s in obj.bbox_size]},
                "class": obj.get_most_probable_class()
            }
            repr_dict["instances"][obj.object_id] = obj_entry
        return json.dumps(repr_dict)

    def __repr__(self) -> str:
        return (
            f"SemanticMap(id={self.semantic_map_id!r}, objects={len(self._objects)}, "
            f"colors_path={self.colors_path!r})"
        )
