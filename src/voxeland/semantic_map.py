import json
from typing import List, Optional
from voxeland.semantic_map_object import SemanticMapObject


class SemanticMap:
    """
    Represents a semantic map containing multiple SemanticMapObjects.
    """

    # Excluded not relevant classes
    EXCLUDED_CLASSES = ["unknown", "wall", "floor", "ceiling", "door"]

    def __init__(self, semantic_map_id: str, objects: List[SemanticMapObject]):
        """Initializes a semantic map with a list of sematic map objects."""
        self.semantic_map_id = semantic_map_id
        self._objects: List[SemanticMapObject] = objects

    def find_object(self, object_id: str) -> Optional[SemanticMapObject]:
        """Finds and returns an object by its ID, None if not found."""
        for obj in self._objects:
            if obj.object_id == object_id:
                return obj
        return None

    def get_all_objects(self, include_all_classes: bool = False) -> List[SemanticMapObject]:
        """Returns the list of semantic map objects. If not include_all_classes, it filters out certain object classes."""
        if not include_all_classes:
            return list(filter(lambda smo: smo.get_most_probable_class() not in self.EXCLUDED_CLASSES, self._objects))
        else:
            return self._objects

    def get_json_representation(self) -> str:
        """Gets a JSON representation of the semantic map."""
        repr_dict = {"instances": {}}
        for object in self.get_all_objects():
            repr_dict["instances"][object.object_id] = {
                "bbox": {
                    "center": [round(coord, 2) for coord in object.bbox_center],
                    "size": [round(coord, 2) for coord in object.bbox_size]
                },
                "class": object.get_most_probable_class()
            }
        return json.dumps(repr_dict)
