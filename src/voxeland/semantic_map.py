from typing import List, Optional
from voxeland.semantic_map_object import SemanticMapObject


class SemanticMap:
    """
    Represents a semantic map containing multiple SemanticMapObjects.
    """

    def __init__(self, objects: List[SemanticMapObject]):
        self.objects: List[SemanticMapObject] = objects

    def find_object(self, object_id: str) -> Optional[SemanticMapObject]:
        """Finds and returns an object by its ID."""
        for obj in self.objects:
            if obj.get_object_id() == object_id:
                return obj
        return None

    def get_objects(self, unknown: bool = False) -> List[SemanticMapObject]:
        """Returns the list of SemanticMapObjects."""
        if not unknown:
            return list(filter(lambda smo: smo.get_most_probable_class() != "unknown", self.objects))
        else:
            return self.objects


if __name__ == "__main__":
    # Example semantic map data
    semantic_map_data = {
        "instances": {
            "obj1": {
                "bbox": {
                    "center": [4.748, 4.526, 1.436],
                    "size": [9.388, 9.139, 3.037]
                },
                "n_observations": 1,
                "results": {
                    "chair": 123.31,
                    "bed": 487.2,
                    "tv": 0.98,
                    "unknown": 35.4
                }
            },
            "obj2": {
                "bbox": {
                    "center": [2.5, 3.6, 1.2],
                    "size": [5.0, 4.2, 2.5]
                },
                "n_observations": 3,
                "results": {
                    "table": 200.5,
                    "sofa": 150.8
                }
            }
        }
    }

    processed_objects = [SemanticMapObject(
        obj_id, obj_data) for obj_id, obj_data in semantic_map_data["instances"].items()]
    semantic_map = SemanticMap(processed_objects)

    # Testing find_object
    obj = semantic_map.find_object("obj1")
    if obj:
        print("Found Object:", obj.get_object_id())
    else:
        print("Object not found.")

    # Testing get_objects
    all_objects = semantic_map.get_objects()
    print("Total objects in the map:", len(all_objects))
