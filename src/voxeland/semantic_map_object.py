from typing import List, Dict
from nltk.corpus import wordnet


class SemanticMapObject:

    def __init__(self, object_id: str, data: Dict):
        self.object_id = object_id
        if data is not None:
            self.bbox_center = data["bbox"]["center"]
            self.bbox_size = data["bbox"]["size"]
            self.n_observations = data["n_observations"]
            self.results = data.get("results", {})
            self.geometric_descriptor = None
            self.semantic_descriptor = None
            self.global_descriptor = None
        else:
            self.object_id = object_id
            self.bbox_center = None
            self.bbox_size = None
            self.n_observations = None
            self.results = None
            self.geometric_descriptor = None
            self.semantic_descriptor = None
            self.global_descriptor = None

    def get_object_id(self) -> str:
        """Returns the object ID."""
        return self.object_id

    def get_bbox_center(self, decimals: int = 2) -> List[float]:
        """Returns the bounding box center coordinates rounded to the given number of decimals."""
        return [round(coord, decimals) for coord in self.bbox_center]

    def get_bbox_size(self, decimals: int = 2) -> List[float]:
        """Returns the bounding box size rounded to the given number of decimals."""
        return [round(size, decimals) for size in self.bbox_size]

    def get_n_observations(self) -> int:
        """Returns the number of observations."""
        return self.n_observations

    def get_classes(self) -> List[str]:
        """Returns the ordered list of classes by relevance."""
        return sorted(self.results.keys(), key=lambda k: self.results[k], reverse=True)

    def get_most_probable_class(self) -> str:
        """Returns the class with the highest probability."""
        if not self.results:
            return "unknown"
        return max(self.results, key=self.results.get)

    def get_most_probable_class_synonyms(self):
        # TODO
        synonyms = set()
        for syn in wordnet.synsets(self.get_most_probable_class()):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
        return list(synonyms)


if __name__ == "__main__":
    # Example usage
    data = {
        "obj1": {
            "bbox": {
                "center": [4.748012542724609, 4.5259833335876465, 1.436192274093628],
                "size": [9.387859344482422, 9.139277458190918, 3.036680221557617]
            },
            "n_observations": 1,
            "results": {
                "chair": 123.31313,
                "bed": 487.2,
                "tv": 0.98,
                "unknown": 35.40000000000013
            }
        }
    }

    for obj_id, obj_data in data.items():
        obj = SemanticMapObject(obj_id, obj_data)
        print("Object ID:", obj.get_object_id())
        print("BBox Center:", obj.get_bbox_center())  # [4.75, 4.53, 1.44]
        print("BBox Size:", obj.get_bbox_size())  # [9.39, 9.14, 3.04]
        print("Number of Observations:", obj.get_n_observations())  # 1
        # ['bed', 'chair', 'unknown', 'tv']
        print("Ordered Classes:", obj.get_classes())
        print("Most Probable Class:", obj.get_most_probable_class())  # 'bed'
