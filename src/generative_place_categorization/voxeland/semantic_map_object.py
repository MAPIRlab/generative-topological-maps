from typing import List, Dict
from nltk.corpus import wordnet


class SemanticMapObject:

    def __init__(self, object_id: str, data: Dict):
        self.object_id = object_id
        if data is not None:
            self.bbox_center = data["bbox"]["center"]
            self.bbox_size = data["bbox"]["size"]
            self.n_observations = data.get("n_observations", None)
            self.results = data.get("results", {})
            # Descriptors
            self.geometric_descriptor = self.bbox_center
            self.semantic_descriptor = None
            self.global_descriptor = None
        else:
            self.object_id = object_id
            self.bbox_center = None
            self.bbox_size = None
            self.n_observations = None
            self.results = None
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

    def get_most_probable_class_synonyms(self):
        """Returns a list of synonyms for the most probable class."""
        synonyms = set()
        for syn in wordnet.synsets(self.get_most_probable_class()):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
        return list(synonyms)

    def __repr__(self) -> str:
        """Returns a string representation of the SemanticMapObject."""
        return (f"SemanticMapObject(object_id={self.object_id!r}, "
                f"bbox_center={self.bbox_center}, "
                f"bbox_size={self.bbox_size}, "
                f"n_observations={self.n_observations}, "
                f"results={self.results}, "
                f"geometric_descriptor={self.geometric_descriptor}, "
                f"semantic_descriptor={self.semantic_descriptor}, "
                f"global_descriptor={self.global_descriptor})")
