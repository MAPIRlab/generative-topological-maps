

from generative_place_categorization.prompt.prompt import Prompt


class PlaceSegmenterPrompt(Prompt):

    SYSTEM_PROMPT = """
Classify objects into places based on their class and their location. Objects that are close and are (more or less) of the same class should be grouped in the same place. A place is not necessarily a room but a meaningful area where objects are commonly found together.
Objects should be grouped into places based on both their semantics (e.g., "chair" is often found in an "eating area") and spatial proximity (objects close together likely belong to the same place).

<INPUT_FORMAT>
- Objects are represented with bounding boxes (bbox), containing their center position (x, y, z) and size (width, depth, height).
- Class: the most probable class of the object, given by an object detector.
The semantic map comes in JSON:
{
    "instances": {
        "obj0": {
            "bbox": {
                "center": [x, y, z],
                "size": [dx, dy, dz]
            },
            "class": "object_class"
        },
        "obj1": { ... },
        "obj2": { ... },
        "obj3": { ... },
        ...
    }
}
</INPUT_FORMAT>

<OUTPUT_FORMAT>
The output format should be the following:
{
    "places": {
        "eating area": ["obj1", "obj2", "obj3"],
        "bathroom": ["obj32", "obj89"],
        "workspace": ["obj10", "obj15"]
    }
}
}
</OUTPUT_FORMAT>

Now, classify the following semantic map, thinking step by step:
{{semantic_map}}
"""

    def get_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    def global_replace(self, prompt_text: str) -> str:
        prompt_text = self.replace_prompt_data_dict(
            self.prompt_data_dict, prompt_text)
        return prompt_text


if __name__ == "__main__":
    sentence_generator_prompt = PlaceSegmenterPrompt(word="hola")
    print(sentence_generator_prompt.get_prompt_text())
