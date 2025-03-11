
from prompt.prompt import Prompt


class PlaceClassifierPrompt(Prompt):

    SYSTEM_PROMPT = """
Classify objects into places based on their class and their location. Objects that are close and are (more or less) of the same class should be grouped in the same place. A place is not necessarily a room but a meaningful area where objects are commonly found together.

Input format:

{
    "instances": {
        "obj0": {
            "bbox": {
                "center": [x, y, z],
                "size": [dx, dy, dz]
            },
            "n_observations": N,
            "class": "object_class"
        },
        "obj1": { ... },
        "obj2": { ... },
        "obj3": { ... },
        ...
    }
}

Output format:

{
    "places": {
        "0": ["obj0", "obj1", "obj2"],
        "1": ["obj5", "obj8"],
        ...
    }
}

Rules:
- Group objects that are together and have a similar functionality in the same place.
- Consider the function and common co-occurrence of objects.
- A place may contain multiple object types (e.g., "table" and "chair" in a dining area).
- Use spatial information first, but take into account semantic meaning too.

Now, classify the following semantic map:
{{semantic_map}}
"""

    def get_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    def global_replace(self, prompt_text: str) -> str:
        prompt_text = self.replace_prompt_data_dict(
            self.prompt_data_dict, prompt_text)
        return prompt_text


if __name__ == "__main__":
    sentence_generator_prompt = PlaceClassifierPrompt(word="hola")
    print(sentence_generator_prompt.get_prompt_text())
