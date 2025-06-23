from generative_place_categorization.prompt.prompt import Prompt


class PlaceSegmenterPrompt(Prompt):

    SYSTEM_PROMPT = """
<INSTRUCTION>
Classify objects into places based on their class and their location. Objects that are close and are (more or less) of the same class should be grouped in the same place. A place is not necessarily a room but a meaningful area where objects are commonly found together.
Objects should be grouped into places based on both their semantics (e.g., "chair" is often found in an "eating area") and spatial proximity (objects close together likely belong to the same place).
</INSTRUCTION>

<GUIDELINES>
You may include your internal reasoning or explanations, but you **must** end your response with **only** the JSON object specified in `<OUTPUT_FORMAT>`.  
Do not wrap the JSON in markdown or code fences, and do not output any text after the JSON.
</GUIDELINES>

<INPUT_FORMAT>
- Objects are represented with bounding boxes (bbox), containing their center position (x, y, z) and size (width, depth, height).
- Class: the most probable class of the object, given by an object detector.
The semantic map comes in JSON:
```json
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
```
</INPUT_FORMAT>

<OUTPUT_FORMAT>
Return exactly this structure as valid JSON, listing each place with a short tag, a human-readable description, and the list of object IDs it contains:

```json
{
    "places": [
        {
            "name": "<place_tag>",
            "description": "<brief description of this place>",
            "objects": ["obj1", "obj2", "obj3"]
        },
        {
            "name": "<another_tag>",
            "description": "<description>",
            "objects": ["obj4", "obj5"]
        }
        // … additional places …
    ]
}
```
</OUTPUT_FORMAT>

Now classify the following semantic map step by step, but **only output** the final JSON matching `<OUTPUT_FORMAT>`:
{{semantic_map}}
"""

    def get_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    def global_replace(self, prompt_text: str) -> str:
        return self.replace_prompt_data_dict(self.prompt_data_dict, prompt_text)


if __name__ == "__main__":
    # Ejemplo de uso
    dummy_map = {
        "instances": {
            "obj1": {"bbox": {"center": [0, 0, 0], "size": [1, 1, 1]}, "class": "chair"},
            "obj2": {"bbox": {"center": [0, 1, 0], "size": [1, 1, 1]}, "class": "table"},
            "obj3": {"bbox": {"center": [5, 5, 0], "size": [1, 1, 1]}, "class": "sink"},
            "obj4": {"bbox": {"center": [5, 6, 0], "size": [1, 1, 1]}, "class": "toilet"}
        }
    }
    prompt = PlaceSegmenterPrompt(semantic_map=dummy_map)
    print(prompt.get_prompt_text())
