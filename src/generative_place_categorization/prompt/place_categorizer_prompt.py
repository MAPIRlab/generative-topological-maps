from prompt.prompt import Prompt
from voxeland.semantic_map_object import SemanticMapObject


class PlaceCategorizerPrompt(Prompt):

    SYSTEM_PROMPT = """You are provided with a collection of objects that belong to a single location within a 3D environment. 
Each object is characterized by its semantic class (label) and a 3D bounding box (center and size). 
Your task is to:

1. Assign a **short tag** (1-2 words) that summarizes the primary function or identity of this location.
2. Provide a **natural language description** of the location, based on the objects it contains and their likely usage together. The description should be concise, limited to 1-2 sentences.

Use room-specific terms (e.g., "kitchen", "bedroom") only if the location is clearly identifiable as such; otherwise, describe the location based on its objects and their arrangement.

<INPUT_FORMAT>
Objects are provided in a JSON format, where each object includes:
- A bounding box (bbox), with center [x, y, z] and size [width, depth, height].
- A class: the predicted object category.

Example:
{
    "obj0": {
        "bbox": {
            "center": [x, y, z],
            "size": [dx, dy, dz]
        },
        "class": "object_class"
    },
    "obj1": { ... },
    ...
}
</INPUT_FORMAT>

<OUTPUT_FORMAT>
Return a JSON of the form:
{
    "tag": "<short_name_for_this_location>",
    "description": "<description_of_the_location_based_on_its_objects>"
}

Now, generate the tag and description for a location containing the following objects:
{{objects}}
"""

    def __init__(self, objects: list[SemanticMapObject] = None, **prompt_data_dict: dict):
        super().__init__(**prompt_data_dict)
        self.objects = objects

    def get_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    def replace_objects(self, objects: list[SemanticMapObject], prompt_text: str) -> str:
        objects_text = ""
        for obj in objects:
            objects_text += f"""
            \"{obj.object_id}\": {{
                \"bbox\": {{
                    \"center\": [{round(obj.bbox_center[0], 2)}, {round(obj.bbox_center[1], 2)}, {round(obj.bbox_center[2], 2)}],
                    \"size\": [{round(obj.bbox_size[0], 2)}, {round(obj.bbox_size[1], 2)}, {round(obj.bbox_size[2], 2)}]
                }},
                \"class\": \"{obj.get_most_probable_class()}\"
            }},"""
        objects_text = objects_text[:-1]
        prompt_text = prompt_text.replace("{{objects}}", objects_text)
        return prompt_text

    def global_replace(self, prompt_text: str) -> str:
        prompt_text = self.replace_prompt_data_dict(
            self.prompt_data_dict, prompt_text
        )
        prompt_text = self.replace_objects(self.objects, prompt_text)
        return prompt_text
