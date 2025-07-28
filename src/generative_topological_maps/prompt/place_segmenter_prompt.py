import json

from generative_topological_maps.prompt.prompt import Prompt


class PlaceSegmenterPrompt(Prompt):

    SYSTEM_PROMPT = """
<INSTRUCTION>
You are given a set of detected objects, each with a semantic class and a 3D bounding box. Your goal is to group these objects into places—meaningful clusters that share a common functionality and are spatially proximate. A place might be “seating_area,” “workstation,” or “storage_zone,” not necessarily a single room.
When grouping:
- Semantic similarity: Objects serving similar functionality (e.g., chair + couch → seating area) should be in the same place.
- Spatial proximity: Objects that are near each other in 3D space and functionally related reinforce the same grouping.
- Completeness and exclusivity: Every object in the semantic_map must be assigned to exactly one place; no object may appear in more than one place.
</INSTRUCTION>

<GUIDELINES>
You may perform internal, step-by-step reasoning, but your final response must contain only the JSON object in the required structure. Do not include any extra text before or after the JSON.
</GUIDELINES>

<INPUT_FORMAT>
You will receive a JSON object named semantic_map with this structure:
{
  "instances": {
    "obj0": {
      "bbox": { "center": [x, y, z], "size": [dx, dy, dz] },
      "class": "object_class"
    },
    "obj1": { … },
    …
  }
}
- bbox.center: [x, y, z] position in meters  
- bbox.size: [width, depth, height]  
- class: semantic label (e.g., "chair", "table")
</INPUT_FORMAT>

<OUTPUT_FORMAT>
Return exactly this JSON structure, listing each place with a short tag, a human-readable description, and the list of object IDs it contains. Do not add comments or extra keys.

{
  "places": [
    {
      "name": "<place_tag>",
      "description": "<brief description of this place>",
      "objects": ["obj1", "obj2", "obj3"]
    },
    {
      "name": "<another_tag>",
      "description": "<brief description>",
      "objects": ["obj4", "obj5"]
    }
    // … more places …
  ]
}

- name: short, unique tag for the place (e.g., "seating_area", "kitchen_zone")
- description: one or two phrases describing the place and its function
- objects: list of object IDs in that place
</OUTPUT_FORMAT>

<TASK> 
Classify the following semantic_map into places using the rules above. Only output the final JSON in the specified format.

{{semantic_map}}
</TASK>
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
    dummy_map_str = json.dumps(dummy_map)
    prompt = PlaceSegmenterPrompt(semantic_map=dummy_map_str)
    print(prompt.get_prompt_text())
