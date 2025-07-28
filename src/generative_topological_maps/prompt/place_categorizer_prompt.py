

from generative_topological_maps.prompt.prompt import Prompt
from generative_topological_maps.voxeland.semantic_map_object import (
    SemanticMapObject,
)


class PlaceCategorizerPrompt(Prompt):

    SYSTEM_PROMPT = """
<INSTRUCTION>  
You are provided with a set of objects that co-occupy a single location in a 3D scene.  
Each object comes with a semantic class label and a 3D bounding box (center and size).  

Your goal is to:  
1. Assign a **short tag** (1-2 words) that captures the primary **function** or **purpose** of this space.  
2. Write a **functional description** (1-2 sentences) explaining **what activities** or **uses** the area supports, **based on how the objects work together**.  
   - **Do not** merely list the objects—you must infer the space's intended use.  
   - Only use standard room names (e.g., “kitchen,” “bedroom”) if the function is unmistakably clear; otherwise, describe by function.  
</INSTRUCTION>  

<INPUT_FORMAT>  
Objects are provided in JSON, each entry includes:  
- **bbox**: { center: [x, y, z], size: [width, depth, height] }  
- **class**: object semantic label  
</INPUT_FORMAT>  

<OUTPUT_FORMAT>  
Return JSON with these two fields:  
{  
  "tag": "<short_functional_tag>",  
  "description": "<functional_description_of_the_space>"  
}  
</OUTPUT_FORMAT>  

<EXAMPLES>  
Example 1  
Input:  
{  
  "obj0": { "bbox": { "center": [1,2,0], "size": [0.5,0.5,1] }, "class": "chair" },  
  "obj1": { "bbox": { "center": [1.5,2,0], "size": [0.5,0.5,1] }, "class": "chair" },  
  "obj2": { "bbox": { "center": [1.25,2.5,0], "size": [1,0.5,0.1] }, "class": "table" },  
  "obj3": { "bbox": { "center": [1.25,1.5,0], "size": [0.8,0.3,0.05] }, "class": "keyboard" },  
  "obj4": { "bbox": { "center": [1.25,1.2,0], "size": [0.4,0.2,0.1] }, "class": "mouse" },  
  "obj5": { "bbox": { "center": [1.75,1.5,0.5], "size": [0.6,0.4,0.3] }, "class": "monitor" }  
}  
Output:  
{  
  "tag": "workstation",  
  "description": "A computer workstation area set up for digital tasks, with seating around a desk equipped for typing and screen-based work."  
}  

Example 2  
Input:  
{  
  "obj0": { "bbox": { "center": [0,0,0], "size": [2,2,0.5] }, "class": "bed" },  
  "obj1": { "bbox": { "center": [1,0,0], "size": [0.5,0.5,0.5] }, "class": "nightstand" },  
  "obj2": { "bbox": { "center": [1,0.2,0.5], "size": [0.2,0.2,0.2] }, "class": "lamp" },  
  "obj3": { "bbox": { "center": [-0.5,0,-0.2], "size": [1,0.5,1.5] }, "class": "wardrobe" }  
}  
Output:  
{  
  "tag": "rest_area",  
  "description": "A restful sleeping zone designed for rest and personal storage, with a bed for sleeping and nearby surfaces for lighting and clothes organization."  
}
</EXAMPLES>

<TASK>  
Generate the tag and functional description for a location containing the following objects:  
{{objects}}  
</TASK>  
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
