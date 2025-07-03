
import json

from generative_place_categorization.prompt.prompt import Prompt


class ComplexSceneHelperPrompt(Prompt):

    SYSTEM_PROMPT = """
<INSTRUCTION>  
You are provided with a semantic map of a 3D scene. Your goal is to reason about the environment using the information provided.  
You will receive:  
1. A list of objects, each with a semantic label and a 3D bounding box.  
2. A segmentation of the scene into functional areas (called "places"), each described by:  
   - A short **tag** that captures the function of the place (e.g., "kitchen_area").  
   - A **description** explaining what kind of objects are present.  
   - A list of **object IDs** belonging to that place.  
3. A list of **semantic relationships** between pairs of objects, which may include spatial, functional, structural, or other types of relations. Each relation includes the object IDs, their semantic classes, the relation type, and a textual predicate (e.g., "is to the left of", "is supported by", "is used with").

You must use this information to answer user queries about the scene with a short sentence, leveraging both the spatial organization of objects and the higher-level place categories and relations.
Include object ids in your answer if relevant.
</INSTRUCTION>  

<INPUT_FORMAT>  
You will receive a JSON object with the following structure:

- "instances": a list of objects present in the scene. Each object contains:
    - "id": a unique identifier string for the object.
    - "class": the semantic label or category of the object (e.g., "chair", "table").
    - "bbox": a dictionary representing the 3D bounding box of the object, with:
        - "center": a list of three floats indicating the (x, y, z) coordinates of the center.
        - "size": a list of three floats indicating the width, depth, and height.

- "clusters": a dictionary mapping place identifiers to place definitions. Each place includes:
    - "tag": a short string summarizing the function or purpose of the place.
    - "description": a textual description of the types of objects or activities in the place.
    - "objects": a list of object IDs that belong to this place.

- "relationships": a list of semantic relationships between pairs of objects. Each relationship includes:
    - "source_id": the ID of the source object.
    - "source": the semantic class of the source object.
    - "target_id": the ID of the target object.
    - "target": the semantic class of the target object.
    - "type": the category of the relationship (e.g., "spatial", "functional", "structural").
    - "predicate": a textual expression describing the relation (e.g., "is next to", "is used with").
</INPUT_FORMAT>

<INPUT_DATA_SECTION
Consider the following semantic map with places and relationships:
{{semantic_map}}
{{places}}
{{relationships}}

Now answer this question with a short sentence about the semantic map: 
{{question}}
</INPUT_DATA_SECTION>
"""

    def get_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    def global_replace(self, prompt_text: str) -> str:
        return self.replace_prompt_data_dict(self.prompt_data_dict, prompt_text)


if __name__ == "__main__":
    # Example usage
    semantic_map = {
        "obj0": {"bbox": {"center": [1, 2, 0], "size": [0.5, 0.5, 1]}, "class": "chair"},
        "obj1": {"bbox": {"center": [1.5, 2, 0], "size": [0.5, 0.5, 1]}, "class": "chair"},
        "obj2": {"bbox": {"center": [1.25, 2.5, 0], "size": [1, 0.5, 0.1]}, "class": "table"},
    }
    places = {
        "place1": {
            "tag": "kitchen_area",
            "description": "A place with kitchen appliances and utensils.",
            "objects": ["obj0", "obj1"]
        }
    }
    relationships = [
        {
            "source_id": "obj0",
            "source": "chair",
            "target_id": "obj2",
            "target": "table",
            "type": "spatial",
            "predicate": "is next to"
        }
    ]
    question = "What is the main purpose of this area?"

    prompt = ComplexSceneHelperPrompt(
        semantic_map=json.dumps(semantic_map),
        places=json.dumps(places),
        relationships=json.dumps(relationships),
        question=question)
    print(prompt.global_replace(prompt.get_system_prompt()))
