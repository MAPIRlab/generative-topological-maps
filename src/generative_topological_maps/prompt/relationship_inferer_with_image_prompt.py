from generative_topological_maps.prompt.prompt import Prompt


class RelationshipInfererWithImagePrompt(Prompt):

    SYSTEM_PROMPT = """
<MAIN_INSTRUCTION>
Given two objects in a semantic map, identified by both name and unique object ID, infer all possible directed relationships in both directions:
1) First using object1 as the source and object2 as the target.
2) Then using object2 as the source and object1 as the target.
</MAIN_INSTRUCTION>

<INFORMATION_SOURCES>
Use all available information, including:
- Geometric properties (centroid position, size, relative position, overlap, distance).
- Semantic labels (class names, known functional roles).
- Visual information from provided images of the scene where the objects appear.
</INFORMATION_SOURCES>

<RELATIONSHIP_TYPES>
Consider subtle relationships, providing an open vocabulary description (predicate):
- Spatial: positional arrangements (e.g., "is on", "is below", "is above", "is adjacent to", "is next to", "is connected to", "is aligned with", "is inside", "is outside", "is part of a group with").
- Structural: assembly or support connections (e.g., "is part of", "is mounted on", "is supported by", "is attached to", "is composed of", "contains", "holds", "forms part of").
- Functional: usage or role connections (e.g., "is used with", "is required for", "enables", "interacts with", "powers", "provides", "is for", "can be operated by").
- Causal: cause-effect interactions (e.g., "activates", "turns on", "turns off", "triggers", "initiates", "stops", "changes state of", "produces").
</RELATIONSHIP_TYPES>

<NO_RELATION_RULE>
If no relationships are observed, return an empty array: [].
</NO_RELATION_RULE>

<OUTPUT_FORMAT_DETAILS>
For every relation detected, include:
- source_id: the source object ID
- source: the source object name
- target_id: the target object ID
- target: the target object name
- type: one of "spatial", "structural", "functional", "causal"
- predicate: an open vocabulary description of the relation
</OUTPUT_FORMAT_DETAILS>

<OUTPUT_INSTRUCTION>
Return a JSON array of relation objects. Include as many entries as apply.
</OUTPUT_OUTPUT_INSTRUCTION>

<EXAMPLES>
Example 1: Objects on a Desk (Spatial & Structural)

Input:
- object1_id: "obj1"
- object1_name: "laptop"
- centroid1: [0.5, 0.7, 1.2]
- size1: [0.3, 0.2, 0.03]
- object2_id: "obj2"
- object2_name: "desk"
- centroid2: [0.5, 0.5, 1.0]
- size2: [1.0, 0.6, 0.1]
- image_context: (Imagine an image showing a laptop open on a desk)

Expected Output:
[
  {
    "source_id": "obj1",
    "source": "laptop",
    "target_id": "obj2",
    "target": "desk",
    "type": "spatial",
    "predicate": "is on"
  },
  {
    "source_id": "obj2",
    "source": "desk",
    "target_id": "obj1",
    "target": "laptop",
    "type": "structural",
    "predicate": "supports"
  }
]

Example 2: Light Switch and Lamp (Causal)

Input:
- object1_id: "obj3"
- object1_name: "light switch"
- centroid1: [1.0, 0.8, 1.5]
- size1: [0.05, 0.05, 0.02]
- object2_id: "obj4"
- object2_name: "lamp"
- centroid2: [2.5, 1.0, 1.8]
- size2: [0.3, 0.3, 0.5]
- image_context: (Imagine an image showing a light switch on a wall and a lamp nearby)

Expected Output:
[
  {
    "source_id": "obj3",
    "source": "light switch",
    "target_id": "obj4",
    "target": "lamp",
    "type": "causal",
    "predicate": "turns on"
  },
  {
    "source_id": "obj4",
    "source": "lamp",
    "target_id": "obj3",
    "target": "light switch",
    "type": "causal",
    "predicate": "is controlled by"
  }
]

Example 3: Fork and Knife (Functional)

Input:
- object1_id: "obj5"
- object1_name: "fork"
- centroid1: [0.1, 0.2, 0.3]
- size1: [0.02, 0.2, 0.01]
- object2_id: "obj6"
- object2_name: "knife"
- centroid2: [0.15, 0.2, 0.3]
- size2: [0.02, 0.2, 0.01]
- image_context: (Imagine an image showing a fork and knife placed together on a dining table)

Expected Output:
[
  {
    "source_id": "obj5",
    "source": "fork",
    "target_id": "obj6",
    "target": "knife",
    "type": "functional",
    "predicate": "is used with"
  },
  {
    "source_id": "obj6",
    "source": "knife",
    "target_id": "obj5",
    "target": "fork",
    "type": "functional",
    "predicate": "is used with"
  }
]

Example 4: Two Distant, Unrelated Objects (No Relationships)

Input:
- object1_id: "obj7"
- object1_name: "book"
- centroid1: [10.0, 5.0, 2.0]
- size1: [0.2, 0.15, 0.05]
- object2_id: "obj8"
- object2_name: "mountain"
- centroid2: [100.0, 200.0, 50.0]
- size2: [5000.0, 5000.0, 2000.0]
- image_context: (Imagine two entirely separate images, one of a book, another of a mountain range)

Expected Output:
[]
</EXAMPLES>

<INPUT_DATA_SECTION>
Now infer relationships for the following objects, also considering any visual information from the scene:
- object1_id: "{{object1_id}}"
- object1_name: "{{object1_name}}"
- centroid1: {{object1_centroid}}
- size1: {{object1_size}}
- object2_id: "{{object2_id}}"
- object2_name: "{{object2_name}}"
- centroid2: {{object2_centroid}}
- object2_size: {{object2_size}}
- image_context: (This is where the actual image data or a description of its content will be provided if available)
</INPUT_DATA_SECTION>

<FINAL_INFERENCE_INSTRUCTION>
Think carefully about positional context, size compatibility, class semantics, and **visual cues from the image** to infer all possible relations.
Return a well-structured JSON array of the detected relationships, or [] if none apply.
</FINAL_INFERENCE_INSTRUCTION>
"""

    def get_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    def global_replace(self, prompt_text: str) -> str:
        return self.replace_prompt_data_dict(self.prompt_data_dict, prompt_text)
