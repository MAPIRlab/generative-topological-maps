import json

from generative_place_categorization.prompt.prompt import Prompt


class SimpleSceneHelperPrompt(Prompt):

    SYSTEM_PROMPT = """
<INSTRUCTION>
You are an LLM assistant specialized in understanding 3D semantic maps. 
Your task is to understnad a semantic map and answer questions about it with a short sentence.
Include object ids in your answer if relevant.
</INSTRUCTION>

<INPUT_FORMAT>  
Objects are provided in JSON, each entry includes:  
- **bbox**: { center: [x, y, z], size: [width, depth, height] }  
- **class**: object semantic label  
</INPUT_FORMAT>  

<INPUT_DATA_SECTION
Consider the following semantic map:
{{semantic_map}}

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
    question = "What is the main purpose of this area?"

    prompt = SimpleSceneHelperPrompt(
        semantic_map=json.dumps(semantic_map), question=question)
    print(prompt.global_replace(prompt.get_system_prompt()))
