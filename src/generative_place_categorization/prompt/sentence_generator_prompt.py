
from prompt.prompt import Prompt


class SentenceGeneratorPrompt(Prompt):

    SYSTEM_PROMPT = """Describe what the object is used for in one sentence using the format: 
"Used to [action] [common targets] in the [room or context]." 
Focus on the object's functionality. Use verbs and targets that are shared with similar objects. 
Return the result in the following JSON format:

{
    "word": "<object_name>",
    "description": "Used to <action> <targets> in the <context>.",
    "keywords": ["<verb1>", "<verb2>", "<target1>", "<context>"]
}

Examples:

For "refrigerator":
{
    "word": "refrigerator",
    "description": "Used to store and cool food and drinks in the kitchen.",
    "keywords": ["store", "cool", "food", "kitchen"]
}

For "toilet"
{
    "word": "toilet",
    "description": "Used to dispose of human waste and maintain hygiene in the bathroom.",
    "keywords": ["dispose", "hygiene", "waste", "bathroom"]
}

For "couch"
{
    "word": "couch",
    "description": "Used to sit, rest, and socialize in the living room.",
    "keywords": ["sit", "rest", "socialize", "living room"]
}

For "chair":
{
    "word": "couch",
    "description": "Used to sit and support the body while performing tasks in the living room.",
    "keywords": ["sit", "relax", "furniture", "living room"]
}

Now, generate the JSON for the object "{{word}}".
"""

    def get_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    def global_replace(self, prompt_text: str) -> str:
        prompt_text = self.replace_prompt_data_dict(
            self.prompt_data_dict, prompt_text)
        return prompt_text