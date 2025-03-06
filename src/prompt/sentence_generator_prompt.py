
from prompt.prompt import Prompt


class SentenceGeneratorPrompt(Prompt):

    SYSTEM_PROMPT = """
Describe objects in JSON format. Keep sentences short. Mention where the object is usually found 
and what you can do with it. Include keywords related to the activity associated with the object.

Examples:

For "refrigerator":
{
    "word": "refrigerator",
    "description": "Found in the kitchen. Used to store food and keep it cold. Helps preserve ingredients and drinks.",
    "keywords": ["store", "cool", "preserve", "food", "drinks"]
}

For "toilet":
{
    "word": "toilet",
    "description": "Located in the bathroom. Used for sanitation. Essential for hygiene and waste disposal.",
    "keywords": ["sanitation", "hygiene", "waste", "clean"]
}

For "sofa":
{
    "word": "couch",
    "description": "Placed in the living room. Used for sitting, relaxing, and watching TV. Can also be used for napping.",
    "keywords": ["sit", "relax", "watch", "nap", "comfort"]
}

Now, generate the JSON for the object "{{word}}".
"""

    def get_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    def global_replace(self, prompt_text: str) -> str:
        prompt_text = self.replace_prompt_data_dict(
            self.prompt_data_dict, prompt_text)
        return prompt_text


if __name__ == "__main__":
    sentence_generator_prompt = SentenceGeneratorPrompt(word="hola")
    print(sentence_generator_prompt.get_prompt_text())
