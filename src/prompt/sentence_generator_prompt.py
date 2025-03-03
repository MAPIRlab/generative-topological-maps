
from prompt import Prompt


class SentenceGeneratorPrompt(Prompt):

    SYSTEM_PROMPT = """
I need description of words in JSON format. Follow a strict JSON format. You can think, but your final answer should be JSON.

Examples               
For "refrigerator":
{
    "word": "refrigerator",
    "description": "A refrigerator is an appliance used for keeping food fresh, typically found in the kitchen"
},

For "toilet":
{
    "word": "toilet",
    "description": "A toilet is used for personal hygiene and is usually found in the bathroom."
}

Now generate the JSON for the word "{{word}}".
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
