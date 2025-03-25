import os
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer
import json

from utils import file_utils


class LargeLanguageModel():
    """A wrapper for a large language model with text generation capabilities."""

    def __init__(self, model_id: str, tokenizer: PreTrainedTokenizer, model, cache_path: str = None):
        """Initializes the model with a tokenizer and a pre-trained model."""
        self._model_id = model_id
        self._tokenizer = tokenizer
        self._model = model
        # Create cache
        if cache_path is not None and not os.path.exists(cache_path):
            file_utils.save_dict_to_json_file({}, cache_path)
        self._cache_path = cache_path

    def get_model_id(self):
        """Returns the model identifier."""
        return self._model_id

    def generate_text(self, prompt: str, params: dict = {}):
        """Generates text based on the given prompt and parameters."""
        cached_response = self.read_cache_entry(prompt)
        if cached_response is not None:
            # print("Cached response!", cached_response)
            return cached_response

        inputs = self._tokenizer(prompt, return_tensors="pt")
        output = self._model.generate(
            inputs.input_ids,
            max_length=200 if "max_length" not in params else params["max_length"],
            num_return_sequences=1,
            temperature=0.7 if "temperature" not in params else params["temperature"],
            top_k=50 if "top_k" not in params else params["top_k"],
            top_p=0.9 if "top_p" not in params else params["top_p"],
            do_sample=True,
            pad_token_id=self._tokenizer.eos_token_id
        )
        response = self._tokenizer.decode(output[0], skip_special_tokens=True)

        self.write_cache_entry(prompt, response)
        # print("Updated cache!")
        return response

    def generate_text_filtered(self, prompt: str, params: dict = {}):
        """Generates text and removes the input prompt from the response."""
        response = self.generate_text(prompt, params)
        filtered_response = response.replace(prompt, "").strip()
        return filtered_response

    import json

    def generate_json(self, prompt: str, params: dict = {}) -> dict:
        """Generates text and extracts JSON content as a Python dictionary."""
        response = self.generate_text_filtered(prompt, params)
        print("La respuesta fue:\n" + "#"*40)
        print(response)

        # Find the first complete JSON object using brace balancing
        start = None
        brace_count = 0
        for i, char in enumerate(response):
            if char == '{':
                if start is None:
                    start = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start is not None:
                    json_str = response[start:i+1]
                    try:
                        result = json.loads(json_str)
                        return result
                    except json.JSONDecodeError:
                        print("Error al codificar a JSON!")
                        break  # Try no more once a balanced but invalid block was found

        print(f"Error al codificar a JSON!")
        self.clear_cache_entry(prompt)
        return None

    def generate_json_retrying(self, prompt: str, params: dict = {}, retries: int = 3, verbose: bool = False):
        """Generates JSON with retries if the response does not contain valid JSON."""
        for i in range(retries):
            json_response = self.generate_json(prompt, params)
            if verbose:
                print(f"Response {i+1}/{retries}: {json_response}")
            if json_response is not None:
                return json_response
        return None

    def read_cache_entry(self, prompt: str) -> Optional[str]:
        if self._cache_path is not None:
            cache = file_utils.load_json(self._cache_path)
            if prompt in cache:
                return cache[prompt]
            else:
                return None
        else:
            return None

    def write_cache_entry(self, prompt: str, response: str):
        if self._cache_path is not None:
            cache = file_utils.load_json(self._cache_path)
            cache[prompt] = response
            file_utils.save_dict_to_json_file(cache, self._cache_path)

    def clear_cache_entry(self, prompt: str):
        cache = file_utils.load_json(self._cache_path)
        del cache[prompt]
        # print("Cleared cache!")
        file_utils.save_dict_to_json_file(cache, self._cache_path)

    @staticmethod
    def create_from_huggingface(model_id: str, cache_path: Optional[str] = None):
        """Creates a LargeLanguageModel instance with the specified model ID from Huggingface."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, device_map="auto")
            model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map="auto")
            return True, LargeLanguageModel(model_id, tokenizer, model, cache_path)
        except OSError:
            return False, None


prompt = """
I need description of words in JSON format. Follow a strict JSON format. You only speak JSON.

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

Now generate the JSON for the word "sofa".
"""

if __name__ == "__main__":
    _, llm = LargeLanguageModel.create_from_huggingface(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    response = llm.generate_json_retrying(
        prompt, {"max_length": 2000, "temperature": 0.6}, retries=5)
    print(response)
