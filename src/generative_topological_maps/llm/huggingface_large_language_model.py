
from transformers import AutoModelForCausalLM, AutoTokenizer

from generative_topological_maps.llm.large_language_model import LargeLanguageModel
from generative_topological_maps.prompt.conversation_history import (
    ConversationHistory,
)


class HuggingfaceLargeLanguageModel(LargeLanguageModel):
    """A wrapper for a Large Language Model from HuggingFace with text generation capabilities."""

    def __init__(self, model_id: str, cache_path: str = None):
        """Initializes the model lazy loading the tokenizer and model."""
        super().__init__(cache_path=cache_path)
        self._model_id = model_id
        self._tokenizer = None
        self._model = None

    def _initialize_model_and_tokenizer(self):
        """Lazy initialization of the tokenizer and model."""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_id, device_map="auto")
        if self._model is None:
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_id, device_map="auto")

    # overrides
    def get_provider_name(self):
        return self._model_id

    # overrides
    def _generate_text(self, conversation_history: ConversationHistory, params: dict = {}):
        """Generates text based on the given prompt and parameters."""
        prompt = conversation_history.get_last_user_message()
        self._initialize_model_and_tokenizer()

        inputs = self._tokenizer(
            prompt, return_tensors="pt").to(self._model.device)
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
        response = self._tokenizer.decode(
            output[0], skip_special_tokens=True)

        return response


if __name__ == "__main__":
    # Example usage
    model = HuggingfaceLargeLanguageModel(
        "Qwen/Qwen2.5-14B")
    conversation = ConversationHistory.create_from_user_message(
        "How are you today??")
    response = model.generate_text(conversation)
    print(response)
