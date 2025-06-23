import hashlib
import io
import json
import os
from typing import Any, Dict, List, Optional

from PIL import Image
from transformers import AutoFeatureExtractor, AutoTokenizer, VisionEncoderDecoderModel

from generative_place_categorization.utils import file_utils


class _JsonCache:
    """Simple JSON file-based cache manager."""

    def __init__(self, path: str):
        self.path = path
        if not os.path.exists(path):
            file_utils.save_dict_to_json_file({}, path)

    def get(self, key: str) -> Optional[str]:
        cache = file_utils.load_json(self.path)
        return cache.get(key)

    def set(self, key: str, value: str) -> None:
        cache = file_utils.load_json(self.path)
        cache[key] = value
        file_utils.save_dict_to_json_file(cache, self.path)

    def clear(self, key: str) -> None:
        cache = file_utils.load_json(self.path)
        cache.pop(key, None)
        file_utils.save_dict_to_json_file(cache, self.path)


class LargeVisionLanguageModel:
    """
    Wrapper for a vision-language model; supports text and JSON generation with optional caching.
    """

    def __init__(
        self,
        model_id: str,
        cache_path: Optional[str] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        processor: Optional[AutoFeatureExtractor] = None,
        model: Optional[VisionEncoderDecoderModel] = None
    ):
        self.model_id = model_id
        self._tokenizer = tokenizer
        self._processor = processor
        self._model = model
        self.cache = _JsonCache(cache_path) if cache_path else None

    def _ensure_model_loaded(self) -> None:
        if not self._processor:
            self._processor = AutoFeatureExtractor.from_pretrained(
                self.model_id)
        if not self._tokenizer:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if not self._model:
            self._model = VisionEncoderDecoderModel.from_pretrained(
                self.model_id)

    def _make_cache_key(self, prompt: str, images: List[Image.Image]) -> str:
        hasher = hashlib.sha256()
        hasher.update(prompt.encode('utf-8'))
        for img in images:
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            hasher.update(buf.getvalue())
        return hasher.hexdigest()

    def generate(
        self,
        prompt: str,
        images: List[Image.Image],
        max_length: int = 200,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> str:
        """Generate or retrieve a cached text response based on images and prompt."""
        key = prompt + '|' + self._make_cache_key(prompt, images)
        if self.cache:
            cached = self.cache.get(key)
            if cached:
                return cached

        self._ensure_model_loaded()
        pixel_values = self._processor(
            images=images, return_tensors='pt').pixel_values
        input_ids = self._tokenizer(prompt, return_tensors='pt').input_ids
        outputs = self._model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

        if self.cache:
            self.cache.set(key, text)
        return text

    def generate_json(
        self,
        prompt: str,
        images: List[Image.Image],
        **kwargs: Any
    ) -> Optional[Dict]:
        """Generate text and parse first JSON object found in it."""
        raw = self.generate(prompt, images, **kwargs)
        raw_trim = raw[len(prompt):].strip() if raw.startswith(prompt) else raw
        brace = 0
        start = None
        for i, ch in enumerate(raw_trim):
            if ch == '{':
                if start is None:
                    start = i
                brace += 1
            elif ch == '}':
                brace -= 1
                if brace == 0 and start is not None:
                    snippet = raw_trim[start:i+1]
                    try:
                        return json.loads(snippet)
                    except json.JSONDecodeError:
                        break
        if self.cache:
            key = prompt + '|' + self._make_cache_key(prompt, images)
            self.cache.clear(key)
        return None

    def generate_json_retrying(
        self,
        prompt: str,
        images: List[Image.Image],
        retries: int = 3,
        **kwargs: Any
    ) -> Optional[Dict]:
        """Retry JSON generation up to `retries` times."""
        for _ in range(retries):
            result = self.generate_json(prompt, images, **kwargs)
            if result is not None:
                return result
        return None


if __name__ == "__main__":
    from PIL import Image

    import generative_place_categorization.constants as constants

    # Using Google's official VisionEncoderDecoderModel
    model_id = "google/vit-gpt2-coco-en"
    lvllm = LargeVisionLanguageModel(model_id=model_id,
                                     cache_path=constants.LLM_CACHE_FILE_PATH)

    # Load example scene images
    img_paths = ['scene1.png', 'scene2.png']
    images = [Image.open(p) for p in img_paths]

    prompt = "Describe the relationship between object A and B:"
    text = lvllm.generate(prompt, images)
    print(f"Response: {text}")

    json_resp = lvllm.generate_json(prompt, images)
    print(f"JSON: {json_resp}")
