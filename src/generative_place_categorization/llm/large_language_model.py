import json
import os
from abc import ABC, abstractmethod
from typing import Optional

from generative_place_categorization.prompt.conversation_history import (
    ConversationHistory,
)
from generative_place_categorization.utils import file_utils


class LargeLanguageModel(ABC):

    JSON_MAX_ATTEMPTS = 10

    def __init__(self, cache_path: Optional[str] = None):
        """
        Args:
            cache_path: optional path to a JSON file used as a prompt→response cache.
        """
        self._cache_path = cache_path
        if self._cache_path and not os.path.exists(self._cache_path):
            file_utils.save_dict_to_json_file({}, self._cache_path)

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the name of the LLM service provider."""
        pass

    @abstractmethod
    def _generate_text(self, conversation_history: ConversationHistory) -> str:
        """
        Subclasses implement this to actually call their LLM provider
        and return the generated text.
        """
        pass

    def generate_text(self, conversation_history: ConversationHistory) -> str:
        """
        Cache-aware entrypoint. Serializes the conversation history as a key,
        checks the cache, and on miss delegates to `_generate_text`.
        """
        key = conversation_history.get_cache_key()

        # 1) Try cache
        if self._cache_path:
            cache = file_utils.load_json(self._cache_path)
            if key in cache:
                # print(f"[generate_text] Cache hit for key: {key}")
                return cache[key]

        # 2) Miss → call provider
        text = self._generate_text(conversation_history)

        # 3) Write back
        if self._cache_path:
            cache[key] = text
            file_utils.save_dict_to_json_file(cache, self._cache_path)

        return text

    def read_cache_entry(self, key: str) -> Optional[str]:
        if not self._cache_path:
            return None
        cache = file_utils.load_json(self._cache_path)
        return cache.get(key)

    def write_cache_entry(self, key: str, response: str):
        if not self._cache_path:
            return
        cache = file_utils.load_json(self._cache_path)
        cache[key] = response
        file_utils.save_dict_to_json_file(cache, self._cache_path)

    def clear_cache_entry(self, key: str):
        if not self._cache_path:
            return
        cache = file_utils.load_json(self._cache_path)
        cache.pop(key, None)
        file_utils.save_dict_to_json_file(cache, self._cache_path)

    def _clean_response(self, text: str) -> str:
        """
        (unchanged) Extracts the first JSON-like substring from `text`.
        """
        obj_start = text.find("{")
        obj_end = text.rfind("}") + 1
        arr_start = text.find("[")
        arr_end = text.rfind("]") + 1

        # pick the earliest valid start
        if (obj_start == -1 or (arr_start != -1 and arr_start < obj_start)) and arr_start != -1:
            start, end = arr_start, arr_end
        else:
            start, end = obj_start, obj_end

        if 0 <= start < end <= len(text):
            return text[start:end]
        return ""

    def generate_json(
        self,
        conversation_history: ConversationHistory,
        retries: Optional[int] = None
    ) -> dict:
        """
        Attempts up to `retries` times (or JSON_MAX_ATTEMPTS if None) to get valid JSON
        from the LLM, parses it, and returns the resulting dict (or list). Returns {}
        on failure.

        Args:
            conversation_history: History to use as context.
            retries: Maximum number of attempts to parse valid JSON.
        """
        print("El prompt es:")
        print(conversation_history)
        max_attempts = retries or self.JSON_MAX_ATTEMPTS

        for attempt in range(1, max_attempts + 1):
            raw = self.generate_text(conversation_history)
            cleaned = self._clean_response(raw)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                print(
                    f"[LargeLanguageModel.generate_json] Attempt {attempt}/{max_attempts} failed to parse JSON: {cleaned}")

        print(
            f"[LargeLanguageModel.generate_json] All {max_attempts} attempts failed; returning empty dict")
        return {}
