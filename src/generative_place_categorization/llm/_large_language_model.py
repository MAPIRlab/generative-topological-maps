import json
import os
from abc import ABC, abstractmethod
from typing import Optional, Tuple

from generative_place_categorization.prompt.conversation_history import (
    ConversationHistory,
)
from generative_place_categorization.utils import file_utils


class LargeLanguageModel(ABC):

    JSON_MAX_ATTEMPTS = 10

    def __init__(
        self,
        cache_path: Optional[str] = None,
    ):
        """
        Args:
            cache_path: if provided, path to a JSON file used as a prompt→response cache.
        """
        self._cache_path = cache_path
        if self._cache_path:
            # ensure file exists
            if not os.path.exists(self._cache_path):
                file_utils.save_dict_to_json_file({}, self._cache_path)

    @abstractmethod
    def get_provider_name(self) -> str:
        """Name of the LLM service provider."""
        pass

    @abstractmethod
    def _generate_text(
        self, conversation_history: ConversationHistory
    ) -> Tuple[str, float]:
        """
        Subclasses generate text+cost here. Do NOT implement caching in subclasses—
        caching is handled in the base class.
        """
        pass

    def generate_text(
        self, conversation_history: ConversationHistory
    ) -> Tuple[str, float]:
        """
        Concrete, cache-aware entrypoint. Checks disk cache, otherwise delegates
        to `_generate_text` and writes back to cache.
        """
        # serialize the conversation as a key
        key = json.dumps(
            conversation_history.get_chat_gpt_conversation_history(), sort_keys=True
        )

        # 1) Try cache
        cached = self.read_cache_entry(key)
        if cached is not None:
            return cached["text"], cached["cost"]

        # 2) Miss → delegate
        text, cost = self._generate_text(conversation_history)

        # 3) Write back
        self.write_cache_entry(key, {"text": text, "cost": cost})

        return text, cost

    def read_cache_entry(self, key: str) -> Optional[dict]:
        """Return {'text': ..., 'cost': ...} or None if missing / no cache."""
        if not self._cache_path:
            return None
        cache = file_utils.load_json(self._cache_path)
        return cache.get(key)

    def write_cache_entry(self, key: str, value: dict):
        """Store a single cache entry."""
        if not self._cache_path:
            return
        cache = file_utils.load_json(self._cache_path)
        cache[key] = value
        file_utils.save_dict_to_json_file(cache, self._cache_path)

    def clear_cache_entry(self, key: str):
        """Remove a specific entry from cache."""
        if not self._cache_path:
            return
        cache = file_utils.load_json(self._cache_path)
        if key in cache:
            del cache[key]
            file_utils.save_dict_to_json_file(cache, self._cache_path)

    def _clean_response(self, text: str) -> str:
        # ... your existing implementation ...
        pass

    def generate_json(self, conversation_history: ConversationHistory) -> Tuple[str, float]:
        # ... your existing implementation, which now benefits from cached generate_text() ...
        pass
