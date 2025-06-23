import hashlib
import io
import json
from typing import List, Optional, Tuple

from PIL import Image as PILImage
from vertexai.generative_models import Content
from vertexai.generative_models import Image as VAIImage
from vertexai.generative_models import Part

from generative_place_categorization.utils import dict_utils

ROLE_SYSTEM = "system"
ROLE_ASSISTANT = "assistant"
ROLE_USER = "user"
ROLE_MODEL = "model"

KEY_ROLE = "role"
KEY_CONTENT = "content"
KEY_IMAGE = "image"  # for PIL images only


class ConversationHistory:

    @staticmethod
    def system_message(system_text: str) -> dict:
        return {KEY_ROLE: ROLE_SYSTEM, KEY_CONTENT: system_text}

    @staticmethod
    def assistant_message(assistant_text: str) -> dict:
        return {KEY_ROLE: ROLE_ASSISTANT, KEY_CONTENT: assistant_text}

    @staticmethod
    def user_message(user_text: str) -> dict:
        return {KEY_ROLE: ROLE_USER, KEY_CONTENT: user_text}

    @staticmethod
    def user_image_message(image: PILImage.Image) -> dict:
        """
        Creates a user message carrying a PIL Image.
        """
        if not isinstance(image, PILImage.Image):
            raise TypeError("user_image_message expects a PIL.Image.Image")
        return {KEY_ROLE: ROLE_USER, KEY_IMAGE: image}

    @staticmethod
    def create_from_user_message(user_text: str) -> "ConversationHistory":
        history = ConversationHistory()
        history.append_user_message(user_text)
        return history

    def __init__(self):
        # Holds raw dicts of messages (text or image)
        self.conversation_history_list: List[dict] = []

    def __str__(self):
        # Build a human-readable log of the conversation
        lines = []
        for msg in self.conversation_history_list:
            role = msg[KEY_ROLE].capitalize()
            if KEY_CONTENT in msg:
                text = msg[KEY_CONTENT].replace("\n", " ").replace("\t", " ")
                lines.append(f"{role}: {text}...")
            else:
                lines.append(f"{role}: <image>...")
        return "\n".join(lines)

    def append_system_message(self, system_text: str):
        self.conversation_history_list.append(self.system_message(system_text))

    def append_assistant_message(self, assistant_text: str):
        self.conversation_history_list.append(
            self.assistant_message(assistant_text))

    def append_user_message(self, user_text: str):
        self.conversation_history_list.append(self.user_message(user_text))

    def append_user_image(self, image: PILImage.Image):
        """
        Append a user message containing a PIL Image.
        """
        self.conversation_history_list.append(self.user_image_message(image))

    def clear(self):
        self.conversation_history_list.clear()

    def get_chat_gpt_conversation_history(self) -> List[dict]:
        # Return the raw history dicts
        return self.conversation_history_list

    def get_gemini_conversation_history(self) -> Tuple[Optional[str], List[Content]]:
        """
        Convert raw history into Gemini-ready (system_instruction, [Content...]).
        Images become VAIImage → Part.from_image.
        """
        # 1) Extract system instruction (if any)
        sys_msg = dict_utils.search_dict_by_key_value(
            self.conversation_history_list, KEY_ROLE, ROLE_SYSTEM
        )
        system_instruction = sys_msg.get(KEY_CONTENT) if sys_msg else None

        # 2) Special case: only a system message → treat its text as a single user prompt
        if (
            len(self.conversation_history_list) == 1
            and self.conversation_history_list[0][KEY_ROLE] == ROLE_SYSTEM
        ):
            only_text = self.conversation_history_list[0][KEY_CONTENT]
            return system_instruction, [
                Content(role=ROLE_USER, parts=[Part.from_text(only_text)])
            ]

        # 3) Otherwise, convert each non-system message
        contents: List[Content] = []
        for msg in self.conversation_history_list:
            role = msg[KEY_ROLE]
            if role == ROLE_SYSTEM:
                continue

            gemini_role = ROLE_MODEL if role == ROLE_ASSISTANT else role
            part = self._msg_to_part(msg)
            contents.append(Content(role=gemini_role, parts=[part]))

        return system_instruction, contents

    def to_serializable(self) -> List[dict]:
        """
        Return a JSON-serializable version of the history, where:
        - text messages keep their 'content'
        - image messages are replaced by an MD5 hash of their PNG bytes
        """
        serializable = []
        for msg in self.conversation_history_list:
            entry = {KEY_ROLE: msg[KEY_ROLE]}
            if KEY_CONTENT in msg:
                entry[KEY_CONTENT] = msg[KEY_CONTENT]
            else:
                # compute a hash of the image bytes for stability
                pil_img = msg[KEY_IMAGE]
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                digest = hashlib.md5(buf.getvalue()).hexdigest()
                entry[KEY_IMAGE] = digest
            serializable.append(entry)
        return serializable

    def get_cache_key(self) -> str:
        """
        Returns a stable JSON string to use as a cache key,
        by dumping the serializable form with sorted keys.
        """
        return json.dumps(self.to_serializable(), sort_keys=True)

    def _msg_to_part(self, msg: dict) -> Part:
        """
        Turn a single message dict into a Part (text or image) for Gemini.
        """
        if KEY_IMAGE in msg:
            pil_img = msg[KEY_IMAGE]
            if not isinstance(pil_img, PILImage.Image):
                raise TypeError("Expected a PIL.Image.Image in image messages")
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            vai_img = VAIImage.from_bytes(buf.getvalue())
            return Part.from_image(vai_img)
        else:
            return Part.from_text(msg[KEY_CONTENT])
