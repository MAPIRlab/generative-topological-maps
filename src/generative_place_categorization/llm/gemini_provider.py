

import base64
from io import BytesIO
from typing import Optional

import google.cloud.aiplatform as aiplatform
import google.oauth2.service_account
from PIL import Image as PILImage
from vertexai.preview.generative_models import GenerativeModel

from generative_place_categorization import constants
from generative_place_categorization.llm.large_language_model import LargeLanguageModel
from generative_place_categorization.prompt.conversation_history import (
    ConversationHistory,
)


class GeminiProvider(LargeLanguageModel):

    GEMINI_1_0_PRO = "gemini-1.0-pro"
    GEMINI_1_0_PRO_VISION = "gemini-1.0-pro-vision"
    GEMINI_1_5_PRO = "gemini-1.5-pro"
    GEMINI_2_0_FLASH = "gemini-2.0-flash"
    GEMINI_2_5 = "gemini-2.5"

    def __init__(
        self,
        credentials_file: str,
        project_id: str,
        project_location: str,
        model_name: str,
        cache_path: Optional[str] = None,
    ):
        """
        Initialize the GeminiProvider with the specified credentials and optional cache.

        Args:
            credentials_file (str): Path to the service account credentials file.
            project_id (str): Google Cloud project ID.
            project_location (str): Google Cloud project location.
            model_name (str): Name of the model to be used.
            cache_path (Optional[str]): Path to JSON cache file for prompt→response.
        """
        # Initialize the base class (sets up on-disk cache if provided)
        super().__init__(cache_path=cache_path)

        # Now set up Google credentials & Vertex AI
        creds = google.oauth2.service_account.Credentials.from_service_account_file(
            filename=credentials_file
        )
        aiplatform.init(
            project=project_id,
            location=project_location,
            credentials=creds,
        )

        self.model_name = model_name

    def get_provider_name(self) -> str:
        return f"Google_{self.model_name}"

    def _generate_text(self, conversation_history: ConversationHistory) -> str:
        """
        Generates a text response for a conversation that may include image messages.
        """
        system_instruction, contents = conversation_history.get_gemini_conversation_history()
        model = GenerativeModel(
            model_name=self.model_name,
            system_instruction=system_instruction
        )
        response = model.generate_content(contents=contents)
        return response.candidates[0].content.parts[0].text


if __name__ == "__main__":
    llm = GeminiProvider(
        credentials_file=constants.GOOGLE_GEMINI_CREDENTIALS_FILENAME,
        project_id=constants.GOOGLE_GEMINI_PROJECT_ID,
        project_location=constants.GOOGLE_GEMINI_PROJECT_LOCATION,
        model_name=GeminiProvider.GEMINI_2_0_FLASH,
        cache_path=constants.LLM_CACHE_FILE_PATH
    )

    # Test text generation from prompt
    response = llm.generate_text(
        ConversationHistory.create_from_user_message("Cuéntame un cuento!")
    )
    print("=== Text-only response ===")
    print(response)
    print()

    # Test text generation from prompt + image
    history = ConversationHistory()
    history.append_system_message(
        "Eres un asistente que describe imágenes con detalle."
    )
    history.append_user_message("¿Qué ves en esta imagen?")

    # Carga la imagen con PIL antes de pasarla al historial
    base64_img = constants.BASE_64_EXAMPLE_IMAGE
    img_bytes = base64.b64decode(base64_img)
    img = PILImage.open(BytesIO(img_bytes))
    history.append_user_image(img)

    response_with_image = llm.generate_text(history)
    print("=== Text + Image response ===")
    print(response_with_image)
