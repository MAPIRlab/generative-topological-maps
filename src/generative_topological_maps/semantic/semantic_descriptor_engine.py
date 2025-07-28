

from generative_topological_maps import constants
from generative_topological_maps.embedding.bert_embedder import BERTEmbedder
from generative_topological_maps.embedding.openai_embedder import OpenAIEmbedder
from generative_topological_maps.embedding.roberta_embedder import RoBERTaEmbedder
from generative_topological_maps.embedding.all_mpnet_base_v2_embedder import (
    AllMpnetBaseV2Embedder,
)
from generative_topological_maps.llm.large_language_model import LargeLanguageModel
from generative_topological_maps.prompt.conversation_history import (
    ConversationHistory,
)
from generative_topological_maps.prompt.sentence_generator_prompt import (
    SentenceGeneratorPrompt,
)


class SemanticDescriptorEngine:
    """
    Engine for generating semantic descriptors using various embedding models.
    """

    def __init__(self,
                 bert_embedder: BERTEmbedder,
                 roberta_embedder: RoBERTaEmbedder,
                 openai_embedder: OpenAIEmbedder,
                 sbert_embedder: AllMpnetBaseV2Embedder,
                 llm: LargeLanguageModel = None):
        """
        Initialize the SemanticDescriptorEngine with embedding models.

        Args:
            bert_embedder (BERTEmbedder): BERT embedding model.
            roberta_embedder (RoBERTaEmbedder): RoBERTa embedding model.
            openai_embedder (OpenAIEmbedder): OpenAI embedding model.
            sbert_embedder (SentenceBERTEmbedder): Sentence-BERT embedding model.
            llm (LargeLanguageModel, optional): Large Language Model for generating sentences. Defaults to None.
        """
        self.bert_embedder = bert_embedder
        self.roberta_embedder = roberta_embedder
        self.openai_embedder = openai_embedder
        self.sbert_embedder = sbert_embedder
        self.llm = llm

    def get_semantic_descriptor_from_method(self, method_arg: str, text: str):
        """
        Retrieve a semantic descriptor based on the specified method and input text.
        Args:
            method_arg (str): The method identifier for selecting the semantic descriptor.
            text (str): The input text to process.
        Returns:
            list or object: The semantic descriptor corresponding to the method.
        Raises:
            NotImplementedError: If the specified method is not supported.
        """
        if method_arg == constants.METHOD_GEOMETRIC:
            return []
        elif method_arg in (constants.METHOD_BERT, constants.METHOD_BERT_POST):
            return self.get_semantic_descriptor(constants.SEMANTIC_DESCRIPTOR_BERT, text)
        elif method_arg == constants.METHOD_ROBERTA:
            return self.get_semantic_descriptor(constants.SEMANTIC_DESCRIPTOR_ROBERTA, text)
        elif method_arg == constants.METHOD_OPENAI:
            return self.get_semantic_descriptor(constants.SEMANTIC_DESCRIPTOR_OPENAI, text)
        elif method_arg in (constants.METHOD_LLM_SBERT, constants.METHOD_LLM_SBERT_POST):
            return self.get_semantic_descriptor(constants.SEMANTIC_DESCRIPTOR_LLM_SBERT, text)
        elif method_arg == constants.METHOD_LLM_OPENAI:
            return self.get_semantic_descriptor(constants.SEMANTIC_DESCRIPTOR_LLM_OPENAI, text)
        else:
            raise NotImplementedError(
                f"Not implemented semantic descriptor {method_arg}")

    def get_semantic_descriptor(self, semantic_descriptor_arg: str, text: str):
        """
        Generate a semantic descriptor for a given word using the specified embedding model.

        Args:
            semantic_descriptor_arg (str): Type of semantic descriptor to generate.
            word (str): Input word for which the descriptor is generated.

        Returns:
            Any: The generated semantic descriptor.

        Raises:
            NotImplementedError: If the specified semantic descriptor type is not implemented.
        """
        if semantic_descriptor_arg == constants.SEMANTIC_DESCRIPTOR_BERT:
            return self.bert_embedder.embed_text(text)
        elif semantic_descriptor_arg == constants.SEMANTIC_DESCRIPTOR_ROBERTA:
            return self.roberta_embedder.embed_text(text)
        elif semantic_descriptor_arg == constants.SEMANTIC_DESCRIPTOR_OPENAI:
            return self.openai_embedder.embed_text(text)
        elif semantic_descriptor_arg in (constants.SEMANTIC_DESCRIPTOR_LLM_SBERT, constants.SEMANTIC_DESCRIPTOR_LLM_OPENAI):

            # Generate sentence
            sentence_generator_prompt = SentenceGeneratorPrompt(word=text)
            conv_his = ConversationHistory.create_from_user_message(
                sentence_generator_prompt.get_prompt_text())
            response = self.llm.generate_json(
                conversation_history=conv_his,
                retries=3,
            )

            sentence = response["description"]

            # Generate embedding
            if semantic_descriptor_arg == constants.SEMANTIC_DESCRIPTOR_LLM_SBERT:
                return self.sbert_embedder.embed_text(sentence)
            elif semantic_descriptor_arg == constants.SEMANTIC_DESCRIPTOR_LLM_OPENAI:
                return self.openai_embedder.embed_text(sentence)
        else:
            raise NotImplementedError(
                f"Not implemented semantic descriptor {semantic_descriptor_arg}"
            )
