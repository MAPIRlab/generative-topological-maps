from scipy.spatial.distance import cdist
import argparse
import constants
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from embedding.bert_embedder import BERTEmbedder
from embedding.openai_embedder import OpenAIEmbedder
from embedding.roberta_embedder import RoBERTaEmbedder
from embedding.sentence_embedder import SentenceEmbedder
from llm.large_language_model import LargeLanguageModel
from prompt.sentence_generator_prompt import SentenceGeneratorPrompt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from dotenv import load_dotenv
load_dotenv()


def get_semantic_descriptor(semantic_descriptor_arg: str, word: str, verbose: bool = False):

    if semantic_descriptor_arg == constants.SEMANTIC_DESCRIPTOR_BERT:
        return bert_embedder.embed_text(word)
    elif semantic_descriptor_arg == constants.SEMANTIC_DESCRIPTOR_ROBERTA:
        return roberta_embedder.embed_text(word)
    elif semantic_descriptor_arg == constants.SEMANTIC_DESCRIPTOR_OPENAI:
        return openai_embedder.embed_text(word)
    elif semantic_descriptor_arg in (constants.SEMANTIC_DESCRIPTOR_DEEPSEEK_SBERT, constants.SEMANTIC_DESCRIPTOR_DEEPSEEK_OPENAI):

        # If LLM not instantiated, instantiate! SLOW PROCESS
        global deepseek_llm
        if deepseek_llm is None:
            _, deepseek_llm = LargeLanguageModel.create_from_huggingface(model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                                                                         cache_path=constants.LLM_CACHE_FILE_PATH)

        # Generate sentence
        sentence_generator_prompt = SentenceGeneratorPrompt(word=word)
        response = deepseek_llm.generate_json_retrying(
            sentence_generator_prompt.get_prompt_text(),
            params={"max_length": 1000},
            retries=10,
        )

        sentence = response["description"]

        # Generate embedding
        if semantic_descriptor_arg == constants.SEMANTIC_DESCRIPTOR_DEEPSEEK_SBERT:
            return sbert_embedder.embed_text(sentence)
        elif semantic_descriptor_arg == constants.SEMANTIC_DESCRIPTOR_DEEPSEEK_OPENAI:
            return openai_embedder.embed_text(sentence)
    else:
        raise NotImplementedError(
            f"Not implemented semantic descriptor {semantic_descriptor_arg}"
        )


def main():
    semantic_descriptors = []
    if args.semantic_descriptor == constants.SEMANTIC_DESCRIPTOR_ALL:
        semantic_descriptors = [
            constants.SEMANTIC_DESCRIPTOR_BERT,
            constants.SEMANTIC_DESCRIPTOR_ROBERTA,
            constants.SEMANTIC_DESCRIPTOR_OPENAI,
            constants.SEMANTIC_DESCRIPTOR_DEEPSEEK_SBERT,
            constants.SEMANTIC_DESCRIPTOR_DEEPSEEK_OPENAI]
    else:
        semantic_descriptors = [args.semantic_descriptor]

    for semantic_descriptor in semantic_descriptors:
        print(
            f"Showing word embeddings for semantic descriptor {semantic_descriptor}")

        # Get embeddings for each word
        embeddings = []
        for word in args.word_set:
            emb = get_semantic_descriptor(semantic_descriptor, word)
            embeddings.append(emb)

        # Convert to NumPy array (assuming each emb is 1D or can be flattened to 1D)
        embeddings = np.array(embeddings)

        # Apply PCA to reduce to 3 dimensions
        pca = PCA(n_components=3)
        embeddings_3d = pca.fit_transform(embeddings)

        # Compute pairwise distances and find the closest neighbor for each point
        # Compute all pairwise distances
        distances = cdist(embeddings_3d, embeddings_3d)
        np.fill_diagonal(distances, np.inf)  # Avoid self-matching
        # Get the index of the closest neighbor
        closest_neighbors = np.argmin(distances, axis=1)

        # Generate unique colors for each point using a colormap
        norm = mcolors.Normalize(vmin=0, vmax=len(args.word_set) - 1)
        # Using 'tab10' for distinct colors
        colormap = cm.get_cmap('tab10', len(args.word_set))
        colors = [colormap(norm(i)) for i in range(len(args.word_set))]

        # Plot the results in 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot points with assigned colors
        for i in range(len(args.word_set)):
            ax.scatter(embeddings_3d[i, 0], embeddings_3d[i, 1], embeddings_3d[i, 2],
                       color=colors[i], label=args.word_set[i])

        # Create dashed lines with variable segment lengths
        segments = []
        segment_colors = []
        for i, neighbor_idx in enumerate(closest_neighbors):
            # Determine number of segments dynamically (e.g., based on index or distance)
            distance = np.linalg.norm(
                embeddings_3d[i] - embeddings_3d[neighbor_idx])
            num_segments = min(5 + i, 20)  # Vary segment count per object
            if distance > 1.0:  # More segments for longer distances
                num_segments += 5

            # Generate evenly spaced points
            x_values = np.linspace(
                embeddings_3d[i, 0], embeddings_3d[neighbor_idx, 0], num=num_segments)
            y_values = np.linspace(
                embeddings_3d[i, 1], embeddings_3d[neighbor_idx, 1], num=num_segments)
            z_values = np.linspace(
                embeddings_3d[i, 2], embeddings_3d[neighbor_idx, 2], num=num_segments)

            # Create segments with gaps by skipping some points
            for j in range(0, len(x_values) - 1, 2):  # Skip every other segment
                segments.append([(x_values[j], y_values[j], z_values[j]),
                                 (x_values[j + 1], y_values[j + 1], z_values[j + 1])])
                segment_colors.append(colors[i])

        # Add dashed segments to the plot
        line_collection = Line3DCollection(
            segments, colors=segment_colors, linewidths=1.5, alpha=0.7)
        ax.add_collection3d(line_collection)

        # Label each point with its corresponding word
        for i, word in enumerate(args.word_set):
            x, y, z = embeddings_3d[i]
            ax.text(x, y, z, word)

        plt.title(f"3D PCA for {semantic_descriptor}")
        plt.show()


# Models
bert_embedder = None
roberta_embedder = None
openai_embedder = None
sbert_embedder = None
deepseek_llm = None

if __name__ == "__main__":
    # Instantiate models
    bert_embedder = BERTEmbedder()
    roberta_embedder = RoBERTaEmbedder()
    openai_embedder = OpenAIEmbedder()
    sbert_embedder = SentenceEmbedder()

    parser = argparse.ArgumentParser(
        description="Performs place categorization on a set of semantic map"
    )

    # SEMANTIC DESCRIPTOR parameters
    parser.add_argument(
        "-s",
        "--semantic-descriptor",
        help="How to compute the semantic descriptor.",
        choices=[
            constants.SEMANTIC_DESCRIPTOR_ALL,
            constants.SEMANTIC_DESCRIPTOR_BERT,
            constants.SEMANTIC_DESCRIPTOR_ROBERTA,
            constants.SEMANTIC_DESCRIPTOR_OPENAI,
            constants.SEMANTIC_DESCRIPTOR_DEEPSEEK_SBERT,
            constants.SEMANTIC_DESCRIPTOR_DEEPSEEK_OPENAI,
        ],
        required=True,
    )

    # Set of words argument
    parser.add_argument(
        "-w",
        "--word-set",
        help="A set of words for processing.",
        nargs="+",  # Accepts one or more words
        required=True,
    )

    args = parser.parse_args()

    main()
