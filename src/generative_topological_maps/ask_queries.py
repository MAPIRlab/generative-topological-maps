import json
import os
from typing import List

from generative_topological_maps import constants
from generative_topological_maps.prompt.complex_scene_helper_prompt import (
    ComplexSceneHelperPrompt,
)
from generative_topological_maps.prompt.simple_scene_helper_prompt import (
    SimpleSceneHelperPrompt,
)
from generative_topological_maps.utils import file_utils
from generative_topological_maps.voxeland.semantic_map import SemanticMap

if __name__ == "__main__":

    queries = [
        "Which stool is between two other stools in the scene?",
        "Which stool is closest to the bathroom?",
        "What objects are on the countertop?",
        "How many garbage cans are in the kitchen?",
        "What object can be filled with discarded tissues?"
    ]

    # Load and pre-process semantic maps
    semantic_maps: List[SemanticMap] = []
    for json_file in constants.SEMANTIC_MAPS_PATHS:
        # Create a SemanticMap directly from its JSON file
        sm = SemanticMap.from_json_path(str(json_file))
        semantic_maps.append(sm)

    semantic_map: SemanticMap = semantic_maps[0]

    for query_id, query in enumerate(queries):
        places_path = os.path.join(constants.PLACES_RESULTS_FOLDER_PATH,
                                   "llm_gemini_2_0_flash",
                                   semantic_map.semantic_map_id,
                                   "clustering.json")
        places = file_utils.load_json(places_path)
        relationships_path = os.path.join(
            constants.RELATIONSHIPS_RESULTS_FOLDER_PATH,
            "lvlm",
            semantic_map.semantic_map_id,
            "relationships.json")
        relationships = file_utils.load_json(relationships_path)

        simple_prompt = SimpleSceneHelperPrompt(
            semantic_map=semantic_map.get_prompt_json_representation(),
            question=query)
        complex_prompt = ComplexSceneHelperPrompt(
            semantic_map=semantic_map.get_prompt_json_representation(),
            places=json.dumps(places),
            relationships=json.dumps(relationships),
            question=query)

        simple_output_path = os.path.join(
            constants.RESULTS_FOLDER_PATH,
            "queries",
            semantic_map.semantic_map_id,
            f"query_{query_id}_simple.txt"
        )
        file_utils.create_directories_for_file(simple_output_path)
        file_utils.save_text_to_file(
            simple_prompt.get_prompt_text(), simple_output_path)
        complex_output_path = os.path.join(
            constants.RESULTS_FOLDER_PATH,
            "queries",
            semantic_map.semantic_map_id,
            f"query_{query_id}_complex.txt"
        )
        file_utils.create_directories_for_file(complex_output_path)
        file_utils.save_text_to_file(
            complex_prompt.get_prompt_text(), complex_output_path)
