#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Any, Dict, List

import tqdm
from dotenv import load_dotenv

import generative_place_categorization.constants as constants
from generative_place_categorization.llm.gemini_provider import GeminiProvider
from generative_place_categorization.prompt.conversation_history import (
    ConversationHistory,
)
from generative_place_categorization.prompt.relationship_inferer_prompt import (
    RelationshipInfererPrompt,
)
from generative_place_categorization.prompt.relationship_inferer_with_image_prompt import (
    RelationshipInfererWithImagePrompt,
)
from generative_place_categorization.utils import file_utils
from generative_place_categorization.voxeland.semantic_map import SemanticMap


def main_relationships(args):
    # Load environment and instantiate LLM
    load_dotenv()
    llm = GeminiProvider(
        credentials_file=constants.GOOGLE_GEMINI_CREDENTIALS_FILENAME,
        project_id=constants.GOOGLE_GEMINI_PROJECT_ID,
        project_location=constants.GOOGLE_GEMINI_PROJECT_LOCATION,
        model_name=GeminiProvider.GEMINI_2_0_FLASH,
        cache_path=constants.LLM_CACHE_FILE_PATH
    )

    # Load semantic maps
    semantic_maps: List[SemanticMap] = []
    for semantic_map_path, colors_path in zip(constants.SEMANTIC_MAPS_PATHS, constants.SEMANTIC_MAPS_COLORS_PATHS):
        sm = SemanticMap.from_json_path(
            semantic_map_path, colors_path=colors_path)
        semantic_maps.append(sm)

    # For each map, infer relations and aggregate into a single JSON
    for semantic_map in semantic_maps:
        base_dir = os.path.join(
            constants.RESULTS_FOLDER_PATH,
            "relationships_results",
            args.method,
            semantic_map.semantic_map_id
        )
        out_path = os.path.join(base_dir, "relationships.json")
        # Ensure output directory exists
        file_utils.create_directories_for_file(out_path)

        all_relationships: List[Dict[str, Any]] = []

        pairs = semantic_map.get_close_object_pairs(
            threshold_distance=args.geometric_threshold,
            include_all_classes=False
        )
        print(
            f"# Processing {len(pairs)} pairs of objects for relationships in {semantic_map.semantic_map_id} with method {args.method}..."
        )

        for obj1, obj2 in tqdm.tqdm(
            pairs, desc=f"Processing pairs for map {semantic_map.semantic_map_id}"
        ):
            pair_dir = os.path.join(
                base_dir, f"{obj1.object_id}_{obj2.object_id}")

            prompt_data = {
                "object1_id": obj1.object_id,
                "object1_name": obj1.get_most_probable_class(),
                "object1_centroid": str(obj1.bbox_center),
                "object1_size": str(obj1.bbox_size),
                "object2_id": obj2.object_id,
                "object2_name": obj2.get_most_probable_class(),
                "object2_centroid": str(obj2.bbox_center),
                "object2_size": str(obj2.bbox_size),
            }

            if args.method == constants.METHOD_LLM:
                # LLM-only prompt
                prompt_obj = RelationshipInfererPrompt(**prompt_data)
                conv_his = ConversationHistory.create_from_user_message(
                    prompt_obj.get_prompt_text()
                )

                if args.llm_request:
                    response = llm.generate_json(
                        conversation_history=conv_his, retries=3
                    )
                    print(response)
                    all_relationships.extend(response)

                    # Update relationships.json on disk
                    with open(out_path, "w") as f:
                        json.dump(
                            {"relationships": all_relationships}, f, indent=2)
                    print(
                        f"[{semantic_map.semantic_map_id}] relationships.json updated "
                        f"({len(all_relationships)} total)"
                    )

                # Save prompt text
                prompt_path = os.path.join(pair_dir, "prompt.txt")
                file_utils.create_directories_for_file(prompt_path)
                file_utils.save_text_to_file(
                    prompt_obj.get_prompt_text(), prompt_path
                )
                print(
                    f"[{semantic_map.semantic_map_id}] Prompt saved to {prompt_path}")

            elif args.method == constants.METHOD_LVLM:
                # LVLM prompt with images
                prompt_obj = RelationshipInfererWithImagePrompt(**prompt_data)

                # Collect a few representative frames
                frames_ids = semantic_map.get_common_frames_by_pixel_count(
                    obj1.object_id, obj2.object_id
                )
                top_frames_ids = frames_ids[: args.num_images]
                frames_imgs = [
                    semantic_map.get_frame_image(frame_id)
                    for frame_id in top_frames_ids
                ]

                # Build multimodal conversation
                conv_his = ConversationHistory()
                conv_his.append_user_message(prompt_obj.get_prompt_text())
                for frame_img in frames_imgs:
                    conv_his.append_user_image(frame_img)

                if args.llm_request:
                    response = llm.generate_json(
                        conversation_history=conv_his, retries=3
                    )
                    print(response)
                    all_relationships.extend(response)

                    # Update relationships.json on disk
                    with open(out_path, "w") as f:
                        json.dump(
                            {"relationships": all_relationships}, f, indent=2)
                    print(
                        f"[{semantic_map.semantic_map_id}] relationships.json updated "
                        f"({len(all_relationships)} total)"
                    )

                # Save prompt text
                prompt_path = os.path.join(pair_dir, "prompt.txt")
                file_utils.create_directories_for_file(prompt_path)
                file_utils.save_text_to_file(
                    prompt_obj.get_prompt_text(), prompt_path
                )
                print(
                    f"[{semantic_map.semantic_map_id}] Prompt saved to {prompt_path}")

                # Save image frames
                for rank, (frame_id, frame_img) in enumerate(
                    zip(top_frames_ids, frames_imgs), start=1
                ):
                    filename = f"{rank:02d}_{frame_id}.png"
                    print(
                        f"[{semantic_map.semantic_map_id}] Saving frame "
                        f"{frame_id} as {filename}"
                    )
                    frame_img.save(os.path.join(pair_dir, filename))

        if args.llm_request:
            print(
                f"[{semantic_map.semantic_map_id}] Done. "
                f"Final relationships.json has {len(all_relationships)} entries."
            )

    print("[main_relationships] Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Infer semantic relationships between objects in semantic maps."
    )
    parser.add_argument(
        "-p", "--persist-log",
        action="store_true",
        help="Redirect output to a log file."
    )
    parser.add_argument(
        "-n", "--number-maps",
        type=int,
        default=10,
        help="Number of semantic maps to process."
    )
    parser.add_argument(
        "-g", "--geometric-threshold",
        type=float,
        required=True,
        help="Max centroid distance to consider object pairs."
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=[constants.METHOD_LLM, constants.METHOD_LVLM],
        help="Inference method: llm or lvlm."
    )
    parser.add_argument(
        "--llm-request",
        action="store_true",
        help="Whether to perform the LLM request or only generate prompts."
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=5,
        help="Max number of frames/images for LVLM."
    )
    args = parser.parse_args()

    # Redirect logs if requested
    if args.persist_log:
        log_path = os.path.join(
            constants.RESULTS_FOLDER_PATH, "relationships_log.txt"
        )
        file_utils.create_directories_for_file(log_path)
        sys.stdout = open(log_path, 'w')
        sys.stderr = sys.stdout

    main_relationships(args)
