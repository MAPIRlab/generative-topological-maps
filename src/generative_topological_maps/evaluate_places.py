import argparse
import os
from typing import Dict, List

import numpy as np
import tqdm

from generative_topological_maps import constants
from generative_topological_maps.show.metrics_table import MetricsTable
from generative_topological_maps.utils import file_utils
from generative_topological_maps.voxeland.clustering import Clustering


def main(args):
    # ─── Load ground-truth clusterings ─────────────────────────────────────────
    ground_truth_clustering_results: Dict[str, List[Clustering]] = {}
    for fname in sorted(os.listdir(constants.CLUSTERINGS_FOLDER_PATH)):
        base = file_utils.get_file_basename(fname)
        semantic_map, opt = base.rsplit("_opt", 1)
        path = os.path.join(constants.CLUSTERINGS_FOLDER_PATH, fname)
        clustering = Clustering.load_from_json(path)
        ground_truth_clustering_results.setdefault(
            semantic_map, []).append(clustering)

    # ─── Load method-generated clusterings ─────────────────────────────────────
    methods_clustering_results: Dict[str, Dict[str, Clustering]] = {}
    for method in file_utils.list_subdirectories(constants.PLACES_RESULTS_FOLDER_PATH):
        methods_clustering_results[method] = {}
        mdir = os.path.join(constants.PLACES_RESULTS_FOLDER_PATH, method)
        for semmap in file_utils.list_subdirectories(mdir):
            cpath = os.path.join(mdir, semmap, "clustering.json")
            methods_clustering_results[method][semmap] = Clustering.load_from_json(
                cpath)

    # ─── Evaluate and pick best ground-truth option ────────────────────────────
    methods_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    for method in tqdm.tqdm(sorted(methods_clustering_results),
                            desc="Evaluating clustering results", unit="method"):
        methods_metrics[method] = {}
        for semmap, method_cr in sorted(methods_clustering_results[method].items()):
            best = {"ARI": -1.0, "NMI": -1.0, "V-Measure": -1.0, "FMI": -1.0}
            best_avg = -1.0
            for gt_cr in ground_truth_clustering_results.get(semmap, []):
                mets = method_cr.evaluate_against_ground_truth(gt_cr)
                avg = np.mean(list(mets.values()))
                if avg > best_avg:
                    best, best_avg = mets, avg
            methods_metrics[method][semmap] = {
                "ARI": best["ARI"],
                "NMI": best["NMI"],
                "V-Measure": best["V-Measure"],
                "FMI": best["FMI"],
            }

    # ─── RESULT 1: show top results via MetricsTable ──────────────────────────
    metrics_table = MetricsTable(methods_metrics)
    metrics_table.display_best(10, "ARI", ["Method"])
    metrics_table.display_best(10, "NMI", ["Method"])
    metrics_table.display_best(10, "V-Measure", ["Method"])
    metrics_table.display_best(10, "FMI", ["Method"])
    metrics_table.filter_dataset("scannet", group_by=["Method"])
    metrics_table.filter_dataset("scenenn", group_by=["Method"])
    metrics_table.semantic_map_vs_method_pivot_table("ARI")

    # ─── RESULT 2: aggregate per base-method and re-use MetricsTable ─────────
    print("\n[INFO] Starting RESULT 2: aggregate per base-method")

    # 1) add a “Base Method” column to the DataFrame
    print("[INFO] Extracting Base Method from Method column")
    metrics_table.df["Base Method"] = metrics_table.df[
        "Method"
    ].str.extract(r"^(geometric|bert|roberta|llm\+sbert)")

    print(f"[INFO] Unique Base Methods found: "
          f"{metrics_table.df['Base Method'].dropna().unique().tolist()}")

    # 2) keep only rows where Base Method is one of those four
    print("[INFO] Filtering rows to keep only valid Base Method entries")
    before_rows = len(metrics_table.df)
    metrics_table.df = metrics_table.df[metrics_table.df["Base Method"].notna(
    )]
    after_rows = len(metrics_table.df)
    print(
        f"[INFO] Rows before filter: {before_rows}, after filter: {after_rows}")

    # 3) display the mean of each metric grouped by Base Method
    print("\n" + "*" * 100)
    print("[INFO] Displaying mean metrics per Base Method")
    metrics_table.display_best(10, "NMI", group_by=["Base Method"])
    metrics_table.filter_dataset("scannet", group_by=["Base Method"])
    metrics_table.filter_dataset("scenenn", group_by=["Base Method"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluates different clusterings")
    args = parser.parse_args()
    main(args)
