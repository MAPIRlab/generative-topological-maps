# Generative Topological Maps

Generative Topological Maps is a framework for semantic place segmentation, categorization, and inference of spatial relationships within pre-built semantic maps. 
It combines clustering techniques with generative AI models to deliver accurate and interpretable map analyses.

This work was completed as the Master’s Thesis of Jesús Moncada Ramírez for the M.Sc. in Mechatronics Engineering at the University of Málaga.  
- **[Full Thesis Report](doc/pdfs/tfm_report.pdf)**  
- **[Presentation Slides](doc/pdfs/tfm_presentation.pdf)**  

![Master Thesis title and author](doc/images/tfm_title.png)

A core component of the place segmentation and categorization pipeline has been accepted for presentation at the European Conference on Mobile Robotics (ECMR) 2025 in Padua, Italy.  
- **[ECMR 2025 Conference Paper](doc/pdfs/ecmr_paper.pdf)**  

![ECMR 2025 paper title and authors](doc/images/ecmr_title.png)

## Setup

In order to run the Python code in this repository you need to:

1. **Create and activate a virtual environment**  
    ```bash
    python -m venv venv
    # On macOS/Linux
    source venv/bin/activate
    # On Windows (PowerShell)
    .\venv\Scripts\Activate.ps1
    ```
2. **Install dependencies**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

## Overview

The project provides three main scripts to run the core pipelines:

- **`tfm_places.sh`**  
  Executes the full place segmentation and categorization pipeline as detailed in the Master’s thesis.  
  ```bash
  bash tfm_places.sh
  ```  
  Outputs are saved to `results/places_results/`.

- **`tfm_relationships.sh`**  
  Infers and exports the spatial relationship data between segmented places using our generative AI models.  
  ```bash
  bash tfm_relationships.sh
  ```  
  Results are written to `results/relationships_results/`.

- **`ecmr.sh`**  
  Generates the semantic place segmentation results used in the ECMR 2025 submission.  
  ```bash
  bash ecmr.sh
  ```  
  This will produce the ECMR-specific output files under the designated `results/places_results/` directory.
  Please note that, due to minor implementation adjustments, your results may differ slightly from those reported in the paper. 
  The model hierarchy, however, remains unchanged.

## Code organization

The project is structured into individual modules and focused subpackages, each responsible for a distinct part of the pipeline:

- **`check_places_ground_truth.py`**
    Checks the ground-truth place segmentation annotations to ensure consistency and correctness.

- **`constants.py`**  
    Centralizes global constants: file paths, method names, default parameter values, environment variable keys, etc.

- **`evaluate_places.py`**  
    Computes quantitative metrics by comparing the place segmentation outputs to the ground truth.

- **`ask_queries.py`**  
    Generates the queries to perform to the LLM to evaluate its behavior in robotic tasks when equipped with a semantic map, and a semantic-topological map.

- **`inspect_clusters.py`**  
    Provides routines to visualize cluster (places) compositions.

- **`inspect_semantics.py`**  
    Offers graphical representations for examining the distribution of semantic descriptors.

- **`main_places.py`**  
  **Entry point for place segmentation & categorization.**  
  1. Argument parsing (`--stage`, embedding method, clustering params, LLM flags, etc.)  
  2. Setup: loads `.env`, instantiates embedders & LLM clients, engines.  
  3. **Segmentation**: builds descriptors, reduces dimensions, clusters, post‑processes, saves JSON & plots.  
  4. **Categorization**: loads clusters, constructs prompts, optionally sends LLM requests, writes tags/descriptions.

- **`main_relationships.py`**  
  **Entry point for spatial relationship inference.**  
  1. Argument parsing (`-g`, `--method`, `--num-images`, etc.)  
  2. Setup: loads `.env`, instantiates LLM client.  
  3. Finds object pairs, builds text or multimodal prompts, optionally sends LLM/LVLM requests, aggregates into `relationships.json`.

- **`embedding/`**  
  Text‑embedding backends and base classes (`all_mpnet_base_v2_embedder.py`, `bert_embedder.py`, etc.).

- **`llm/`**  
  LLM provider abstractions (`gemini_provider.py`, `huggingface_large_language_model.py`, `large_language_model.py`).

- **`prompt/`**  
  Prompt templates and history management (`place_segmenter_prompt.py`, `conversation_history.py`, etc.).

- **`semantic/`**  
  Engines for descriptor computation, dimensionality reduction, and clustering (`semantic_descriptor_engine.py`, `clustering_engine.py`, etc.).

- **`show/`**  
  Visualization helpers (`metrics_table.py`).

- **`utils/`**  
  General utilities for file I/O and data structures (`file_utils.py`, `dict_utils.py`).

- **`voxeland/`**  
  Domain classes for loading and handling semantic maps (`semantic_map.py`, `cluster.py`, etc.).
