# Generative Topological Maps

A comprehensive system for semantic place segmentation and categorization, and semantic relationships inference on pre-built semantic maps using AI models and clustering algorithms.

## Overview

This repository provides tools for transforming raw spatial data into categorized places and inferred relationships through two main processing pipelines:

- **Place Understanding Pipeline**: Groups objects into spatial clusters and assigns semantic labels [2](#0-1) 
- **Relationship Inference Pipeline**: Analyzes spatial and semantic relationships between object pairs <cite/>

## Key Features

### Semantic Processing
- Multiple embedding models (BERT, RoBERTa, OpenAI, Sentence-BERT) [3](#0-2) 
- Dimensionality reduction with PCA/UMAP [4](#0-3) 
- LLM-based categorization using Gemini [5](#0-4) 

### Clustering Algorithms
- DBSCAN and HDBSCAN clustering [6](#0-5) 
- Post-processing with merge/split operations [7](#0-6) 
- Evaluation metrics (ARI, NMI, V-Measure, FMI) [8](#0-7) 

### Visualization Tools
- 2D/3D cluster visualization [9](#0-8) 
- Point cloud overlay visualization [10](#0-9) 
- Semantic embedding space analysis <cite/>

## Usage

### Place Categorization
```bash
python src/generative_place_categorization/main_places.py \
    --method METHOD_BERT \
    --clustering-algorithm dbscan \
    --stage segmentation
```

### Cluster Visualization
```bash
python src/generative_place_categorization/inspect_clusters.py \
    --clustering-file results/clustering.json \
    --semantic-map map_name \
    --visualization-method 2D
``` [11](#0-10) 

## Core Components

### Data Structures
- `Clustering`: Container for object clusters with evaluation capabilities [12](#0-11) 
- `SemanticMap`: Spatial object representation with semantic properties [13](#0-12) 

### Processing Engines
- `SemanticDescriptorEngine`: Multi-model embedding generation [14](#0-13) 
- `ClusteringEngine`: Spatial grouping algorithms [15](#0-14) 
- `DimensionalityReductionEngine`: Feature space optimization [16](#0-15) 

## Installation

```bash
pip install -r requirements.txt
```

Set up environment variables for API access (OpenAI, Google Gemini). <cite/>

## Notes

The system supports both embedding-based clustering and direct LLM segmentation approaches, with comprehensive visualization tools for analysis and validation. [17](#0-16)  The codebase includes extensive parameter configuration for systematic experimentation across different semantic descriptors and clustering methods. <cite/>

Wiki pages you might want to explore:
- [Processing Pipelines (MAPIRlab/generative-topological-maps)](/wiki/MAPIRlab/generative-topological-maps#4)
- [Visualization and Inspection Tools (MAPIRlab/generative-topological-maps)](/wiki/MAPIRlab/generative-topological-maps#7)