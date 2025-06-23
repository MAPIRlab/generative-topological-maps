# Credentials
GOOGLE_GEMINI_CREDENTIALS_FILENAME = "credentials/procesamiento-documentos-ia-574a79820d6f.json"
GOOGLE_GEMINI_PROJECT_ID = "procesamiento-documentos-ia"
GOOGLE_GEMINI_PROJECT_LOCATION = "us-central1"

# Paths
SEMANTIC_MAPS_FOLDER_PATH = "data/semantic_maps/"
SEMANTIC_MAPS_PATHS = ["data/semantic_maps/scannet_scene0000_00.json",
                       "data/semantic_maps/scannet_scene0101_00.json",
                       "data/semantic_maps/scannet_scene0392_01.json",
                       "data/semantic_maps/scannet_scene0515_00.json",
                       "data/semantic_maps/scannet_scene0673_04.json",
                       "data/semantic_maps/scenenn_011.json",
                       "data/semantic_maps/scenenn_030.json",
                       "data/semantic_maps/scenenn_078.json",
                       "data/semantic_maps/scenenn_086.json",
                       "data/semantic_maps/scenenn_096.json"]
SEMANTIC_MAPS_COLORS_PATHS = ["/home/ubuntu/datasets/ScanNet/raw_data/scene0000_00/color/",
                              "/home/ubuntu/datasets/ScanNet/raw_data/scene0101_00/color/",
                              "/home/ubuntu/datasets/ScanNet/raw_data/scene0392_01/color/",
                              "/home/ubuntu/datasets/ScanNet/raw_data/scene0515_00/color/",
                              "/home/ubuntu/datasets/ScanNet/raw_data/scene0673_04/color/",
                              None,
                              None,
                              None,
                              None,
                              None]
RESULTS_FOLDER_PATH = "results/"
CLUSTERINGS_FOLDER_PATH = "data/clusterings/"
LLM_CACHE_FILE_PATH = "results/llm_cache.json"

#######################################################
# Constants for PLACE SEGMENTATION AND CATEGORIZATION
#######################################################

# STAGES
STAGE_SEGMENTATION = "segmentation"
STAGE_CATEGORIZATION = "categorization"

# METHODS
# Word embeddings
METHOD_GEOMETRIC = "geometric"
METHOD_BERT = "bert"
METHOD_OPENAI = "openai"
METHOD_ROBERTA = "roberta"

# Contextualized sentence embeddings
METHOD_LLM_SBERT = "llm+sbert"
METHOD_LLM_OPENAI = "llm+openai"

# Word embeddings + Cluster post-processing
METHOD_BERT_POST = "bert+post"

# Contextualized sentence embeddings + Cluster post-processing
METHOD_LLM_SBERT_POST = "llm+sbert+post"

# LLM only
METHOD_LLM = "llm"

# SEMANTIC DESCRIPTORS
SEMANTIC_DESCRIPTOR_ALL = "all"
SEMANTIC_DESCRIPTOR_BERT = "bert"
SEMANTIC_DESCRIPTOR_ROBERTA = "roberta"
SEMANTIC_DESCRIPTOR_OPENAI = "openai"
SEMANTIC_DESCRIPTOR_LLM_SBERT = "llm+sbert"
SEMANTIC_DESCRIPTOR_LLM_OPENAI = "llm+openai"

# DIMENSIONALITY REDUCTORS
DIM_REDUCTOR_PCA = "pca"
DIM_REDUCTOR_UMAP = "umap"

# CLUSTERING ALGORITHMS
CLUSTERING_ALGORITHM_DBSCAN = "dbscan"
CLUSTERING_ALGORITHM_HDBSCAN = "hdbscan"
CLUSTERING_ALGORITHM_KMEANS = "kmeans"

#######################################################
# Constants for RELATIONSHIPS INFERENCE
#######################################################

# METHODS
METHOD_LLM = "llm"
METHOD_LVLM = "lvlm"

# Others

BASE_64_EXAMPLE_IMAGE = "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"
