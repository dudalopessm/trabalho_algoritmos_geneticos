import os

# Dataset paths
HISTOGRAM_OUTPUT_PATH = "data/out/histograms/"
RETRIEVED_IMAGES_PATH = "data/out/retrieved_images/"
GA_FEATURE_SELECTION_OUTPUT_FILE = (
    "data/out/genetic_algorithm/best_selected_features.txt"
)
GA_RESULTS_CSV_FILE = "data/out/genetic_algorithm/ga_log.csv"

# Image file extension
IMAGE_FILE_EXTENSION = ".png"
IMAGE_FILE_EXTENSIONS_SUPPORTED = (".jpg", ".jpeg")

# Image processing constants
IMAGE_SIZE = (8, 8)  # Resize images to this size for faster processing
HISTOGRAM_BINS = 8  # Number of bins for 1D histograms
HISTOGRAM_2D_BINS = 8  # Number of bins for 2D histograms

# LBP settings
LBP_RADIUS = 1  # Radius for LBP calculation
LBP_POINTS = 8 * LBP_RADIUS  # Points around the radius for LBP
LBP_BINS = 2**LBP_POINTS  # Number of bins for LBP histograms

# Color spaces
COLOR_SPACES = {
    # "BGR": "BGR",
    # "RGB": "RGB",
    "HSV": "HSV",
    # "HLS": "HLS",
    "LAB": "LAB",
    "YUV": "YUV",
    # "YCrCb": "YCrCb",
    # "XYZ": "XYZ",
}

# KNN retrieval constants
K_NEIGHBORS = 10  # Number of images to retrieve
LEAF_SIZE = 100  # Leaf size for KNN (used in image retrieval)

# Evaluation metrics constants
NUM_CLASSES = 100  # Number of classes (varies by dataset: Corel-10K=100, Corel-1K=10, GHIM-10K=20, Olivia-2688=8, Produce-1400=14)
NUM_IMAGES_PER_CLASS = 100  # Varies by dataset

# Dataset configurations
DATASET_CONFIG = {
    "Corel-1K": {"num_images": 1000, "num_classes": 10},
    "Corel-10K": {"num_images": 10000, "num_classes": 100},
    "GHIM-10K": {"num_images": 10000, "num_classes": 20},
    "Produce-1400": {"num_images": 1400, "num_classes": 14},
    "Olivia-2688": {"num_images": 2688, "num_classes": 8},
}

# Directories
ONE_D_HISTOGRAMS_DIR = "1d_histograms"
TWO_D_HISTOGRAMS_DIR = "2d_histograms"
INTRA_COLORSPACE_DIR = "intra_colorspace"
INTER_COLORSPACE_DIR = "inter_colorspace"

# Genetic Algorithm Constants
GA_POPULATION_SIZE = 500  # Number of individuals in each generation
GA_NUMBER_OF_GENERATIONS = 1000  # Number of generations to evolve
GA_CROSSOVER_PROBABILITY = 0.85  # Probability of crossover between individuals
GA_BASE_MUTATION_PROBABILITY = 0.05  # Base probability of mutation in individuals
GA_MUTATION_INDEPENDENCE_PROBABILITY = 0.1  # Probability for each gene to mutate
GA_PRECISION_WEIGHT = 0.99  # Higher value prioritizes precision; lower value prioritizes minimizing features
TOURNAMENT_SIZE = 4  # Tournament size for selection
CROSSOVER_INDP_PROBABILITY = 0.7  # Probability for independent crossover per gene

# Define the path to the image datasets
IMAGE_DATASET_PATH = "datasets/"

# Path to the CSV file for storing results
CSV_FILE_PATH = "data/out/results.csv"