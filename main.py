from src.image_processing.process_image import (
    process_all_images,
    process_dataset,
    #generate_dataset_statistics,
)
from src.image_processing.knn_image_retrieval import retrieve_and_save_images_for_all_dataset
from src.genetic_algorithm.evaluation import evaluate_all_retrievals
from src.utils.genetic_algorithm_utils import (
    run_genetic_algorithm,
)
from src.utils.constants import HISTOGRAM_BINS, LBP_POINTS, LBP_RADIUS

if __name__ == "__main__":
    # =============== Image Feature Generation ===============
    # process_all_images()
    #=========================================================
    
    # =============== Retrieval and Evaluation ===============
    # After features are generated, uncomment these:
    # Retrieve and save images for each image in the dataset
    retrieve_and_save_images_for_all_dataset()
   
    # Evaluate the retrieval performance
    #evaluate_all_retrievals()
    #=========================================================

    pass