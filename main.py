from src.process_image import process_all_images, process_dataset
#from src.knn_image_retrieval import retrieve_and_save_images_for_all_dataset
#from src.evaluation import evaluate_all_retrievals
#from utils.genetic_algorithm_utils import (
#    run_genetic_algorithm,
#)

if __name__ == "__main__":
    # =============== Image Feature Generation ===============
    # Choose mode: 'test' for single dataset or 'all' for all datasets
    # MODE = "test" 
    # if MODE == "test":
    #     # Test with Corel-10K
    #     print("Running in TEST MODE - Processing Corel-10K only (10000 images)")
    #     process_dataset("datasets/Corel-10K", "Corel-10K", resume=True)
    # elif MODE == "all":
    #     # Process all datasets EXCEPT Corel-1K (already tested)
    #     print("Running in FULL MODE - Processing all datasets except Corel-1K (34,088 images)")
    #     print("This will process: Corel-10K, GHIM-10K, Produce-1400, Olivia-2688\n")
    #     process_all_images(exclude=["Corel-1K"])
    #=========================================================
    
    # =============== Retrieval and Evaluation ===============
    # After features are generated, uncomment these:
    # Retrieve and save images for each image in the dataset
    # retrieve_and_save_images_for_all_dataset()

    # Evaluate the retrieval performance
    #evaluate_all_retrievals()
    #=========================================================

    # =============== Genetic Algorithm for Feature Selection ===============
    # Run the genetic algorithm for feature selection
    # run_genetic_algorithm()
    pass
    #=========================================================================