import os
import cv2
import re
import pandas as pd
from pathlib import Path
from src.utils.image_utils import (
    read_and_resize_image,
    convert_image_to_color_spaces,
    natural_sort_key,
)
from src.utils.histogram_utils import (
    compute_histogram,
    compute_2d_histogram,
    compute_lbp_histogram,
    append_histogram_to_csv,
)
from src.utils.constants import (
    HISTOGRAM_OUTPUT_PATH,
    IMAGE_FILE_EXTENSION,
    IMAGE_FILE_EXTENSIONS_SUPPORTED,
    COLOR_SPACES,
    ONE_D_HISTOGRAMS_DIR,
    TWO_D_HISTOGRAMS_DIR,
    INTRA_COLORSPACE_DIR,
    INTER_COLORSPACE_DIR,
)

def get_all_datasets(exclude=None):
    """
    Discovers all available datasets in the datasets directory.
    Handles both flat structures (like GHIM-10K) and nested structures (like Corel-1K/Corel-1K).
    
    Args:
        exclude (list): List of dataset names to exclude from processing.
    
    Returns:
        list: List of absolute paths to dataset directories.
    """
    if exclude is None:
        exclude = []
    
    datasets_dir = "datasets"
    all_datasets = []
    
    if not os.path.exists(datasets_dir):
        print(f"Datasets directory not found: {datasets_dir}")
        return all_datasets
    
    for item in os.listdir(datasets_dir):
        item_path = os.path.join(datasets_dir, item)
        if os.path.isdir(item_path):
            # Skip excluded datasets
            if item in exclude:
                print(f"Skipping excluded dataset: {item}")
                continue
            
            # Check if it's a dataset directory (contains subdirectories or images)
            contents = os.listdir(item_path)
            if contents:
                # Check if there's a nested folder with the same name
                nested_path = os.path.join(item_path, item)
                if os.path.isdir(nested_path):
                    # Use the nested path (e.g., Corel-1K/Corel-1K)
                    all_datasets.append(nested_path)
                # If it has subdirectories, use it as dataset root
                elif any(os.path.isdir(os.path.join(item_path, c)) for c in contents):
                    all_datasets.append(item_path)
                # If it has images directly, use it as dataset
                elif any(c.endswith(IMAGE_FILE_EXTENSIONS_SUPPORTED) for c in contents):
                    all_datasets.append(item_path)
    
    return sorted(all_datasets)


def get_images_from_dataset(dataset_path):
    """
    Retrieves all image files from a dataset, handling both flat and nested structures.
    
    Args:
        dataset_path (str): Path to the dataset directory.
    
    Returns:
        tuple: (list of image paths, list of relative image identifiers)
    """
    image_paths = []
    image_identifiers = []
    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(IMAGE_FILE_EXTENSIONS_SUPPORTED):
                full_path = os.path.join(root, file)
                # Create identifier relative to dataset root
                relative_path = os.path.relpath(full_path, dataset_path)
                image_paths.append(full_path)
                image_identifiers.append(relative_path)
    
    return image_paths, image_identifiers



def process_image(image_path, image_identifier, csv_filename, dataset_path):
    """
    Processes a single image: reads, converts to color spaces, computes histograms,
    and appends the histogram to a CSV file.

    Args:
        image_path (str): Full path to the image file.
        image_identifier (str): Identifier for the image (relative path or filename).
        csv_filename (str): Path to the CSV file to append histogram data.
        dataset_path (str): Path to the dataset directory (for relative path calculations).
    """
    try:
        # Read the image and convert to different color spaces
        image = read_and_resize_image(image_path)
        color_space_images = convert_image_to_color_spaces(image)

        # Create directory for saving histograms based on the image identifier
        image_folder = os.path.join(
            HISTOGRAM_OUTPUT_PATH, os.path.splitext(image_identifier)[0]
        )
        os.makedirs(image_folder, exist_ok=True)

        # Create directories for 1D and 2D histograms
        one_d_histograms_folder = os.path.join(image_folder, ONE_D_HISTOGRAMS_DIR)
        two_d_histograms_folder = os.path.join(image_folder, TWO_D_HISTOGRAMS_DIR)

        os.makedirs(one_d_histograms_folder, exist_ok=True)
        os.makedirs(two_d_histograms_folder, exist_ok=True)

        # Create folders for color spaces within 1D histogram folder
        for color_space in COLOR_SPACES.values():
            os.makedirs(
                os.path.join(one_d_histograms_folder, color_space), exist_ok=True
            )

        # Create folders for intra and inter color space 2D histograms
        intra_colorspace_folder = os.path.join(
            two_d_histograms_folder, INTRA_COLORSPACE_DIR
        )
        inter_colorspace_folder = os.path.join(
            two_d_histograms_folder, INTER_COLORSPACE_DIR
        )

        os.makedirs(intra_colorspace_folder, exist_ok=True)
        os.makedirs(inter_colorspace_folder, exist_ok=True)

        for color_space in COLOR_SPACES.values():
            os.makedirs(
                os.path.join(intra_colorspace_folder, color_space), exist_ok=True
            )

        # Compute and save 1D histograms
        all_histograms = {}
        for color_space, color_space_image in color_space_images.items():
            histograms = compute_histogram(color_space_image, color_space)
            all_histograms[color_space] = histograms

            # Save each histogram as a JPG file
            #plot_and_save_histograms(
            #    histograms, os.path.join(one_d_histograms_folder, color_space)
            #)

        # Compute LBP histogram and save to the 1D histograms folder
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        lbp_histogram = compute_lbp_histogram(grayscale_image)

        # Save LBP histogram as an image
        lbp_histograms = {"LBP_Histogram": lbp_histogram}
        lbp_folder = os.path.join(one_d_histograms_folder, "LBP")
        os.makedirs(lbp_folder, exist_ok=True)
        #plot_and_save_histograms(lbp_histograms, lbp_folder)

        # Include LBP histogram in the combined_histograms dictionary
        all_histograms["LBP_Histogram"] = lbp_histogram

        # Compute and save 2D histograms (intra-color space)
        all_2d_histograms = {}
        for color_space in COLOR_SPACES.values():
            for channel1 in range(3):
                for channel2 in range(channel1 + 1, 3):  # Avoid redundant combinations
                    hist_key = f"{color_space}_Channel_{channel1}_vs_{color_space}_Channel_{channel2}"
                    channel1_image = color_space_images[color_space][:, :, channel1]
                    channel2_image = color_space_images[color_space][:, :, channel2]
                    hist_2d = compute_2d_histogram(
                        channel1_image, channel2_image, color_space, color_space
                    )
                    all_2d_histograms[hist_key] = hist_2d

        # Save intra-color space 2D histograms
        for color_space in COLOR_SPACES.values():
            intra_color_space_histograms = {
                k: v for k, v in all_2d_histograms.items() if f"{color_space}" in k
            }
            #plot_and_save_2d_histograms(
            #    intra_color_space_histograms,
            #    os.path.join(intra_colorspace_folder, color_space),
            #)

        # Compute and save inter-color space 2D histograms
        all_inter_2d_histograms = {}
        for cs1_idx, cs1 in enumerate(COLOR_SPACES.values()):
            for cs2_idx, cs2 in enumerate(COLOR_SPACES.values()):
                if cs1_idx < cs2_idx:  # Ensure only one combination is calculated
                    for channel1 in range(3):
                        for channel2 in range(3):
                            hist_key = (
                                f"{cs1}_Channel_{channel1}_vs_{cs2}_Channel_{channel2}"
                            )
                            channel1_image = color_space_images[cs1][:, :, channel1]
                            channel2_image = color_space_images[cs2][:, :, channel2]
                            hist_2d = compute_2d_histogram(
                                channel1_image, channel2_image, cs1, cs2
                            )
                            all_inter_2d_histograms[hist_key] = hist_2d

        # Save inter-color space 2D histograms
        for cs1_idx, cs1 in enumerate(COLOR_SPACES.values()):
            for cs2_idx, cs2 in enumerate(COLOR_SPACES.values()):
                if cs1_idx < cs2_idx:  # Avoid redundancy in inter-color space as well
                    folder_name = f"{cs1}_vs_{cs2}"
                    inter_color_space_folder_path = os.path.join(
                        inter_colorspace_folder, folder_name
                    )
                    os.makedirs(inter_color_space_folder_path, exist_ok=True)
                    inter_color_space_histograms = {
                        k: v
                        for k, v in all_inter_2d_histograms.items()
                        if f"{cs1}_Channel_" in k and f"vs_{cs2}_Channel_" in k
                    }
                    #plot_and_save_2d_histograms(
                    #    inter_color_space_histograms,
                    #    inter_color_space_folder_path,
                    #)

        # Combine histograms
        combined_histograms = {
            **all_histograms,
            **all_2d_histograms,
            **all_inter_2d_histograms,
        }

        # Append histogram data to CSV
        append_histogram_to_csv(image_identifier, combined_histograms, csv_filename)

        print(f"Processed and saved histograms for {image_identifier}")

    except Exception as e:
        print(f"Error processing {image_identifier}: {e}")


def process_dataset(dataset_path, dataset_name=None):
    """
    Processes all images in a single dataset and saves results to a CSV file.
    
    Args:
        dataset_path (str): Path to the dataset directory.
        dataset_name (str): Name of the dataset (used for CSV filename). If None, extracted from path.
    """
    if dataset_name is None:
        dataset_name = os.path.basename(dataset_path.rstrip(os.sep))
    
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_name}")
    print(f"Dataset path: {dataset_path}")
    print(f"{'='*60}\n")
    
    # Get all images from the dataset
    image_paths, image_identifiers = get_images_from_dataset(dataset_path)
    
    if not image_paths:
        print(f"No images found in {dataset_path}")
        return
    
    # Sort images
    sorted_indices = sorted(range(len(image_identifiers)), key=lambda i: natural_sort_key(image_identifiers[i]))
    image_paths = [image_paths[i] for i in sorted_indices]
    image_identifiers = [image_identifiers[i] for i in sorted_indices]
    
    # Create CSV filename for this dataset
    csv_filename = os.path.join(HISTOGRAM_OUTPUT_PATH, f"{dataset_name}_histograms.csv")
    
    # Ensure the CSV file starts fresh
    if os.path.exists(csv_filename):
        os.remove(csv_filename)
    
    print(f"Found {len(image_paths)} images to process")
    print(f"CSV output: {csv_filename}\n")
    
    for idx, (image_path, image_identifier) in enumerate(zip(image_paths, image_identifiers), 1):
        print(f"[{idx}/{len(image_paths)}] Processing {image_identifier}...")
        process_image(image_path, image_identifier, csv_filename, dataset_path)
    
    print(f"\nDataset '{dataset_name}' processing completed!")


def process_all_images(exclude=None):
    """
    Discovers and processes all available datasets in the datasets directory.
    Creates individual CSV files for each dataset.
    
    Args:
        exclude (list): List of dataset names to exclude from processing.
    """
    datasets = get_all_datasets(exclude=exclude)
    
    if not datasets:
        print("No datasets found in the 'datasets' directory!")
        return
    
    print(f"\nFound {len(datasets)} dataset(s) to process:")
    for dataset_path in datasets:
        print(f"  - {os.path.basename(dataset_path)}")
    
    failed_datasets = []
    successful_datasets = []
    
    for dataset_path in datasets:
        dataset_name = os.path.basename(dataset_path.rstrip(os.sep))
        try:
            process_dataset(dataset_path, dataset_name)
            successful_datasets.append(dataset_name)
        except Exception as e:
            print(f"\n❌ Error processing dataset '{dataset_name}': {e}")
            failed_datasets.append(dataset_name)
            continue  # Continue to next dataset even if this one fails
    
    print(f"\n{'='*60}")
    print("All datasets processing completed!")
    print(f"{'='*60}")
    print(f"\n✅ Successfully processed: {len(successful_datasets)} dataset(s)")
    for ds in successful_datasets:
        print(f"   - {ds}")
    
    if failed_datasets:
        print(f"\n❌ Failed to process: {len(failed_datasets)} dataset(s)")
        for ds in failed_datasets:
            print(f"   - {ds}")
    
    print()



